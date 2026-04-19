"""Offline data prep: train a BPE tokenizer on TinyStories and emit .bin shards.

Run once to produce:
  {output_dir}/tokenizer.json
  {output_dir}/train.bin  (uint16 tokens, stories separated by <|endoftext|>)
  {output_dir}/val.bin

Then train.py points at {output_dir} and mmaps the .bin files directly.
"""
import argparse
import os

import numpy as np
from datasets import load_dataset

from data import BIN_DTYPE
from tokenizer import BPETokenizer


EOS_TOKEN = '<|endoftext|>'


def _bpe_training_corpus(dataset, num_docs):
    texts = []
    for i, row in enumerate(dataset):
        if i >= num_docs:
            break
        texts.append(row['text'])
    return '\n'.join(texts)


def _encode_row(tokenizer, row, eos_id):
    ids = tokenizer.encode_ordinary(row['text'])
    ids.append(eos_id)
    arr = np.array(ids, dtype=BIN_DTYPE)
    if arr.max() >= np.iinfo(BIN_DTYPE).max:
        raise ValueError(f"token id {arr.max()} exceeds {BIN_DTYPE} range")
    return arr


def _tokenize_split_to_bin(tokenizer, dataset, eos_id, out_path, max_docs=None):
    doc_count = 0
    token_count = 0
    with open(out_path, 'wb') as f:
        for row in dataset:
            if max_docs is not None and doc_count >= max_docs:
                break
            arr = _encode_row(tokenizer, row, eos_id)
            f.write(arr.tobytes())
            doc_count += 1
            token_count += len(arr)
            if doc_count % 10000 == 0:
                print(f"  tokenized {doc_count} docs, {token_count:,} tokens")
    return token_count


def _tokenize_split_two_bins(tokenizer, dataset, eos_id, val_path, test_path, max_docs=None):
    """Stream a dataset to two .bin shards by alternating doc index
    (even -> val, odd -> test). Gives a deterministic ~50/50 split."""
    doc_count = 0
    val_tokens = 0
    test_tokens = 0
    with open(val_path, 'wb') as vf, open(test_path, 'wb') as tf:
        for i, row in enumerate(dataset):
            if max_docs is not None and doc_count >= max_docs:
                break
            arr = _encode_row(tokenizer, row, eos_id)
            if i % 2 == 0:
                vf.write(arr.tobytes())
                val_tokens += len(arr)
            else:
                tf.write(arr.tobytes())
                test_tokens += len(arr)
            doc_count += 1
    return val_tokens, test_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='tinystories', choices=['tinystories'])
    parser.add_argument('--output-dir', default='data_cache/tinystories')
    parser.add_argument('--vocab-size', type=int, default=4096,
                        help='Total vocab size including special tokens')
    parser.add_argument('--bpe-train-docs', type=int, default=5000,
                        help='Number of docs to use for BPE training')
    parser.add_argument('--max-train-docs', type=int, default=None,
                        help='Cap for tokenizing train split (default: full split)')
    parser.add_argument('--max-eval-docs', type=int, default=None,
                        help='Cap for tokenizing validation split (halved into val + test)')
    args = parser.parse_args()

    assert args.vocab_size > 256, "vocab_size must leave room for base bytes"
    assert np.iinfo(BIN_DTYPE).max >= args.vocab_size - 1, \
        f"vocab_size too large for {BIN_DTYPE}"

    os.makedirs(args.output_dir, exist_ok=True)

    eos_id = args.vocab_size - 1
    tokenizer = BPETokenizer(special_tokens={EOS_TOKEN: eos_id})

    print(f"Loading {args.bpe_train_docs} docs for BPE training...")
    train_for_bpe = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
    bpe_corpus = _bpe_training_corpus(train_for_bpe, args.bpe_train_docs)
    print(f"  BPE training corpus: {len(bpe_corpus):,} chars")

    target_ordinary_vocab = args.vocab_size - len(tokenizer.special_tokens)
    print(f"Training BPE to {target_ordinary_vocab} ordinary tokens (+{len(tokenizer.special_tokens)} special)...")
    tokenizer.train(bpe_corpus, vocab_size=target_ordinary_vocab, verbose=True)

    tok_path = os.path.join(args.output_dir, 'tokenizer.json')
    tokenizer.save(tok_path)
    print(f"Saved tokenizer: {tok_path}")

    train_ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
    val_ds = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)

    train_path = os.path.join(args.output_dir, 'train.bin')
    val_path = os.path.join(args.output_dir, 'val.bin')
    test_path = os.path.join(args.output_dir, 'test.bin')

    print("Tokenizing train split...")
    n_train = _tokenize_split_to_bin(tokenizer, train_ds, eos_id, train_path, args.max_train_docs)
    print(f"  wrote {n_train:,} tokens to {train_path}")

    print("Tokenizing validation split into val + test (50/50 by doc index)...")
    n_val, n_test = _tokenize_split_two_bins(
        tokenizer, val_ds, eos_id, val_path, test_path, args.max_eval_docs
    )
    print(f"  wrote {n_val:,} tokens to {val_path}")
    print(f"  wrote {n_test:,} tokens to {test_path}")


if __name__ == '__main__':
    main()
