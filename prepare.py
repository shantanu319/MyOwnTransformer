"""Offline data prep: train a BPE tokenizer and emit .bin shards.

Run once per corpus to produce:
  {output_dir}/tokenizer.json
  {output_dir}/train.bin  (uint16 tokens, docs separated by <|endoftext|>)
  {output_dir}/val.bin
  {output_dir}/test.bin

Then train.py points at {output_dir} and mmaps the .bin files directly.

Supported corpora (see CORPUS_SPECS): tinystories, cosmopedia.
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


def _tokenize_stream_three_bins(
    tokenizer, dataset, eos_id, train_path, val_path, test_path,
    max_docs=None, holdout_period=500,
):
    """Single-pass streaming tokenize into three .bin shards.

    Doc index i routes as: i % holdout_period == 0 -> val,
    == 1 -> test, else -> train. Deterministic, no second pass,
    and works for datasets that ship a single split only."""
    doc_count = 0
    train_tokens = 0
    val_tokens = 0
    test_tokens = 0
    with open(train_path, 'wb') as trf, open(val_path, 'wb') as vf, open(test_path, 'wb') as tf:
        for i, row in enumerate(dataset):
            if max_docs is not None and doc_count >= max_docs:
                break
            arr = _encode_row(tokenizer, row, eos_id)
            bucket = i % holdout_period
            if bucket == 0:
                vf.write(arr.tobytes())
                val_tokens += len(arr)
            elif bucket == 1:
                tf.write(arr.tobytes())
                test_tokens += len(arr)
            else:
                trf.write(arr.tobytes())
                train_tokens += len(arr)
            doc_count += 1
            if doc_count % 10000 == 0:
                total = train_tokens + val_tokens + test_tokens
                print(f"  tokenized {doc_count} docs, {total:,} tokens total")
    return train_tokens, val_tokens, test_tokens


CORPUS_SPECS = {
    'tinystories': {
        'path': 'roneneldan/TinyStories',
        'config': None,
        'has_validation_split': True,
    },
    'cosmopedia': {
        'path': 'HuggingFaceTB/smollm-corpus',
        'config': 'cosmopedia-v2',
        'has_validation_split': False,
    },
}


def _load_split(spec, split):
    if spec['config'] is not None:
        return load_dataset(spec['path'], spec['config'], split=split, streaming=True)
    return load_dataset(spec['path'], split=split, streaming=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='tinystories', choices=sorted(CORPUS_SPECS))
    parser.add_argument('--output-dir', default=None,
                        help='Defaults to data_cache/{corpus}')
    parser.add_argument('--vocab-size', type=int, default=8192,
                        help='Total vocab size including special tokens')
    parser.add_argument('--bpe-train-docs', type=int, default=5000,
                        help='Number of docs to use for BPE training')
    parser.add_argument('--max-train-docs', type=int, default=None,
                        help='Cap for tokenizing train split (default: full split)')
    parser.add_argument('--max-eval-docs', type=int, default=None,
                        help='Cap for tokenizing validation split (halved into val + test).'
                             ' Only used for corpora with a native validation split.')
    parser.add_argument('--holdout-period', type=int, default=500,
                        help='For corpora without a validation split, reserve 1-in-N docs each'
                             ' for val and test from the train stream.')
    args = parser.parse_args()

    assert args.vocab_size > 256, "vocab_size must leave room for base bytes"
    assert np.iinfo(BIN_DTYPE).max >= args.vocab_size - 1, \
        f"vocab_size too large for {BIN_DTYPE}"

    output_dir = args.output_dir or f'data_cache/{args.corpus}'
    os.makedirs(output_dir, exist_ok=True)

    spec = CORPUS_SPECS[args.corpus]

    eos_id = args.vocab_size - 1
    tokenizer = BPETokenizer(special_tokens={EOS_TOKEN: eos_id})

    print(f"Loading {args.bpe_train_docs} docs from {args.corpus} for BPE training...")
    train_for_bpe = _load_split(spec, 'train')
    bpe_corpus = _bpe_training_corpus(train_for_bpe, args.bpe_train_docs)
    print(f"  BPE training corpus: {len(bpe_corpus):,} chars")

    target_ordinary_vocab = args.vocab_size - len(tokenizer.special_tokens)
    print(f"Training BPE to {target_ordinary_vocab} ordinary tokens (+{len(tokenizer.special_tokens)} special)...")
    tokenizer.train(bpe_corpus, vocab_size=target_ordinary_vocab, verbose=True)

    tok_path = os.path.join(output_dir, 'tokenizer.json')
    tokenizer.save(tok_path)
    print(f"Saved tokenizer: {tok_path}")

    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    test_path = os.path.join(output_dir, 'test.bin')

    if spec['has_validation_split']:
        train_ds = _load_split(spec, 'train')
        val_ds = _load_split(spec, 'validation')

        print("Tokenizing train split...")
        n_train = _tokenize_split_to_bin(tokenizer, train_ds, eos_id, train_path, args.max_train_docs)
        print(f"  wrote {n_train:,} tokens to {train_path}")

        print("Tokenizing validation split into val + test (50/50 by doc index)...")
        n_val, n_test = _tokenize_split_two_bins(
            tokenizer, val_ds, eos_id, val_path, test_path, args.max_eval_docs
        )
        print(f"  wrote {n_val:,} tokens to {val_path}")
        print(f"  wrote {n_test:,} tokens to {test_path}")
    else:
        train_ds = _load_split(spec, 'train')
        print(f"Tokenizing {args.corpus} train stream into train/val/test "
              f"(holdout 2-in-{args.holdout_period})...")
        n_train, n_val, n_test = _tokenize_stream_three_bins(
            tokenizer, train_ds, eos_id, train_path, val_path, test_path,
            max_docs=args.max_train_docs, holdout_period=args.holdout_period,
        )
        print(f"  wrote {n_train:,} tokens to {train_path}")
        print(f"  wrote {n_val:,} tokens to {val_path}")
        print(f"  wrote {n_test:,} tokens to {test_path}")


if __name__ == '__main__':
    main()
