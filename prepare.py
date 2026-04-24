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
import multiprocessing as mp
import os

import numpy as np
from datasets import load_dataset

from data import BIN_DTYPE
from tokenizer import BPETokenizer


EOS_TOKEN = '<|endoftext|>'

# Worker-process state, populated once per worker by _init_worker.
_WORKER_TOKENIZER = None
_WORKER_EOS_ID = None


def _init_worker(tokenizer_path, eos_id):
    global _WORKER_TOKENIZER, _WORKER_EOS_ID
    _WORKER_TOKENIZER = BPETokenizer()
    _WORKER_TOKENIZER.load(tokenizer_path)
    _WORKER_EOS_ID = eos_id


def _worker_encode(text):
    ids = _WORKER_TOKENIZER.encode_ordinary(text)
    ids.append(_WORKER_EOS_ID)
    arr = np.array(ids, dtype=BIN_DTYPE)
    if arr.size and arr.max() >= np.iinfo(BIN_DTYPE).max:
        raise ValueError(f"token id {arr.max()} exceeds {BIN_DTYPE} range")
    return arr


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
    if arr.size and arr.max() >= np.iinfo(BIN_DTYPE).max:
        raise ValueError(f"token id {arr.max()} exceeds {BIN_DTYPE} range")
    return arr


def _iter_encoded(tokenizer, tokenizer_path, dataset, eos_id, max_docs, num_workers):
    """Yield (doc_index, encoded_array) preserving doc order.

    Uses a process pool when num_workers > 1; falls back to in-process
    serial encoding otherwise."""
    def texts():
        for i, row in enumerate(dataset):
            if max_docs is not None and i >= max_docs:
                break
            yield row['text']

    if num_workers <= 1:
        for i, row in enumerate(dataset):
            if max_docs is not None and i >= max_docs:
                break
            yield i, _encode_row(tokenizer, row, eos_id)
        return

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers,
                  initializer=_init_worker,
                  initargs=(tokenizer_path, eos_id)) as pool:
        for i, arr in enumerate(pool.imap(_worker_encode, texts(), chunksize=64)):
            yield i, arr


def _tokenize_split_to_bin(tokenizer, tokenizer_path, dataset, eos_id, out_path,
                           max_docs=None, num_workers=1):
    token_count = 0
    with open(out_path, 'wb') as f:
        for i, arr in _iter_encoded(tokenizer, tokenizer_path, dataset, eos_id,
                                     max_docs, num_workers):
            f.write(arr.tobytes())
            token_count += len(arr)
            if (i + 1) % 10000 == 0:
                print(f"  tokenized {i + 1} docs, {token_count:,} tokens")
    return token_count


def _tokenize_split_two_bins(tokenizer, tokenizer_path, dataset, eos_id,
                             val_path, test_path, max_docs=None, num_workers=1):
    """Stream a dataset to two .bin shards by alternating doc index
    (even -> val, odd -> test). Gives a deterministic ~50/50 split."""
    val_tokens = 0
    test_tokens = 0
    with open(val_path, 'wb') as vf, open(test_path, 'wb') as tf:
        for i, arr in _iter_encoded(tokenizer, tokenizer_path, dataset, eos_id,
                                     max_docs, num_workers):
            if i % 2 == 0:
                vf.write(arr.tobytes())
                val_tokens += len(arr)
            else:
                tf.write(arr.tobytes())
                test_tokens += len(arr)
    return val_tokens, test_tokens


def _tokenize_stream_three_bins(
    tokenizer, tokenizer_path, dataset, eos_id, train_path, val_path, test_path,
    max_docs=None, holdout_period=500, num_workers=1,
):
    """Single-pass streaming tokenize into three .bin shards.

    Doc index i routes as: i % holdout_period == 0 -> val,
    == 1 -> test, else -> train. Deterministic, no second pass,
    and works for datasets that ship a single split only."""
    train_tokens = 0
    val_tokens = 0
    test_tokens = 0
    with open(train_path, 'wb') as trf, open(val_path, 'wb') as vf, open(test_path, 'wb') as tf:
        for i, arr in _iter_encoded(tokenizer, tokenizer_path, dataset, eos_id,
                                     max_docs, num_workers):
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
            if (i + 1) % 10000 == 0:
                total = train_tokens + val_tokens + test_tokens
                print(f"  tokenized {i + 1} docs, {total:,} tokens total")
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
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Process pool size for tokenize-to-bin. Default 1 (serial) —'
                             ' the per-chunk encode cache makes serial competitive on'
                             ' small/repetitive corpora; bump for large jobs (e.g. cosmopedia'
                             ' at 32k vocab on a many-core box).')
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

    print(f"Tokenizing with num_workers={args.num_workers}...")
    if spec['has_validation_split']:
        train_ds = _load_split(spec, 'train')
        val_ds = _load_split(spec, 'validation')

        print("Tokenizing train split...")
        n_train = _tokenize_split_to_bin(
            tokenizer, tok_path, train_ds, eos_id, train_path,
            max_docs=args.max_train_docs, num_workers=args.num_workers,
        )
        print(f"  wrote {n_train:,} tokens to {train_path}")

        print("Tokenizing validation split into val + test (50/50 by doc index)...")
        n_val, n_test = _tokenize_split_two_bins(
            tokenizer, tok_path, val_ds, eos_id, val_path, test_path,
            max_docs=args.max_eval_docs, num_workers=args.num_workers,
        )
        print(f"  wrote {n_val:,} tokens to {val_path}")
        print(f"  wrote {n_test:,} tokens to {test_path}")
    else:
        train_ds = _load_split(spec, 'train')
        print(f"Tokenizing {args.corpus} train stream into train/val/test "
              f"(holdout 2-in-{args.holdout_period})...")
        n_train, n_val, n_test = _tokenize_stream_three_bins(
            tokenizer, tok_path, train_ds, eos_id, train_path, val_path, test_path,
            max_docs=args.max_train_docs, holdout_period=args.holdout_period,
            num_workers=args.num_workers,
        )
        print(f"  wrote {n_train:,} tokens to {train_path}")
        print(f"  wrote {n_val:,} tokens to {val_path}")
        print(f"  wrote {n_test:,} tokens to {test_path}")


if __name__ == '__main__':
    main()
