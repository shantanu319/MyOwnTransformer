"""Tests for prepare.py — focus on parity between serial and parallel encode."""
import numpy as np

from data import BIN_DTYPE
from prepare import _iter_encoded, _tokenize_stream_three_bins
from tokenizer import BPETokenizer


CORPUS = (
    "the quick brown fox jumps over the lazy dog "
    "hello world hello there hello friends "
    "lorem ipsum dolor sit amet consectetur " * 30
)

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world",
    "lorem ipsum dolor sit amet",
    "the quick brown",
    "consectetur adipiscing elit",
    "",  # empty doc edge case
    "single",
    "punctuation, and -- some! numbers 42 too",
] * 6  # 48 docs


def _train_tokenizer(tmp_path):
    eos_id = 511
    tok = BPETokenizer(special_tokens={'<|endoftext|>': eos_id})
    tok.train(CORPUS, vocab_size=400)
    path = tmp_path / 'tok.json'
    tok.save(str(path))
    return tok, str(path), eos_id


def _rows():
    return [{'text': t} for t in TEXTS]


def test_iter_encoded_parallel_matches_serial(tmp_path):
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)

    serial = [(i, arr.tolist()) for i, arr in
              _iter_encoded(tok, tok_path, _rows(), eos_id, max_docs=None, num_workers=1)]

    parallel = [(i, arr.tolist()) for i, arr in
                _iter_encoded(tok, tok_path, _rows(), eos_id, max_docs=None, num_workers=2)]

    assert serial == parallel
    assert len(serial) == len(TEXTS)
    # And every encoded doc ends with the EOS id.
    for _, ids in serial:
        assert ids[-1] == eos_id


def test_iter_encoded_respects_max_docs(tmp_path):
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)
    out = list(_iter_encoded(tok, tok_path, _rows(), eos_id, max_docs=5, num_workers=2))
    assert [i for i, _ in out] == [0, 1, 2, 3, 4]


def test_three_bins_parallel_byte_equal_to_serial(tmp_path):
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)
    holdout = 5  # forces several rows into val + test buckets

    s_train = tmp_path / 's_train.bin'
    s_val = tmp_path / 's_val.bin'
    s_test = tmp_path / 's_test.bin'
    s_n = _tokenize_stream_three_bins(
        tok, tok_path, _rows(), eos_id, str(s_train), str(s_val), str(s_test),
        holdout_period=holdout, num_workers=1,
    )

    p_train = tmp_path / 'p_train.bin'
    p_val = tmp_path / 'p_val.bin'
    p_test = tmp_path / 'p_test.bin'
    p_n = _tokenize_stream_three_bins(
        tok, tok_path, _rows(), eos_id, str(p_train), str(p_val), str(p_test),
        holdout_period=holdout, num_workers=4,
    )

    assert s_n == p_n
    assert s_train.read_bytes() == p_train.read_bytes()
    assert s_val.read_bytes() == p_val.read_bytes()
    assert s_test.read_bytes() == p_test.read_bytes()
