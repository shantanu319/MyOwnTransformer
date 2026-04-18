import numpy as np
import torch

from data import BIN_DTYPE, data_feeder, load_bin


def test_data_feeder_batch_shapes():
    data = list(range(100))
    batches = list(data_feeder(data, batch_size=2, seq_len=8, device=torch.device("cpu")))
    assert len(batches) > 0
    for x, y in batches:
        assert x.shape == (2, 7)
        assert y.shape == (2, 7)


def test_data_feeder_target_is_input_shifted_by_one():
    data = list(range(100))
    for x, y in data_feeder(data, batch_size=2, seq_len=8, device=torch.device("cpu")):
        # y at position t should equal x at position t+1 within the same sequence
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_data_feeder_drops_incomplete_last_batch():
    # 13 sequences of length 8 with batch_size=4 → 3 full batches, last partial dropped
    data = list(range(13 * 8))
    batches = list(data_feeder(data, batch_size=4, seq_len=8, device=torch.device("cpu")))
    assert len(batches) == 3
    for x, _ in batches:
        assert x.size(0) == 4


def test_load_bin_roundtrip(tmp_path):
    # Write uint16 tokens to disk; load_bin should return them mmapped.
    path = tmp_path / "toy.bin"
    arr = np.arange(100, dtype=BIN_DTYPE)
    arr.tofile(path)

    loaded = load_bin(str(path))
    assert loaded.dtype == BIN_DTYPE
    assert loaded.shape == (100,)
    assert np.array_equal(loaded, arr)


def test_data_feeder_accepts_mmap(tmp_path):
    path = tmp_path / "toy.bin"
    arr = np.arange(100, dtype=BIN_DTYPE)
    arr.tofile(path)
    mmap_data = load_bin(str(path))

    batches = list(data_feeder(mmap_data, batch_size=2, seq_len=8, device=torch.device("cpu")))
    assert len(batches) > 0
    for x, y in batches:
        assert x.shape == (2, 7)
        assert y.shape == (2, 7)
        assert torch.equal(x[:, 1:], y[:, :-1])
