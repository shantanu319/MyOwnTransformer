import pytest
import torch

from train import build_vocab_indices, lr_factor, resolve_device


def test_resolve_device_prefers_cpu_when_disabled(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(no_cuda=True).type == "cpu"


def test_resolve_device_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device(no_cuda=False).type == "cpu"


def test_build_vocab_indices_uses_target_device():
    indices = build_vocab_indices(6, torch.device("cpu"))
    assert indices.device.type == "cpu"
    assert torch.equal(indices, torch.arange(6))


def test_lr_factor_warmup_grows_linearly():
    assert lr_factor(0, 1000, warmup_steps=100) == pytest.approx(0.01)
    assert lr_factor(49, 1000, warmup_steps=100) == pytest.approx(0.5)
    assert lr_factor(99, 1000, warmup_steps=100) == pytest.approx(1.0)


def test_lr_factor_post_warmup_starts_at_peak():
    # First post-warmup step should be at peak (cosine(0) = 1)
    assert lr_factor(100, 1000, warmup_steps=100) == pytest.approx(1.0)


def test_lr_factor_decays_toward_min_ratio():
    # Final step should be near min_lr_ratio
    assert lr_factor(999, 1000, warmup_steps=100, min_lr_ratio=0.1) == pytest.approx(0.1, abs=0.01)


def test_lr_factor_never_below_min_ratio():
    # Clamped past total_steps
    assert lr_factor(5000, 1000, warmup_steps=100, min_lr_ratio=0.1) == pytest.approx(0.1)
