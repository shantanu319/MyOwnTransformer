import torch

from train import build_vocab_indices, resolve_device


def test_resolve_device_prefers_cpu_when_disabled(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(no_cuda=True).type == "cpu"


def test_resolve_device_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(no_cuda=False).type == "cpu"


def test_build_vocab_indices_uses_target_device():
    indices = build_vocab_indices(6, torch.device("cpu"))
    assert indices.device.type == "cpu"
    assert torch.equal(indices, torch.arange(6))
