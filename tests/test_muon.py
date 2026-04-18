import pytest
import torch

from muon import Muon


def test_muon_rejects_1d_params():
    p = torch.nn.Parameter(torch.randn(10))
    opt = Muon([p])
    p.grad = torch.randn(10)
    with pytest.raises(RuntimeError):
        opt.step()


def test_muon_reduces_loss_on_tall_matrix():
    """Muon should fit a linear regression Y = XW^T when W is (out, in) with out > in."""
    torch.manual_seed(0)
    d_in, d_out = 4, 16
    W_true = torch.randn(d_out, d_in)
    X = torch.randn(64, d_in)
    Y = X @ W_true.T

    model = torch.nn.Linear(d_in, d_out, bias=False)
    opt = Muon([model.weight], lr=0.1)

    initial_loss = ((model(X) - Y) ** 2).mean().item()
    for _ in range(200):
        loss = ((model(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = ((model(X) - Y) ** 2).mean().item()

    assert final_loss < initial_loss * 0.01, f"{initial_loss:.4f} -> {final_loss:.4f}"


def test_muon_reduces_loss_on_wide_matrix():
    """Exercises the Newton-Schulz transpose branch (in > out)."""
    torch.manual_seed(0)
    d_in, d_out = 16, 4
    W_true = torch.randn(d_out, d_in)
    X = torch.randn(64, d_in)
    Y = X @ W_true.T

    model = torch.nn.Linear(d_in, d_out, bias=False)
    opt = Muon([model.weight], lr=0.1)

    initial_loss = ((model(X) - Y) ** 2).mean().item()
    for _ in range(200):
        loss = ((model(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = ((model(X) - Y) ** 2).mean().item()

    assert final_loss < initial_loss * 0.01, f"{initial_loss:.4f} -> {final_loss:.4f}"
