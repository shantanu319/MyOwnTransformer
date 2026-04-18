import pytest
import torch
import torch.nn.functional as F

from data import data_feeder, load_tinystories
from model import Transformer, nopeak_mask


@pytest.mark.slow
def test_tinystories_end_to_end():
    """Download a TinyStories slice, tokenize, train briefly, verify loss drops.

    Exercises the full pipeline: HF dataset -> tokenizer -> data_feeder ->
    model forward/backward -> optimizer step. If any link breaks, this fails.
    """
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = load_tinystories(tokenizer, split="train", max_docs=20)
    assert len(tokens) > 200, f"too few tokens: {len(tokens)}"

    V = tokenizer.vocab_size
    device = torch.device("cpu")

    torch.manual_seed(0)
    model = Transformer(V, d_model=32, N=1, heads=2, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step, (x, y) in enumerate(data_feeder(tokens, batch_size=2, seq_len=32, device=device)):
        if step >= 20:
            break
        mask = nopeak_mask(x.size(1), device)
        logits = model(x, mask)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert len(losses) >= 10, f"only {len(losses)} batches produced"
    assert all(torch.isfinite(torch.tensor(l)).item() for l in losses), f"non-finite loss: {losses}"

    mid = len(losses) // 2
    first_half_avg = sum(losses[:mid]) / mid
    second_half_avg = sum(losses[mid:]) / (len(losses) - mid)
    assert second_half_avg < first_half_avg, (
        f"loss did not drop across halves: {first_half_avg:.3f} -> {second_half_avg:.3f}"
    )
