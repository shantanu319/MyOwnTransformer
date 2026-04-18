"""Generate text from a trained checkpoint using temperature + nucleus (top-p) sampling.

Example:
  python sample.py --checkpoint saved/model/ckpt_final.pt --prompt "Once upon a time" \
                   --max-tokens 200 --temperature 0.8 --top-p 0.9
"""
import argparse
import os

import torch
import torch.nn.functional as F

from model import Transformer, nopeak_mask
from tokenizer import BPETokenizer


def top_p_filter(probs, top_p):
    """Zero out tokens outside the smallest nucleus whose cumulative prob >= top_p.
    Always keeps at least the single highest-prob token."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    drop = cumsum >= top_p
    # Shift right so the token that first crosses the threshold is still kept.
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_idx, sorted_probs)
    return filtered


def _resolve_device(no_cuda):
    use_cuda = (not no_cuda) and torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens, temperature, top_p,
             max_context, device, eos_id=None, stop_at_eos=True):
    model.eval()

    if prompt:
        ids = tokenizer.encode(prompt)
    elif eos_id is not None:
        ids = [eos_id]  # start a fresh document
    else:
        ids = [0]

    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_tokens):
        context = tokens[:, -max_context:]
        mask = nopeak_mask(context.size(1), device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(context, mask)
        logits = logits[:, -1, :].float() / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        if top_p < 1.0:
            probs = top_p_filter(probs, top_p)
            probs = probs / probs.sum()

        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        if stop_at_eos and eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='data_cache/tinystories')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--max-context', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = _resolve_device(args.no_cuda)

    tok_path = os.path.join(args.data_dir, 'tokenizer.json')
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer not found at {tok_path}")
    tokenizer = BPETokenizer()
    tokenizer.load(tok_path)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'config' not in ckpt or ckpt['config'] is None:
        raise ValueError(
            f"checkpoint {args.checkpoint} lacks a 'config' field — "
            f"retrain with the current save_checkpoint to include model architecture"
        )
    cfg = ckpt['config']
    model = Transformer(
        vocab=cfg['vocab_size'],
        d_model=cfg['d_model'],
        N=cfg['n_layers'],
        heads=cfg['heads'],
        dropout=cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model'])

    eos_id = tokenizer.special_tokens.get('<|endoftext|>')

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- sample {i+1}/{args.num_samples} ---")
        text = generate(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_context=args.max_context,
            device=device,
            eos_id=eos_id,
        )
        print(text)


if __name__ == '__main__':
    main()
