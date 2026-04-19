"""Long-running inference server. Reads JSON-line requests from stdin and
writes JSON-line responses to stdout. Intended to be spawned by the Rust
`chat` CLI as a child process — not invoked directly by humans.

Protocol:
  Server -> client (stdout):
    {"type": "ready"}                         sent once after warm-up
    {"type": "response", "text": "..."}       after each prompt
    {"type": "reset_ok"}                      after each reset
    {"type": "error", "error": "..."}         on any failure

  Client -> server (stdin):
    {"type": "prompt", "prompt": "..."}       generate a continuation
    {"type": "reset"}                         clear running token context
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

from model import Transformer, nopeak_mask
from sample import top_p_filter
from tokenizer import BPETokenizer


def log(msg):
    # Non-protocol output goes to stderr so stdout stays machine-parseable.
    print(f"[chat_server] {msg}", file=sys.stderr, flush=True)


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _resolve_device(no_cuda):
    use_cuda = (not no_cuda) and torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


@torch.no_grad()
def generate_into(context_ids, model, eos_id, max_tokens, temperature, top_p,
                  max_context, device):
    """Append up to max_tokens to context_ids in place. Returns the newly generated ids."""
    model.eval()
    start = len(context_ids)
    for _ in range(max_tokens):
        window = context_ids[-max_context:]
        x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        mask = nopeak_mask(x.size(1), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x, mask)
        logits = logits[:, -1, :].float() / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        if top_p < 1.0:
            probs = top_p_filter(probs, top_p)
            probs = probs / probs.sum()
        next_token = torch.multinomial(probs, num_samples=1).item()
        context_ids.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break
    return context_ids[start:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-context', type=int, default=512)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = _resolve_device(args.no_cuda)
    log(f"device: {device}")

    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(args.data_dir, 'tokenizer.json'))
    log(f"tokenizer vocab_size: {tokenizer.vocab_size}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'config' not in ckpt or ckpt['config'] is None:
        _send({"type": "error", "error": "checkpoint missing 'config' field"})
        return
    cfg = ckpt['config']
    model = Transformer(
        vocab=cfg['vocab_size'],
        d_model=cfg['d_model'],
        N=cfg['n_layers'],
        heads=cfg['heads'],
        dropout=cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    log(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    eos_id = tokenizer.special_tokens.get('<|endoftext|>')
    context_ids = []

    _send({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"type": "error", "error": f"invalid json: {e}"})
            continue

        kind = msg.get("type")
        if kind == "reset":
            context_ids = []
            _send({"type": "reset_ok"})
        elif kind == "prompt":
            prompt = msg.get("prompt", "")
            try:
                context_ids.extend(tokenizer.encode(prompt))
                new_ids = generate_into(
                    context_ids, model, eos_id,
                    max_tokens=msg.get("max_tokens", args.max_tokens),
                    temperature=msg.get("temperature", args.temperature),
                    top_p=msg.get("top_p", args.top_p),
                    max_context=args.max_context,
                    device=device,
                )
                _send({"type": "response", "text": tokenizer.decode(new_ids)})
            except Exception as e:  # noqa: BLE001
                _send({"type": "error", "error": repr(e)})
        else:
            _send({"type": "error", "error": f"unknown type: {kind!r}"})


if __name__ == "__main__":
    main()
