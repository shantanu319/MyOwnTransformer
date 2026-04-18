import os
import math
import random
import time

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from config import parse_args
from data import data_feeder, load_bin
from model import get_model, nopeak_mask
from tokenizer import BPETokenizer


def resolve_device(no_cuda):
    use_cuda = (not no_cuda) and torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


def build_vocab_indices(vocab_size, device):
    return torch.arange(vocab_size, device=device)


def make_optimizers(model, muon_lr=0.02, adamw_lr=3e-4):
    embedding_weight = model.decoder.embed.embed.weight
    muon_params, adamw_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p is embedding_weight or p.ndim < 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    muon = torch.optim.Muon(muon_params, lr=muon_lr)
    adamw = torch.optim.AdamW(
        adamw_params, lr=adamw_lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    for opt in (muon, adamw):
        for group in opt.param_groups:
            group['peak_lr'] = group['lr']
    return [muon, adamw]


def lr_factor(step, total_steps, warmup_steps=100, min_lr_ratio=0.1):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def apply_lr_schedule(optimizers, step, total_steps, warmup_steps):
    factor = lr_factor(step, total_steps, warmup_steps=warmup_steps)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['peak_lr'] * factor


def save_checkpoint(model, optimizers, step, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizers': [o.state_dict() for o in optimizers],
    }, path)


def _checkpoint_path(opt, tag):
    base = opt.savename or 'ckpt'
    return os.path.join(opt.dir_name, f'{base}_{tag}.pt')


def train_model(model, opt):
    print("training model...")
    model.train()
    training_losses = []
    validation_losses = []

    step = 0
    for epoch in range(opt.epochs):
        epoch_loss = 0
        epoch_tokens = 0
        iter = 0

        for inX, out in data_feeder(opt.train, opt.batchsize, opt.seqlen, opt.device):
            iter += 1
            apply_lr_schedule(opt.optimizers, step, opt.total_steps, opt.warmup_steps)

            mask = nopeak_mask(inX.size(1), opt.device)
            with torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
                pred = model(inX, mask)
                pred = pred.view(-1, opt.vocab_size)
                out = out.reshape(-1)
                loss = F.cross_entropy(pred, out)

            epoch_loss += loss.item() * out.size(0)
            epoch_tokens += out.size(0)

            for o in opt.optimizers:
                o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.norm)
            for o in opt.optimizers:
                o.step()

            if iter % opt.printevery == 0:
                current_pplx = math.exp(loss.item())
                print(f"Epoch {epoch+1} | iter {iter} | step {step} | Loss: {loss.item():.4f} | pplx: {current_pplx:.2f}")

            step += 1
            if opt.save_every and step % opt.save_every == 0:
                path = _checkpoint_path(opt, f'step{step}')
                save_checkpoint(model, opt.optimizers, step, path)
                print(f"Saved checkpoint: {path}")

        train_loss = epoch_loss / epoch_tokens
        training_losses.append(train_loss)
        print(f"Epoch {epoch+1} finished: Train Loss = {train_loss:.4f}")

        # Validate at the end of each epoch:
        valid_loss = validate_model(model, opt)
        validation_losses.append(valid_loss)

    final_path = _checkpoint_path(opt, 'final')
    save_checkpoint(model, opt.optimizers, step, final_path)
    print(f"Saved final checkpoint: {final_path}")

    return training_losses, validation_losses


def validate_model(model, opt):
    print("validating model...")
    model.eval()  # Set to evaluation mode so dropout, etc. are disabled
    total_loss = 0
    total_tokens = 0

    with torch.no_grad(), torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
        for inX, out in data_feeder(opt.valid, opt.batchsize, opt.seqlen, opt.device):
            mask = nopeak_mask(inX.size(1), opt.device)
            pred = model(inX, mask)
            pred = pred.view(-1, opt.vocab_size)
            out = out.reshape(-1)
            loss = F.cross_entropy(pred, out)
            total_loss += loss.item() * out.size(0)
            total_tokens += out.size(0)

    avg_loss = total_loss / total_tokens
    print(f"Validation Loss = {avg_loss:.4f}")
    return avg_loss


def plot_learning_curves(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training')
    plt.plot(valid_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.show()


def test_model(model, opt, epoch):
    print("testing model...")
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad(), torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
        for x_in, x_out in data_feeder(opt.test, opt.batchsize, opt.seqlen, opt.device):
            mask = nopeak_mask(x_in.size(1), opt.device)
            preds = model(x_in, mask)
            preds = preds.view(-1, opt.vocab_size)
            x_out = x_out.reshape(-1)
            loss = F.cross_entropy(preds, x_out)
            total_loss += loss.item() * x_out.size(0)
            total_tokens += x_out.size(0)

    pplx = math.exp(total_loss / total_tokens)
    print(f"Epoch {epoch+1}: Perplexity = {pplx:.2f}")

    return pplx


def main():

    random.seed(42)

    opt = parse_args()
    opt.verbose = False

    opt.device = resolve_device(opt.no_cuda)

    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    opt.dir_name = dir_name
    opt.log_file = dir_name + "log_file.txt"

    print(str(opt))

    tok_path = os.path.join(opt.data_dir, 'tokenizer.json')
    train_bin = os.path.join(opt.data_dir, 'train.bin')
    val_bin = os.path.join(opt.data_dir, 'val.bin')
    for p in (tok_path, train_bin, val_bin):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {p} — run `python prepare.py --output-dir {opt.data_dir}` first"
            )

    tokenizer = BPETokenizer()
    tokenizer.load(tok_path)
    opt.tokenizer = tokenizer
    opt.vocab_size = tokenizer.vocab_size

    opt.train = load_bin(train_bin)
    opt.valid = load_bin(val_bin)
    opt.test = opt.valid

    model = get_model(opt, opt.vocab_size)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {params}')

    opt.optimizers = make_optimizers(model, muon_lr=opt.muon_lr, adamw_lr=opt.lr)
    batches_per_epoch = max(1, len(opt.train) // (opt.batchsize * opt.seqlen))
    opt.total_steps = max(1, opt.epochs * batches_per_epoch)

    train_losses, valid_losses = train_model(model, opt)
    plot_learning_curves(train_losses, valid_losses)
    test_model(model, opt, -1)


if __name__ == "__main__":
    main()
