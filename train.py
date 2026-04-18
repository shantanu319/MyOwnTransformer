import os
import math
import random
import time

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from config import parse_args
from data import load_tinystories, data_feeder
from model import get_model, nopeak_mask


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


def train_model(model, opt):
    print("training model...")
    model.train()
    training_perplexities = []
    validation_perplexities = []

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

        train_loss = epoch_loss / epoch_tokens
        train_pplx = math.exp(train_loss)
        training_perplexities.append(train_pplx)
        print(f"Epoch {epoch+1} finished: Train Perplexity = {train_pplx:.2f}")

        # Validate at the end of each epoch:
        valid_pplx = validate_model(model, opt)
        validation_perplexities.append(valid_pplx)

    if opt.savename:
        torch.save(model.state_dict(), opt.dir_name + opt.savename)

    return training_perplexities, validation_perplexities


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

    pplx = math.exp(total_loss / total_tokens)
    print(f"Validation Perplexity = {pplx:.2f}")
    return pplx


def plot_learning_curves(train_perplexities, valid_perplexities):
    plt.figure(figsize=(10, 6))
    plt.plot(train_perplexities, label='Training')
    plt.plot(valid_perplexities, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
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

    random.seed(10)

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

    opt.savename = "/weights"

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.train = load_tinystories(tokenizer, split='train', max_docs=opt.max_docs)
    opt.valid = load_tinystories(tokenizer, split='validation', max_docs=max(10, opt.max_docs // 10))
    opt.test = opt.valid

    opt.vocab_size = 50257
    opt.indices = build_vocab_indices(opt.vocab_size, opt.device)

    model = get_model(opt, opt.vocab_size)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {params}')

    opt.optimizers = make_optimizers(model, muon_lr=opt.muon_lr, adamw_lr=opt.lr)
    batches_per_epoch = max(1, len(opt.train) // (opt.batchsize * opt.seqlen))
    opt.total_steps = max(1, opt.epochs * batches_per_epoch)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0

    train_pplx, valid_pplx = train_model(model, opt)
    plot_learning_curves(train_pplx, valid_pplx)
    test_model(model, opt, -1)


if __name__ == "__main__":
    main()
