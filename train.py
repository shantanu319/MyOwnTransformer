import os
import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from config import parse_args
from data import read_corpus, data_feeder
from model import get_model, nopeak_mask


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


def train_model(model, opt):
    print("training model...")
    model.train()
    training_perplexities = []
    validation_perplexities = []

    for epoch in range(opt.epochs):
        epoch_loss = 0
        epoch_tokens = 0
        iter = 0

        for inX, out in data_feeder(opt.train, opt.batchsize, opt.seqlen, opt.device):
            iter += 1
            mask = nopeak_mask(inX.size(1), opt.device)
            pred = model(inX, mask)
            pred = pred.view(-1, opt.vocab_size)
            out = out.reshape(-1)

            loss = F.cross_entropy(pred, out)
            epoch_loss += loss.item() * out.size(0)
            epoch_tokens += out.size(0)

            opt.optimizer.zero_grad()
            loss.backward()
            opt.optimizer.step()

            if iter % opt.printevery == 0:
                current_pplx = math.exp(loss.item())
                print(f"Epoch {epoch+1} | iter {iter} | Loss: {loss.item():.4f} | pplx: {current_pplx:.2f}")

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

    # Use no_grad() to prevent gradient computations during validation
    with torch.no_grad():
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

    # write code to generate perplexity of test set
    with torch.no_grad():
        for x_in, x_out in data_feeder(opt.test, opt.batchsize, opt.seqlen, opt.device):
            mask = nopeak_mask(x_in.size(1), opt.device)

            preds = model(x_in, mask)
            preds = preds.view(-1, opt.vocab_size)
            x_out = x_out.reshape(-1)

            loss = F.cross_entropy(preds, x_out)

            batch_tokens = x_out.size(0)
            total_loss += loss.item() * x_out.size(0)
            total_tokens += x_out.size(0)

    pplx = math.exp(total_loss / total_tokens)
    print(f"Epoch {epoch+1}: Perplexity = {pplx:.2f}")

    return pplx


def main():

    random.seed(10)

    opt = parse_args()
    opt.verbose = False

    opt.device = torch.device("cuda:0" if (not opt.no_cuda and torch.cuda.is_available()) else "cpu")

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
    opt.train = read_corpus('wiki2.train.txt', tokenizer)
    opt.valid = read_corpus('wiki2.valid.txt', tokenizer)
    opt.test = read_corpus('wiki2.test.txt', tokenizer)

    obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()

    model = get_model(opt, opt.vocab_size)  # cut params down to vocab_size and opt

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    text = 'total params: %d' % (params)
    print(text)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

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
