# -*- coding: utf-8 -*-
import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast

def read_corpus(filename,tokenizer): 
    seq = []
    with open(filename,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

class Embedder(nn.Module): 
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module): 
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module): 
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

def attention_cosine(q, k, v, mask=None, dropout=None):
    # q, k shape: [batch_size, heads, seq_len, d_k]
    # Compute dot product: shape [batch_size, heads, seq_len, seq_len]
    dot = torch.matmul(q, k.transpose(-2, -1))

    # Compute norms for q and k
    q_norm = torch.norm(q, dim=-1, keepdim=True)  # shape: [batch_size, heads, seq_len, 1]
    k_norm = torch.norm(k, dim=-1, keepdim=True)  # shape: [batch_size, heads, seq_len, 1]

    # Compute pairwise product of norms: shape becomes [batch_size, heads, seq_len, seq_len]
    denom = q_norm * k_norm.transpose(-2, -1) + 1e-8  # add epsilon to avoid division by zero

    # Calculate cosine similarity scores
    scores = dot / denom

    if mask is not None:
        mask = mask.unsqueeze(1)  # Broadcast mask for each head
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to obtain attention weights
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    # Multiply by v to get the final output
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module): 
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = attention_cosine(q, k, v, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class FeedForward(nn.Module): 
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def get_clones(module, N): 
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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

# class EncoderLayer(nn.Module): # not needed since GPT2 is decoder only
#     def __init__(self, d_model, heads, dropout=0.1):
#         super().__init__()
#         self.norm_1 = Norm(d_model)
#         self.norm_2 = Norm(d_model)
#         self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
#         self.ff = FeedForward(d_model, dropout=dropout)
#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)

#     def forward(self, x, mask):
#         x2 = self.norm_1(x)
#         x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
#         x2 = self.norm_2(x)
#         x = x + self.dropout_2(self.ff(x2))
#         return x

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module): # deleted any reference to encoder outputs
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        # self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        # self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, mask): # can remove e_outputs and src_mask since we only need one mask
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        # get rid of self.attn_2 and self.dropout_2 which connect to encoder
        # x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        # src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

# class Encoder(nn.Module): # not needed since GPT2 is decoder only
#     def __init__(self, vocab_size, d_model, N, heads, dropout):
#         super().__init__()
#         self.N = N
#         self.embed = Embedder(vocab_size, d_model)
#         self.pe = PositionalEncoder(d_model, dropout=dropout)
#         self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
#         self.norm = Norm(d_model)
#     def forward(self, src, mask):
#         x = self.embed(src)
#         x = self.pe(x)
#         for i in range(self.N):
#             x = self.layers[i](x, mask)
#         return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, mask): # removed e_outputs and merged src_mask and trg_mask
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask) # removed e_outputs
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout): #trg_vocab and src_vocab merged to vocab
        super().__init__()
        # self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, vocab)
        self.out.weight = self.decoder.embed.embed.weight

    def forward(self, vocab, mask): # merged src and trg masks
        # e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(vocab, mask) # removed e_outputs and src_mask
        output = self.out(d_output)
        return output

def get_model(opt, vocab):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)

    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model

def data_feeder(data, batch_size, seq_len, device):
    total = len(data)
    num_sequences = total // seq_len
    data = data[:num_sequences*seq_len]
    data = torch.tensor(data, dtype=torch.long, device=device)

    data = data.view(num_sequences, seq_len)

    for i in range(0, num_sequences, batch_size):
        x = data[i:i+batch_size] 
        if x.size(0) < batch_size:
            break
        yield x[:, :-1], x[:, 1:] 


def nopeak_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).unsqueeze(0)
    mask = (mask == 0)
    return mask

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

    return training_perplexities,validation_perplexities

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

    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=16)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str)
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)

    opt, unknown = parser.parse_known_args()
    opt.verbose = False

    # opt.device = 0 if opt.no_cuda is False else -1
    opt.device = torch.device("cuda:0" if (not opt.no_cuda and torch.cuda.is_available()) else "cpu")

    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # source_name = sys.argv[0]
    # dir_name = dir_name + "//"
    opt.dir_name = dir_name
    # shutil.copy(source_name,dir_name + source_name)
    opt.log_file = dir_name + "log_file.txt"

    print(str(opt))

    opt.savename = "/weights"

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.train = read_corpus('wiki2.train.txt',tokenizer)
    opt.valid = read_corpus('wiki2.valid.txt',tokenizer)
    opt.test = read_corpus('wiki2.test.txt',tokenizer)

    obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()

    model = get_model(opt,opt.vocab_size) # cut params down to vocab_size and opt

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

    train_pplx,valid_pplx = train_model(model,opt)
    plot_learning_curves(train_pplx, valid_pplx)
    test_model(model,opt,-1)

if __name__ == "__main__":
    main()



