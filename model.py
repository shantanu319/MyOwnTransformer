import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x.int())


def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (..., T, D) with D even; cos/sin: (T, D/2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x1 * sin + x2 * cos
    return torch.stack((rotated_1, rotated_2), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        cos, sin = precompute_rope_freqs(self.d_k, max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        T = q.size(2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):  # deleted any reference to encoder outputs
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = RMSNorm(d_model)
        self.norm_3 = RMSNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = SwiGLU(d_model, dropout=dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab, d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = RMSNorm(d_model)

    def forward(self, trg, mask):
        x = self.embed(trg)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout):
        super().__init__()
        self.decoder = Decoder(vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, vocab)
        self.out.weight = self.decoder.embed.embed.weight

    def forward(self, vocab, mask):
        d_output = self.decoder(vocab, mask)
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


def nopeak_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).unsqueeze(0)
    mask = (mask == 0)
    return mask
