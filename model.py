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


def attention(q, k, v, mask=None, dropout_p=0.0):
    # Wrap SDPA so we keep the (1, T, S) bool-mask convention used by nopeak_mask.
    if mask is not None and mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)


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

        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, q, k, v, mask=None, start_pos=None):

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
        pos = start_pos if start_pos is not None else 0
        q = apply_rope(q, self.rope_cos[pos:pos+T], self.rope_sin[pos:pos+T])
        k = apply_rope(k, self.rope_cos[pos:pos+T], self.rope_sin[pos:pos+T])

        if start_pos is not None:
            if self.k_cache is None:
                max_len = self.rope_cos.size(0)
                self.k_cache = torch.zeros(bs, self.h, max_len, self.d_k,
                                           device=q.device, dtype=q.dtype)
                self.v_cache = torch.zeros(bs, self.h, max_len, self.d_k,
                                           device=q.device, dtype=q.dtype)
            self.k_cache[:, :, start_pos:start_pos+T] = k
            self.v_cache[:, :, start_pos:start_pos+T] = v
            k = self.k_cache[:, :, :start_pos+T]
            v = self.v_cache[:, :, :start_pos+T]

        dropout_p = self.dropout.p if self.training else 0.0
        scores = attention(q, k, v, mask, dropout_p)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


def _round_to_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            # Match a 4*d_model 2-matmul FFN param budget with 3 matmuls: 8/3*d_model.
            d_ff = _round_to_multiple(8 * d_model // 3, 64)
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

    def forward(self, x, mask, start_pos=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask, start_pos=start_pos))
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

    def forward(self, trg, mask, start_pos=None):
        x = self.embed(trg)
        for i in range(self.N):
            x = self.layers[i](x, mask, start_pos=start_pos)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout):
        super().__init__()
        self.decoder = Decoder(vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, vocab)
        self.out.weight = self.decoder.embed.embed.weight

    def forward(self, vocab, mask, start_pos=None):
        d_output = self.decoder(vocab, mask, start_pos=start_pos)
        output = self.out(d_output)
        return output

    def reset_cache(self):
        for layer in self.decoder.layers:
            layer.attn_1.reset_cache()


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
