import json
import os
import re
from collections import Counter


# GPT-2 style pre-tokenization regex. Pattern is applied before BPE so merges
# can't cross word boundaries (punctuation, whitespace, etc.).
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""


def _get_pair_counts(tokens, counts=None):
    counts = counts if counts is not None else Counter()
    for pair in zip(tokens, tokens[1:]):
        counts[pair] += 1
    return counts


def _merge(tokens, pair, new_id):
    result = []
    i = 0
    n = len(tokens)
    while i < n:
        if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


class BPETokenizer:
    def __init__(self, pattern=None, special_tokens=None):
        self.pattern = pattern or GPT2_SPLIT_PATTERN
        self._compiled = re.compile(self.pattern)
        self.special_tokens = dict(special_tokens or {})
        self.merges = {}  # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._chunk_cache = {}  # str chunk -> tuple of token ids

    def __getstate__(self):
        # Don't ship the chunk cache through pickle (e.g. multiprocessing) —
        # workers should accumulate their own cache.
        state = self.__dict__.copy()
        state['_chunk_cache'] = {}
        return state

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per byte)"
        num_merges = vocab_size - 256

        # Pre-tokenize and dedupe: identical chunks share a merge schedule, so
        # we only merge each unique chunk once per step, weighted by frequency.
        chunk_freq = Counter(self._compiled.findall(text))
        chunks = {tuple(c.encode('utf-8')): f for c, f in chunk_freq.items()}

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._chunk_cache = {}

        for i in range(num_merges):
            counts = Counter()
            for chunk_ids, freq in chunks.items():
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    counts[pair] += freq
            if not counts:
                break

            top_pair = counts.most_common(1)[0][0]
            new_id = 256 + i
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            new_chunks = {}
            for chunk_ids, freq in chunks.items():
                merged = tuple(_merge(list(chunk_ids), top_pair, new_id))
                new_chunks[merged] = new_chunks.get(merged, 0) + freq
            chunks = new_chunks

            if verbose and (i + 1) % 100 == 0:
                merged_bytes = self.vocab[new_id]
                try:
                    preview = merged_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    preview = repr(merged_bytes)
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {new_id} ({preview!r})")

    def _encode_chunk(self, chunk):
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached
        chunk_ids = list(chunk.encode('utf-8'))
        while len(chunk_ids) >= 2:
            pairs = set(zip(chunk_ids, chunk_ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            chunk_ids = _merge(chunk_ids, pair, self.merges[pair])
        result = tuple(chunk_ids)
        self._chunk_cache[chunk] = result
        return result

    def encode_ordinary(self, text):
        """Encode ignoring special tokens."""
        out = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text):
        if not self.special_tokens:
            return self.encode_ordinary(text)
        specials_pattern = '(' + '|'.join(re.escape(s) for s in self.special_tokens) + ')'
        out = []
        for piece in re.split(specials_pattern, text):
            if piece in self.special_tokens:
                out.append(self.special_tokens[piece])
            elif piece:
                out.extend(self.encode_ordinary(piece))
        return out

    def decode(self, ids):
        inv_specials = {v: k for k, v in self.special_tokens.items()}
        parts = []
        byte_buf = bytearray()
        for i in ids:
            if i in inv_specials:
                if byte_buf:
                    parts.append(bytes(byte_buf).decode('utf-8', errors='replace'))
                    byte_buf.clear()
                parts.append(inv_specials[i])
            else:
                byte_buf.extend(self.vocab[i])
        if byte_buf:
            parts.append(bytes(byte_buf).decode('utf-8', errors='replace'))
        return ''.join(parts)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'version': 'bpe-v1',
            'pattern': self.pattern,
            'special_tokens': self.special_tokens,
            'merges': [[a, b] for (a, b) in sorted(self.merges, key=self.merges.get)],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        assert data.get('version') == 'bpe-v1', f"unexpected file version: {data.get('version')!r}"
        self.pattern = data['pattern']
        self._compiled = re.compile(self.pattern)
        self.special_tokens = dict(data.get('special_tokens') or {})
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._chunk_cache = {}
        for i, (a, b) in enumerate(data['merges']):
            new_id = 256 + i
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
