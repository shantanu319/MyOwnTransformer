from tokenizer import BPETokenizer


CORPUS = (
    "the quick brown fox jumps over the lazy dog " * 20
    + "hello world hello there hello friends " * 20
)


def test_base_vocab_is_all_bytes():
    t = BPETokenizer()
    assert len(t.vocab) == 256
    for i in range(256):
        assert t.vocab[i] == bytes([i])


def test_train_produces_exact_vocab_size():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=300)
    assert len(t.vocab) == 300
    assert len(t.merges) == 44  # 300 - 256


def test_train_merges_shorten_token_sequence():
    """Trained tokenizer should produce fewer tokens than raw bytes."""
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    sample = "the quick brown fox"
    assert len(t.encode(sample)) < len(sample.encode('utf-8'))


def test_ascii_roundtrip():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    text = "the quick brown fox jumps"
    assert t.decode(t.encode(text)) == text


def test_unicode_roundtrip():
    t = BPETokenizer()
    t.train("café résumé naïve " * 20, vocab_size=320)
    for text in ["café", "résumé", "naïve", "café résumé"]:
        assert t.decode(t.encode(text)) == text


def test_unseen_text_roundtrip():
    """Encode/decode should round-trip on text not seen during training."""
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    unseen = "Zebras don't eat pizza, 42 times!"
    assert t.decode(t.encode(unseen)) == unseen


def test_special_tokens_tokenized_as_single_id():
    t = BPETokenizer(special_tokens={'<|endoftext|>': 500})
    t.train(CORPUS, vocab_size=400)
    ids = t.encode("hello <|endoftext|> world")
    assert 500 in ids
    assert t.decode(ids) == "hello <|endoftext|> world"


def test_train_equivalent_to_naive():
    """Dedup-based train must produce the exact same merges as the
    straightforward per-chunk algorithm — same merges, same ids, same order."""
    from collections import Counter
    from tokenizer import _get_pair_counts, _merge, GPT2_SPLIT_PATTERN
    import re

    def naive_train(text, vocab_size):
        compiled = re.compile(GPT2_SPLIT_PATTERN)
        ids = [list(c.encode('utf-8')) for c in compiled.findall(text)]
        merges = {}
        for i in range(vocab_size - 256):
            counts = Counter()
            for chunk in ids:
                _get_pair_counts(chunk, counts)
            if not counts:
                break
            top = counts.most_common(1)[0][0]
            new_id = 256 + i
            merges[top] = new_id
            ids = [_merge(c, top, new_id) for c in ids]
        return merges

    for corpus, vocab in [
        (CORPUS, 400),
        ("café résumé naïve " * 20, 320),
        ("a" * 100 + " b" * 100 + " ab " * 50, 280),
    ]:
        t = BPETokenizer()
        t.train(corpus, vocab_size=vocab)
        assert t.merges == naive_train(corpus, vocab), \
            f"dedup train diverged from naive for vocab={vocab}"


def test_save_load_roundtrip(tmp_path):
    t = BPETokenizer(special_tokens={'<|endoftext|>': 500})
    t.train(CORPUS, vocab_size=350)

    path = tmp_path / "bpe.json"
    t.save(str(path))

    t2 = BPETokenizer()
    t2.load(str(path))

    text = "the quick brown fox <|endoftext|>"
    assert t2.encode(text) == t.encode(text)
    assert t2.decode(t.encode(text)) == text
    assert t2.merges == t.merges
    assert t2.special_tokens == t.special_tokens
