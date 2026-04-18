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
