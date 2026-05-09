# file to test my tokenizer on some sample corpus and text

from minigpt.tokenizer.create_tokenizer import Tokenizer

def make_tokenizer(corpus=None):
    corpus = corpus or [
        "Hello World!",
        "The quick brown fox.",
        "Langauge models are powerful",
        "a.b x2 42 café"
    ] * 20

    tok = Tokenizer(lowercase=False)
    tok.train_bpe(corpus, num_merges=50)
    tok.build_vocab(corpus=corpus)
    return tok

def test_roundtrip_basic_text():
    tok = make_tokenizer()
    text = "Hello world!"
    assert tok.decode(tok.encode(text)) == text

def test_roundtrip_punctuation():
    tok = make_tokenizer(["a.b hello! wow?"] * 50)
    text = "a.b hello! wow?"
    assert tok.decode(tok.encode(text)) == text

def test_digits_do_not_merge_into_words():
    tok = make_tokenizer(["x2 42 7b"] * 50)
    tokens = [tok.id_to_token(i) for i in tok.encode("x2 42 7b")]
    assert "x2</w>" not in tokens
    assert "42</w>" not in tokens

def test_bos_eos():
    tok = make_tokenizer()
    ids = tok.encode("Hello", add_bos=True, add_eos=True)
    assert tok.id_to_token(ids[0]) == "<BOS>"
    assert tok.id_to_token(ids[-1]) == "<EOS>"

def test_batch_padding():
    tok = make_tokenizer()
    batch = tok.batch_encode(["hi", "hello"], max_len=10, pad=True)
    assert len(batch) == 2
    assert all(len(row) == 10 for row in batch)

def test_save_load_roundtrip(tmp_path):
    tok = make_tokenizer()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    loaded = Tokenizer()
    loaded.load(str(path))
    text = "Hello world!"
    assert loaded.decode(loaded.encode(text)) == text





