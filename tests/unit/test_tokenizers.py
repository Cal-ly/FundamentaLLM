"""Tests for tokenizers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fundamentallm.data.tokenizers.character import CharacterTokenizer


def test_vocab_size(sample_tokenizer, sample_text):
    assert sample_tokenizer.vocab_size >= 4
    training_corpus = [sample_text, "hello world", "test data", "fundamentallm"]
    unique_chars = set("".join(training_corpus))
    assert sample_tokenizer.vocab_size == len(unique_chars) + 4


def test_encode_decode_roundtrip(sample_tokenizer, sample_text):
    tokens = sample_tokenizer.encode(sample_text)
    decoded = sample_tokenizer.decode(tokens)
    assert decoded == sample_text


def test_unknown_character_handling(sample_tokenizer):
    tokens = sample_tokenizer.encode("hello !")
    assert sample_tokenizer.unk_token_id in tokens
    decoded = sample_tokenizer.decode(tokens, skip_special_tokens=False)
    assert "<UNK>" in decoded


def test_special_tokens(sample_tokenizer):
    tokens = sample_tokenizer.encode("hello", add_special_tokens=True)
    assert tokens[0] == sample_tokenizer.bos_token_id
    assert tokens[-1] == sample_tokenizer.eos_token_id


def test_batch_encode_decode(sample_tokenizer):
    texts = ["hello", "world"]
    batch_tokens = sample_tokenizer.batch_encode(texts, add_special_tokens=True)
    assert len(batch_tokens) == 2
    roundtrip = sample_tokenizer.batch_decode(batch_tokens)
    assert roundtrip == texts


def test_save_load(tmp_path, sample_tokenizer):
    path = tmp_path / "tokenizer.json"
    sample_tokenizer.save(path)
    assert path.exists()

    loaded = CharacterTokenizer.load(path)
    assert loaded.vocab_size == sample_tokenizer.vocab_size
    assert loaded.encode("hello") == sample_tokenizer.encode("hello")

    raw = json.loads(path.read_text())
    assert "char_to_id" in raw


def test_padding_token(sample_tokenizer):
    assert sample_tokenizer.pad_token_id in sample_tokenizer.char_to_id.values()


def test_edge_cases():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["a"])
    assert tokenizer.encode("") == []
    long_text = "a" * 1000
    tokens = tokenizer.encode(long_text)
    assert len(tokens) == 1000
    decoded = tokenizer.decode(tokens)
    assert decoded == long_text


def test_untrained_guard():
    tokenizer = CharacterTokenizer()
    with pytest.raises(RuntimeError):
        tokenizer.encode("hi")
