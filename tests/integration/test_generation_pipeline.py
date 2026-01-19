"""Integration test for text generation pipeline."""

from __future__ import annotations

from fundamentallm.config import TransformerConfig
from fundamentallm.data.tokenizers.character import CharacterTokenizer
from fundamentallm.generation.generator import TextGenerator
from fundamentallm.generation.sampling import TopKSampler
from fundamentallm.models.transformer import Transformer


def test_generation_pipeline_end_to_end(sample_text):
    tokenizer = CharacterTokenizer()
    tokenizer.train([sample_text])

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=1,
        sequence_length=32,
    )
    model = Transformer(config)

    generator = TextGenerator(model, tokenizer, device="cpu")
    output = generator.generate("hello", max_tokens=5, temperature=1.2, top_k=5)

    assert output.startswith("hello")
    assert len(output) >= len("hello")

    outputs = generator.batch_generate(["hi", "world"], max_tokens=3, sampler=TopKSampler(k=5))
    assert len(outputs) == 2
    assert all(isinstance(text, str) and text for text in outputs)
