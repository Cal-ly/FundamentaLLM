"""Text generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch

from fundamentallm.config import TransformerConfig
from fundamentallm.data.tokenizers import CharacterTokenizer
from fundamentallm.data.tokenizers.base import BaseTokenizer
from fundamentallm.generation.constraints import StopSequenceConstraint
from fundamentallm.generation.sampling import (
    GreedySampler,
    Sampler,
    TemperatureSampler,
    TopKSampler,
    TopPSampler,
)
from fundamentallm.models.transformer import Transformer


def _load_config_from_artifacts(checkpoint_path: Path, checkpoint_payload: dict) -> TransformerConfig:
    if "config" in checkpoint_payload:
        return TransformerConfig.model_validate(checkpoint_payload["config"])
    if "model_config" in checkpoint_payload:
        return TransformerConfig.model_validate(checkpoint_payload["model_config"])

    candidates = [
        checkpoint_path.with_suffix(".yaml"),
        checkpoint_path.with_suffix(".yml"),
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent / "model.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return TransformerConfig.from_yaml(candidate)

    raise ValueError(
        "TransformerConfig not found in checkpoint; provide config or include it in the checkpoint payload"
    )


def _load_tokenizer_from_artifacts(checkpoint_path: Path) -> BaseTokenizer:
    candidate = checkpoint_path.parent / "tokenizer.json"
    if candidate.exists():
        return CharacterTokenizer.load(candidate)
    raise FileNotFoundError(
        f"Tokenizer artifact not found next to checkpoint at {candidate}" \
        "; pass a tokenizer instance explicitly"
    )


def _select_sampler(
    default_sampler: Sampler,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    override: Optional[Sampler],
) -> Sampler:
    if override is not None:
        return override
    if top_k is not None:
        return TopKSampler(k=top_k, temperature=temperature)
    if top_p is not None:
        return TopPSampler(p=top_p, temperature=temperature)
    if isinstance(default_sampler, TemperatureSampler):
        return TemperatureSampler(temperature)
    if isinstance(default_sampler, GreedySampler) and temperature != 1.0:
        return TemperatureSampler(temperature)
    return default_sampler


class TextGenerator:
    """Generate text from a trained language model."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: BaseTokenizer,
        device: str | torch.device = "cpu",
        sampler: Optional[Sampler] = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.sampler = sampler or GreedySampler()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        config: Optional[TransformerConfig] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        device: str = "cpu",
    ) -> "TextGenerator":
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=device)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise RuntimeError(f"Invalid checkpoint format at {path}")

        model_config = config or _load_config_from_artifacts(path, checkpoint)
        model = Transformer(model_config)
        model.load_state_dict(checkpoint["model_state"])

        tokenizer_instance = tokenizer or _load_tokenizer_from_artifacts(path)
        return cls(model, tokenizer_instance, device=device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        sampler: Optional[Sampler] = None,
    ) -> str:
        prompt_tokens: List[int] = self.tokenizer.encode(prompt)
        if not prompt_tokens:
            raise ValueError("Prompt must yield at least one token")

        input_ids = torch.tensor([prompt_tokens], device=self.device, dtype=torch.long)
        generated: List[int] = []

        stop_constraint = (
            StopSequenceConstraint(stop_sequences, self.tokenizer) if stop_sequences else None
        )
        sampler_impl = _select_sampler(self.sampler, temperature, top_k, top_p, sampler)

        max_seq_len = getattr(getattr(self.model, "config", None), "sequence_length", None)

        for _ in range(max_tokens):
            if max_seq_len is not None and input_ids.size(1) >= max_seq_len:
                break

            logits = self.model(input_ids)
            next_logits = logits[:, -1, :]
            next_token = sampler_impl.sample(next_logits)
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)

            token_id = int(next_token[0].item())
            generated.append(token_id)

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

            if stop_constraint is not None:
                combined = prompt_tokens + generated
                satisfied, _ = stop_constraint.is_met(combined)
                if satisfied:
                    break

        full_tokens = prompt_tokens + generated
        return self.tokenizer.decode(full_tokens)

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 200,
        **kwargs,
    ) -> List[str]:
        return [self.generate(prompt, max_tokens=max_tokens, **kwargs) for prompt in prompts]
