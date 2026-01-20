"""Text generation utilities."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def _load_config_from_artifacts(
    checkpoint_path: Path, checkpoint_payload: dict
) -> TransformerConfig:
    """Load transformer config from checkpoint or nearby files.

    Attempts to load config in order:
    1. From checkpoint payload ("config" or "model_config" keys)
    2. From {checkpoint_path}.yaml or .yml
    3. From config.yaml or model.yaml in same directory

    Args:
        checkpoint_path: Path to checkpoint file.
        checkpoint_payload: Loaded checkpoint dictionary.

    Returns:
        TransformerConfig instance.

    Raises:
        ValueError: If config cannot be found in any location.
    """
    # Try loading from checkpoint payload
    if "config" in checkpoint_payload:
        logger.debug("Loaded config from checkpoint['config']")
        return TransformerConfig.model_validate(checkpoint_payload["config"])
    if "model_config" in checkpoint_payload:
        logger.debug("Loaded config from checkpoint['model_config']")
        return TransformerConfig.model_validate(checkpoint_payload["model_config"])

    # Try loading from nearby files
    candidates = [
        checkpoint_path.with_suffix(".yaml"),
        checkpoint_path.with_suffix(".yml"),
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent / "model.yaml",
    ]

    logger.debug(f"Searching for config in: {candidates}")
    for candidate in candidates:
        if candidate.exists():
            logger.debug(f"Loaded config from {candidate}")
            return TransformerConfig.from_yaml(candidate)

    # Detailed error message with search paths
    search_paths = "\n".join(f"  - {c}" for c in candidates)
    raise ValueError(
        f"TransformerConfig not found. Searched in checkpoint payload and:\n{search_paths}\n"
        "Solutions:\n"
        "  1. Include config in checkpoint payload under 'config' key\n"
        "  2. Place config.yaml next to checkpoint file\n"
        "  3. Pass config explicitly to from_checkpoint(config=...)"
    )


def _load_tokenizer_from_artifacts(checkpoint_path: Path) -> BaseTokenizer:
    """Load tokenizer from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        CharacterTokenizer instance.

    Raises:
        FileNotFoundError: If tokenizer.json not found next to checkpoint.
    """
    candidate = checkpoint_path.parent / "tokenizer.json"
    logger.debug(f"Looking for tokenizer at: {candidate}")

    if candidate.exists():
        logger.debug(f"Loaded tokenizer from {candidate}")
        return CharacterTokenizer.load(candidate)

    raise FileNotFoundError(
        f"Tokenizer artifact not found at {candidate}\n"
        "Solutions:\n"
        f"  1. Save tokenizer to {candidate}\n"
        "  2. Pass tokenizer explicitly to from_checkpoint(tokenizer=...)"
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
        """Load generator from checkpoint with config and tokenizer.

        Args:
            checkpoint_path: Path to model checkpoint.
            config: Optional TransformerConfig. If not provided, will search for it.
            tokenizer: Optional tokenizer. If not provided, will search for it.
            device: Device to load model to ("cpu", "cuda", etc.).

        Returns:
            TextGenerator instance ready for generation.

        Raises:
            FileNotFoundError: If checkpoint not found.
            RuntimeError: If checkpoint format is invalid.
            ValueError: If config/tokenizer not found and not provided.
        """
        path = Path(checkpoint_path)

        # Load and validate checkpoint file
        try:
            logger.debug(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=device)
        except FileNotFoundError as exc:
            logger.error(f"Checkpoint file not found: {path}")
            raise FileNotFoundError(f"Checkpoint not found at {path}") from exc
        except Exception as exc:
            logger.error(f"Failed to load checkpoint: {exc}")
            raise RuntimeError(f"Checkpoint appears corrupted or invalid: {exc}") from exc

        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            logger.error(f"Invalid checkpoint format at {path}")
            raise RuntimeError(
                f"Invalid checkpoint format. Expected dict with 'model_state' key, "
                f"got {type(checkpoint)}"
            )

        # Load config
        try:
            model_config = config or _load_config_from_artifacts(path, checkpoint)
        except ValueError as exc:
            logger.error(f"Could not load config: {exc}")
            raise

        # Create and load model
        try:
            model = Transformer(model_config)
            model.load_state_dict(checkpoint["model_state"])
            logger.debug(f"Model loaded with {model.count_parameters():,} parameters")
        except Exception as exc:
            logger.error(f"Failed to load model state: {exc}")
            raise RuntimeError(f"Failed to load model from checkpoint: {exc}") from exc

        # Load tokenizer
        try:
            tokenizer_instance = tokenizer or _load_tokenizer_from_artifacts(path)
        except FileNotFoundError as exc:
            logger.error(f"Could not load tokenizer: {exc}")
            raise

        logger.info(f"Successfully loaded generator from {path}")
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
