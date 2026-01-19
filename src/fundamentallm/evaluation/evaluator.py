"""Model evaluation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from fundamentallm.config import TransformerConfig
from fundamentallm.data.tokenizers import CharacterTokenizer
from fundamentallm.data.tokenizers.base import BaseTokenizer
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


class ModelEvaluator:
    """Evaluate model performance on a dataset."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: BaseTokenizer,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        *,
        return_predictions: bool = False,
    ) -> Dict[str, float | torch.Tensor]:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        predictions_list: list[torch.Tensor] = [] if return_predictions else []
        targets_list: list[torch.Tensor] = [] if return_predictions else []

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            mask = (targets != -100) & (targets != -1)
            tokens = int(mask.sum().item())
            if tokens == 0:
                tokens = targets.numel()

            total_loss += float(loss.item())
            total_tokens += tokens

            preds = torch.argmax(logits, dim=-1)
            total_correct += int((preds[mask] == targets[mask]).sum().item()) if mask.any() else 0

            if return_predictions:
                predictions_list.append(preds.cpu())
                targets_list.append(targets.cpu())

        denom = max(total_tokens, 1)
        avg_loss = total_loss / denom
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        accuracy = total_correct / denom

        results: Dict[str, float | torch.Tensor] = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
        }

        if return_predictions:
            results["predictions"] = torch.cat(predictions_list, dim=0) if predictions_list else torch.empty(0)
            results["targets"] = torch.cat(targets_list, dim=0) if targets_list else torch.empty(0)

        return results

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        config: Optional[TransformerConfig] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        device: str = "cpu",
    ) -> "ModelEvaluator":
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=device)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise RuntimeError(f"Invalid checkpoint format at {path}")

        model_config = config or _load_config_from_artifacts(path, checkpoint)
        model = Transformer(model_config)
        model.load_state_dict(checkpoint["model_state"])

        tokenizer_instance = tokenizer or _load_tokenizer_from_artifacts(path)
        return cls(model, tokenizer_instance, device=device)
