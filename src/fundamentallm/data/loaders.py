"""DataLoader builders for language modeling."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader

from fundamentallm.config.training import TrainingConfig
from fundamentallm.data.dataset import LanguageModelDataset
from fundamentallm.data.tokenizers.base import BaseTokenizer


def create_dataloaders(
    text: str,
    tokenizer: BaseTokenizer,
    config: TrainingConfig,
    return_tokenizer: bool = False,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, BaseTokenizer]]:
    """Tokenize text and produce train/validation dataloaders.

    Splits at the token level to avoid leakage.
    """

    tokens = tokenizer.encode(text)
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    train_size = int(len(token_tensor) * config.train_split)
    train_size = max(train_size, 1)
    train_tokens = token_tensor[:train_size]
    val_tokens = token_tensor[train_size:]

    train_dataset = LanguageModelDataset(train_tokens, config.sequence_length)
    val_dataset = LanguageModelDataset(val_tokens, config.sequence_length)

    pin_memory = config.device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    if return_tokenizer:
        return train_loader, val_loader, tokenizer
    return train_loader, val_loader
