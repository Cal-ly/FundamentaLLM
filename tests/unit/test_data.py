"""Tests for datasets and dataloaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from fundamentallm.data.dataset import LanguageModelDataset
from fundamentallm.data.loaders import create_dataloaders


def test_dataset_length():
    tokens = torch.arange(12)
    dataset = LanguageModelDataset(tokens, sequence_length=4, stride=2)
    expected = max(0, (len(tokens) - 4 - 1) // 2 + 1)
    assert len(dataset) == expected


def test_getitem_shape():
    tokens = torch.arange(10)
    dataset = LanguageModelDataset(tokens, sequence_length=4, stride=1)
    sample = dataset[0]
    assert sample[0].shape[0] == 4
    assert sample[1].shape[0] == 4


def test_getitem_values():
    tokens = torch.tensor([1, 2, 3, 4, 5])
    dataset = LanguageModelDataset(tokens, sequence_length=3, stride=1)
    inputs, targets = dataset[0]
    assert inputs.tolist() == [1, 2, 3]
    assert targets.tolist() == [2, 3, 4]


def test_stride_behavior():
    tokens = torch.arange(10)
    ds_overlap = LanguageModelDataset(tokens, sequence_length=3, stride=1)
    ds_nonoverlap = LanguageModelDataset(tokens, sequence_length=3, stride=3)
    assert len(ds_overlap) > len(ds_nonoverlap)


def test_edge_cases():
    tokens = torch.arange(3)
    dataset = LanguageModelDataset(tokens, sequence_length=5)
    assert len(dataset) == 0


def test_create_dataloaders(sample_text, sample_tokenizer, train_config):
    sample_tokenizer.train([sample_text])
    train_loader, val_loader = create_dataloaders(sample_text, sample_tokenizer, train_config)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    for batch in train_loader:
        inputs, targets = batch
        assert inputs.shape[1] == train_config.sequence_length
        assert targets.shape[1] == train_config.sequence_length
        break


def test_train_val_split(sample_text, sample_tokenizer, train_config):
    sample_tokenizer.train([sample_text])
    train_loader, val_loader = create_dataloaders(sample_text, sample_tokenizer, train_config)

    tokens = torch.tensor(sample_tokenizer.encode(sample_text))
    split_idx = int(len(tokens) * train_config.train_split)
    split_idx = max(split_idx, 1)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_dataset: LanguageModelDataset = train_loader.dataset  # type: ignore[attr-defined]
    val_dataset: LanguageModelDataset = val_loader.dataset  # type: ignore[attr-defined]
    assert train_dataset.token_ids.tolist() == train_tokens.tolist()
    assert val_dataset.token_ids.tolist() == val_tokens.tolist()


def test_dataloader_iteration(sample_text, sample_tokenizer, train_config):
    sample_tokenizer.train([sample_text])
    train_loader, val_loader = create_dataloaders(sample_text, sample_tokenizer, train_config)

    train_batches = sum(1 for _ in train_loader)
    val_batches = sum(1 for _ in val_loader)
    assert train_batches >= 0
    assert val_batches >= 0
