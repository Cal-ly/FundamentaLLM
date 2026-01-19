"""Model configuration definitions."""

from __future__ import annotations

from pydantic import Field, field_validator

from .base import BaseConfig


class TransformerConfig(BaseConfig):
    """Transformer architecture configuration."""

    vocab_size: int = Field(..., gt=0)
    d_model: int = Field(512, gt=0)
    num_heads: int = Field(8, gt=0)
    num_layers: int = Field(6, gt=0, le=48)
    sequence_length: int = Field(256, gt=0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    ffn_expansion: int = Field(4, gt=0)
    pos_encoding: str = Field("learned", pattern="^(learned|sinusoidal|rope)$")

    @field_validator("d_model")
    @classmethod
    def validate_d_model(cls, value: int, info) -> int:
        num_heads = info.data.get("num_heads", cls.model_fields["num_heads"].default)
        if value % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        return value

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads
