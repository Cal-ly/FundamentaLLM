"""Unit tests for feed-forward network component."""

import pytest
import torch

from fundamentallm.models.components.feedforward import FeedForwardNetwork


class TestFeedForwardNetwork:
    """Tests for FeedForwardNetwork."""

    @pytest.fixture
    def ffn(self):
        """Create FeedForwardNetwork instance."""
        return FeedForwardNetwork(d_model=512, dropout=0.1)

    def test_output_shape(self, ffn):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 32, 512)
        output = ffn(x)
        assert output.shape == x.shape

    def test_learnable_parameters(self, ffn):
        """Test that FFN has learnable parameters."""
        params = list(ffn.parameters())
        # linear1 (weight + bias) + linear2 (weight + bias) = 4 params
        assert len(params) == 4

        for param in params:
            assert param.requires_grad

    def test_parameter_dimensions(self, ffn):
        """Test that parameter dimensions are correct."""
        d_model = 512
        d_ff = 4 * d_model

        # linear1: d_model -> d_ff
        assert ffn.linear1.weight.shape == (d_ff, d_model)
        assert ffn.linear1.bias.shape == (d_ff,)

        # linear2: d_ff -> d_model
        assert ffn.linear2.weight.shape == (d_model, d_ff)
        assert ffn.linear2.bias.shape == (d_model,)

    def test_gradient_flow(self, ffn):
        """Test that gradients flow through FFN."""
        x = torch.randn(2, 32, 512, requires_grad=True)
        output = ffn(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert ffn.linear1.weight.grad is not None
        assert ffn.linear2.weight.grad is not None

    def test_different_batch_sizes(self, ffn):
        """Test that FFN works with different batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 32, 512)
            output = ffn(x)
            assert output.shape == (batch_size, 32, 512)

    def test_different_sequence_lengths(self, ffn):
        """Test that FFN works with different sequence lengths."""
        for seq_len in [1, 8, 16, 32, 64, 128]:
            x = torch.randn(2, seq_len, 512)
            output = ffn(x)
            assert output.shape == (2, seq_len, 512)

    def test_different_model_dimensions(self):
        """Test FFN with different model dimensions."""
        for d_model in [64, 128, 256, 512, 1024]:
            ffn = FeedForwardNetwork(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            output = ffn(x)
            assert output.shape == x.shape

    def test_custom_d_ff(self):
        """Test FFN with custom hidden dimension."""
        ffn = FeedForwardNetwork(d_model=512, d_ff=2048)
        x = torch.randn(2, 32, 512)
        output = ffn(x)

        assert output.shape == x.shape
        assert ffn.linear1.weight.shape == (2048, 512)
        assert ffn.linear2.weight.shape == (512, 2048)

    def test_default_d_ff_is_4x_d_model(self):
        """Test that default d_ff is 4 * d_model."""
        d_model = 256
        ffn = FeedForwardNetwork(d_model=d_model)
        assert ffn.d_ff == 4 * d_model

    def test_relu_activation(self):
        """Test FFN with ReLU activation."""
        ffn = FeedForwardNetwork(d_model=256, activation="relu")
        x = torch.randn(2, 16, 256)
        output = ffn(x)

        assert output.shape == x.shape
        assert isinstance(ffn.activation, torch.nn.ReLU)

    def test_gelu_activation(self):
        """Test FFN with GELU activation (default)."""
        ffn = FeedForwardNetwork(d_model=256, activation="gelu")
        x = torch.randn(2, 16, 256)
        output = ffn(x)

        assert output.shape == x.shape
        assert isinstance(ffn.activation, torch.nn.GELU)

    def test_invalid_activation_error(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            FeedForwardNetwork(d_model=256, activation="sigmoid")

    def test_activation_case_insensitive(self):
        """Test that activation names are case-insensitive."""
        ffn1 = FeedForwardNetwork(d_model=256, activation="GELU")
        ffn2 = FeedForwardNetwork(d_model=256, activation="Gelu")
        ffn3 = FeedForwardNetwork(d_model=256, activation="gelu")

        assert isinstance(ffn1.activation, torch.nn.GELU)
        assert isinstance(ffn2.activation, torch.nn.GELU)
        assert isinstance(ffn3.activation, torch.nn.GELU)

    def test_output_bounded_reasonably(self, ffn):
        """Test that output values are in reasonable range."""
        x = torch.randn(2, 32, 512)
        output = ffn(x)

        # Check not NaN or Inf
        assert torch.isfinite(output).all()

        # Check reasonable scale (std not too large/small)
        std = output.std().item()
        assert 0.1 < std < 10.0

    def test_dropout_effect(self):
        """Test that dropout affects training but not eval."""
        ffn = FeedForwardNetwork(d_model=256, dropout=0.5)
        x = torch.randn(2, 16, 256)

        # Training mode: outputs vary due to dropout
        ffn.train()
        output1 = ffn(x)
        output2 = ffn(x)

        # Due to high dropout (0.5), outputs should differ
        assert not torch.allclose(output1, output2, atol=0.01)

        # Eval mode: outputs identical
        ffn.eval()
        output3 = ffn(x)
        output4 = ffn(x)

        assert torch.allclose(output3, output4)

    def test_eval_mode_reproducible(self, ffn):
        """Test that eval mode is reproducible."""
        ffn.eval()
        x = torch.randn(2, 32, 512)

        with torch.no_grad():
            output1 = ffn(x)
            output2 = ffn(x)

        assert torch.allclose(output1, output2)


class TestFeedForwardComparison:
    """Compare different FFN configurations."""

    def test_gelu_vs_relu(self):
        """Test that GELU and ReLU produce different outputs."""
        x = torch.randn(1, 10, 256)

        ffn_gelu = FeedForwardNetwork(d_model=256, activation="gelu")
        ffn_relu = FeedForwardNetwork(d_model=256, activation="relu")

        # Copy weights to make them identical except activation
        with torch.no_grad():
            ffn_relu.linear1.weight.copy_(ffn_gelu.linear1.weight)
            ffn_relu.linear1.bias.copy_(ffn_gelu.linear1.bias)
            ffn_relu.linear2.weight.copy_(ffn_gelu.linear2.weight)
            ffn_relu.linear2.bias.copy_(ffn_gelu.linear2.bias)

        ffn_gelu.eval()
        ffn_relu.eval()

        output_gelu = ffn_gelu(x)
        output_relu = ffn_relu(x)

        # Outputs should differ due to different activations
        assert not torch.allclose(output_gelu, output_relu, atol=0.01)

    def test_different_d_ff_values(self):
        """Test FFN with different hidden dimensions."""
        x = torch.randn(2, 16, 512)

        ffn_2x = FeedForwardNetwork(d_model=512, d_ff=1024)
        ffn_4x = FeedForwardNetwork(d_model=512, d_ff=2048)
        ffn_8x = FeedForwardNetwork(d_model=512, d_ff=4096)

        output_2x = ffn_2x(x)
        output_4x = ffn_4x(x)
        output_8x = ffn_8x(x)

        # All should have same output shape
        assert output_2x.shape == output_4x.shape == output_8x.shape == x.shape


class TestFFNInContext:
    """Test FFN in context of typical transformer usage."""

    def test_ffn_composition_with_norm(self):
        """Test that FFN can be composed with normalization."""
        from fundamentallm.models.components.normalization import RMSNorm

        norm = RMSNorm(d_model=256)
        ffn = FeedForwardNetwork(d_model=256)

        x = torch.randn(2, 16, 256)

        # Typical use: norm(x) -> ffn(x)
        normed = norm(x)
        output = ffn(normed)

        assert output.shape == x.shape

    def test_ffn_with_residual_connection(self):
        """Test FFN with residual connection (typical transformer pattern)."""
        ffn = FeedForwardNetwork(d_model=256)
        x = torch.randn(2, 16, 256)

        # Residual: x + ffn(x)
        output = x + ffn(x)

        assert output.shape == x.shape

        # Gradient should flow through both paths
        loss = output.mean()
        loss.backward()
        # If we got here without error, gradients flowed correctly
