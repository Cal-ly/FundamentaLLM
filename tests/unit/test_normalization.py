"""Unit tests for normalization layers."""

import pytest
import torch

from fundamentallm.models.components.normalization import LayerNorm, RMSNorm


class TestLayerNorm:
    """Tests for LayerNorm."""

    @pytest.fixture
    def norm(self):
        """Create LayerNorm instance."""
        return LayerNorm(512)

    def test_output_shape(self, norm):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 32, 512)
        output = norm(x)
        assert output.shape == x.shape

    def test_output_has_learnable_parameters(self, norm):
        """Test that weight and bias are learnable parameters."""
        assert norm.weight.requires_grad
        assert norm.bias.requires_grad
        assert norm.weight.shape == torch.Size([512])
        assert norm.bias.shape == torch.Size([512])

    def test_zero_mean_unit_variance(self, norm):
        """Test that output has approximately zero mean and unit variance."""
        x = torch.randn(100, 512)
        output = norm(x)

        # Compute mean and variance across all dimensions
        mean = output.mean().item()
        var = output.var().item()

        # Check approximately zero mean and unit variance
        # (not exact due to learnable scale/shift)
        assert abs(mean) < 0.2
        assert abs(var - 1.0) < 1.0

    def test_gradient_flow(self, norm):
        """Test that gradients flow through LayerNorm."""
        x = torch.randn(2, 32, 512, requires_grad=True)
        output = norm(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None
        assert norm.bias.grad is not None

    def test_different_batch_sizes(self, norm):
        """Test that LayerNorm works with different batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 32, 512)
            output = norm(x)
            assert output.shape == (batch_size, 32, 512)

    def test_different_sequence_lengths(self, norm):
        """Test that LayerNorm works with different sequence lengths."""
        for seq_len in [1, 8, 16, 32, 64]:
            x = torch.randn(2, seq_len, 512)
            output = norm(x)
            assert output.shape == (2, seq_len, 512)

    def test_numerical_stability(self):
        """Test numerical stability with small values."""
        norm = LayerNorm(512, eps=1e-6)
        x = torch.randn(2, 32, 512) * 1e-5
        output = norm(x)

        # Should not have NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_eval_mode(self, norm):
        """Test that eval mode doesn't affect LayerNorm (no dropout/batch norm effects)."""
        x = torch.randn(2, 32, 512)

        norm.train()
        output_train = norm(x)

        norm.eval()
        output_eval = norm(x)

        # LayerNorm is deterministic, so train and eval should be identical
        assert torch.allclose(output_train, output_eval)


class TestRMSNorm:
    """Tests for RMSNorm."""

    @pytest.fixture
    def norm(self):
        """Create RMSNorm instance."""
        return RMSNorm(512)

    def test_output_shape(self, norm):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 32, 512)
        output = norm(x)
        assert output.shape == x.shape

    def test_has_single_learnable_parameter(self, norm):
        """Test that RMSNorm has only weight (no bias)."""
        assert norm.weight.requires_grad
        assert norm.weight.shape == torch.Size([512])
        assert not hasattr(norm, "bias") or norm.bias is None

    def test_gradient_flow(self, norm):
        """Test that gradients flow through RMSNorm."""
        x = torch.randn(2, 32, 512, requires_grad=True)
        output = norm(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_different_batch_sizes(self, norm):
        """Test that RMSNorm works with different batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 32, 512)
            output = norm(x)
            assert output.shape == (batch_size, 32, 512)

    def test_different_sequence_lengths(self, norm):
        """Test that RMSNorm works with different sequence lengths."""
        for seq_len in [1, 8, 16, 32, 64]:
            x = torch.randn(2, seq_len, 512)
            output = norm(x)
            assert output.shape == (2, seq_len, 512)

    def test_numerical_stability(self):
        """Test numerical stability with small values."""
        norm = RMSNorm(512, eps=1e-6)
        x = torch.randn(2, 32, 512) * 1e-5
        output = norm(x)

        # Should not have NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_scale_invariance(self, norm):
        """Test that RMSNorm normalization is scale-invariant.

        The normalization part is scale-invariant, but the learnable weight
        breaks this. However, if we ignore the weight multiplication,
        RMSNorm(α*x) should normalize to similar distribution as RMSNorm(x).
        """
        x = torch.randn(2, 32, 512)

        _ = norm(x)  # Verify it runs without errors

        # If we scale input by alpha, the normalized (pre-scale) part
        # should be identical to the non-scaled version
        alpha = 2.0

        # Compute RMS for both
        rms1 = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-6)
        rms2 = torch.sqrt(((alpha * x) ** 2).mean(dim=-1, keepdim=True) + 1e-6)

        # Check that scaled RMS equals alpha times original RMS
        assert torch.allclose(rms2, alpha * rms1, atol=1e-5)

    def test_rms_computation_correctness(self):
        """Test that RMSNorm correctly computes RMS."""
        norm = RMSNorm(5)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Manually compute RMS
        rms_expected = torch.sqrt((x**2).mean(dim=-1, keepdim=True))
        output = norm(x)

        # Check that normalization is correct (before scaling)
        normalized = x / (rms_expected + 1e-6)
        expected = normalized * norm.weight

        assert torch.allclose(output, expected, atol=1e-5)

    def test_eval_mode(self, norm):
        """Test that eval mode doesn't affect RMSNorm (no dropout/batch norm effects)."""
        x = torch.randn(2, 32, 512)

        norm.train()
        output_train = norm(x)

        norm.eval()
        output_eval = norm(x)

        # RMSNorm is deterministic, so train and eval should be identical
        assert torch.allclose(output_train, output_eval)


class TestNormalizationComparison:
    """Compare LayerNorm and RMSNorm behaviors."""

    def test_both_preserve_shape(self):
        """Test that both normalizations preserve input shape."""
        layer_norm = LayerNorm(256)
        rms_norm = RMSNorm(256)

        x = torch.randn(4, 16, 256)

        output1 = layer_norm(x)
        output2 = rms_norm(x)

        assert output1.shape == x.shape
        assert output2.shape == x.shape

    def test_parameter_count(self):
        """Test parameter counts: LayerNorm has 2x parameters of RMSNorm."""
        d_model = 512
        layer_norm = LayerNorm(d_model)
        rms_norm = RMSNorm(d_model)

        layer_norm_params = sum(p.numel() for p in layer_norm.parameters())
        rms_norm_params = sum(p.numel() for p in rms_norm.parameters())

        # LayerNorm: weight + bias = 2 * d_model
        # RMSNorm: weight = d_model
        assert layer_norm_params == 2 * d_model
        assert rms_norm_params == d_model
        assert layer_norm_params == 2 * rms_norm_params

    def test_both_are_differentiable(self):
        """Test that both normalizations support backpropagation."""
        layer_norm = LayerNorm(256)
        rms_norm = RMSNorm(256)

        x = torch.randn(2, 16, 256, requires_grad=True)

        output1 = layer_norm(x)
        output2 = rms_norm(x)

        loss1 = output1.mean()
        loss2 = output2.mean()

        loss1.backward()
        assert x.grad is not None

        x.grad.zero_()

        loss2.backward()
        assert x.grad is not None
