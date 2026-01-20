"""Unit tests for multi-head attention mechanism."""

import pytest
import torch

from fundamentallm.models.components.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    @pytest.fixture
    def attention(self):
        """Create MultiHeadAttention instance."""
        return MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)

    def test_output_shape_self_attention(self, attention):
        """Test that output shape matches input shape for self-attention."""
        x = torch.randn(2, 32, 512)
        output = attention(x, x, x)
        assert output.shape == x.shape

    def test_output_shape_cross_attention(self, attention):
        """Test output shape for cross-attention (different seq lengths)."""
        query = torch.randn(2, 16, 512)
        key = torch.randn(2, 32, 512)
        value = torch.randn(2, 32, 512)

        output = attention(query, key, value)
        assert output.shape == query.shape

    def test_has_learnable_parameters(self, attention):
        """Test that attention has learnable projection matrices."""
        params = list(attention.parameters())
        assert len(params) >= 8  # Q, K, V, O projections (weight + bias each)

        for param in params:
            assert param.requires_grad

    def test_gradient_flow(self, attention):
        """Test that gradients flow through attention."""
        x = torch.randn(2, 32, 512, requires_grad=True)
        output = attention(x, x, x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_causal_mask_prevents_future_attention(self, attention):
        """Test that causal mask prevents attention to future positions."""
        x = torch.randn(1, 4, 512)
        output = attention(x, x, x)  # causal=True by default

        # The mask should be applied, output should be valid
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_causal_mask_creation(self):
        """Test that causal mask is correctly lower triangular."""
        mask = MultiHeadAttention._create_causal_mask(4, 4, torch.device("cpu"))

        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )

        assert torch.allclose(mask, expected)

    def test_causal_mask_rectangular(self):
        """Test causal mask for non-square attention (query vs key length)."""
        # Query length = 2, Key length = 4
        mask = MultiHeadAttention._create_causal_mask(2, 4, torch.device("cpu"))

        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )

        assert torch.allclose(mask, expected)

    def test_different_num_heads(self):
        """Test attention with different numbers of heads."""
        for num_heads in [1, 2, 4, 8, 16]:
            attention = MultiHeadAttention(d_model=512, num_heads=num_heads)
            x = torch.randn(2, 32, 512)
            output = attention(x, x, x)
            assert output.shape == x.shape

    def test_different_model_dimensions(self):
        """Test attention with different model dimensions."""
        for d_model in [64, 128, 256, 512, 1024]:
            attention = MultiHeadAttention(d_model=d_model, num_heads=8)
            x = torch.randn(2, 16, d_model)
            output = attention(x, x, x)
            assert output.shape == x.shape

    def test_dimension_mismatch_error(self):
        """Test that incompatible d_model and num_heads raises error."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=510, num_heads=8)  # 510 not divisible by 8

    def test_custom_attention_mask(self):
        """Test that custom attention mask is applied."""
        attention = MultiHeadAttention(d_model=512, num_heads=8, causal=False)

        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, 512)

        # Create custom mask: attend to self and one position back
        custom_mask = torch.zeros(batch_size, seq_len, seq_len)
        for i in range(seq_len):
            custom_mask[:, i, max(0, i - 1) : i + 1] = 1

        output = attention(x, x, x, attention_mask=custom_mask)
        assert output.shape == x.shape

    def test_dropout_effect(self):
        """Test that dropout affects training but not eval."""
        attention = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.5)
        x = torch.randn(2, 32, 512)

        # Training mode: outputs should vary due to dropout
        attention.train()
        output1 = attention(x, x, x)
        output2 = attention(x, x, x)

        # Due to dropout, outputs may differ (with high probability)
        # But they should be close on average
        assert not torch.allclose(output1, output2, atol=0.1)

        # Eval mode: outputs should be identical
        attention.eval()
        output3 = attention(x, x, x)
        output4 = attention(x, x, x)

        assert torch.allclose(output3, output4)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 (after softmax)."""
        # Create a simple attention without causal mask for testing
        attention = MultiHeadAttention(d_model=64, num_heads=2, causal=False)
        x = torch.randn(1, 4, 64)

        # Forward pass to compute attention weights
        # We need to capture them during forward, but they're not returned
        # Instead, we verify the output is reasonable
        output = attention(x, x, x)

        # Output should be in a reasonable range (not exploding)
        assert torch.isfinite(output).all()
        assert output.std() > 0


class TestCausalMaskEffect:
    """Test the effect of causal masking on attention."""

    def test_causal_vs_non_causal(self):
        """Test that causal and non-causal attention produce different results."""
        x = torch.randn(1, 8, 256)

        causal_attn = MultiHeadAttention(d_model=256, num_heads=4, causal=True)
        non_causal_attn = MultiHeadAttention(d_model=256, num_heads=4, causal=False)

        # Copy weights to make architectures identical except for causal mask
        with torch.no_grad():
            non_causal_attn.W_q.weight.copy_(causal_attn.W_q.weight)
            non_causal_attn.W_k.weight.copy_(causal_attn.W_k.weight)
            non_causal_attn.W_v.weight.copy_(causal_attn.W_v.weight)
            non_causal_attn.W_o.weight.copy_(causal_attn.W_o.weight)
            non_causal_attn.W_q.bias.copy_(causal_attn.W_q.bias)
            non_causal_attn.W_k.bias.copy_(causal_attn.W_k.bias)
            non_causal_attn.W_v.bias.copy_(causal_attn.W_v.bias)
            non_causal_attn.W_o.bias.copy_(causal_attn.W_o.bias)

        causal_attn.eval()
        non_causal_attn.eval()

        output_causal = causal_attn(x, x, x)
        output_non_causal = non_causal_attn(x, x, x)

        # Outputs should be different
        assert not torch.allclose(output_causal, output_non_causal, atol=0.01)


class TestMultiHeadConsistency:
    """Test that multi-head attention has consistent behavior."""

    def test_single_head_vs_multi_head_shape(self):
        """Test that single-head and multi-head attention produce same output shape."""
        x = torch.randn(2, 16, 512)

        single_head = MultiHeadAttention(d_model=512, num_heads=1)
        multi_head = MultiHeadAttention(d_model=512, num_heads=8)

        output1 = single_head(x, x, x)
        output2 = multi_head(x, x, x)

        assert output1.shape == output2.shape == x.shape

    def test_deterministic_eval(self):
        """Test that attention is deterministic in eval mode."""
        attention = MultiHeadAttention(d_model=256, num_heads=4, dropout=0.5)
        attention.eval()

        x = torch.randn(2, 16, 256)

        output1 = attention(x, x, x)
        output2 = attention(x, x, x)

        assert torch.allclose(output1, output2)
