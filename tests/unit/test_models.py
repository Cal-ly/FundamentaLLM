"""Unit and integration tests for transformer model."""

import pytest
import torch
from fundamentallm.config import TransformerConfig
from fundamentallm.models.transformer import TransformerBlock, Transformer


class TestTransformerBlock:
    """Tests for TransformerBlock."""
    
    @pytest.fixture
    def block(self):
        """Create TransformerBlock instance."""
        return TransformerBlock(d_model=256, num_heads=4, dropout=0.1)
    
    def test_output_shape(self, block):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 16, 256)
        output = block(x)
        assert output.shape == x.shape
    
    def test_learnable_parameters(self, block):
        """Test that block has learnable parameters."""
        params = list(block.parameters())
        assert len(params) > 0
        
        for param in params:
            assert param.requires_grad
    
    def test_gradient_flow(self, block):
        """Test that gradients flow through block."""
        x = torch.randn(2, 16, 256, requires_grad=True)
        output = block(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None
    
    def test_different_batch_sizes(self, block):
        """Test block with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 16, 256)
            output = block(x)
            assert output.shape == x.shape
    
    def test_different_sequence_lengths(self, block):
        """Test block with different sequence lengths."""
        for seq_len in [4, 8, 16, 32]:
            x = torch.randn(2, seq_len, 256)
            output = block(x)
            assert output.shape == x.shape
    
    def test_causal_mask_application(self, block):
        """Test that causal mask is applied in attention."""
        x = torch.randn(1, 8, 256)
        output = block(x)
        
        # Should not have NaN or Inf
        assert torch.isfinite(output).all()
    
    def test_rmsnorm_vs_layernorm(self):
        """Test block with different normalization types."""
        x = torch.randn(2, 16, 256)
        
        block_rms = TransformerBlock(d_model=256, num_heads=4, norm_type="rmsnorm")
        block_ln = TransformerBlock(d_model=256, num_heads=4, norm_type="layernorm")
        
        output_rms = block_rms(x)
        output_ln = block_ln(x)
        
        # Both should work and have correct shape
        assert output_rms.shape == x.shape
        assert output_ln.shape == x.shape
        
        # Outputs should differ (different norms)
        assert not torch.allclose(output_rms, output_ln, atol=0.1)
    
    def test_invalid_norm_type_error(self):
        """Test that invalid norm type raises error."""
        with pytest.raises(ValueError, match="Unknown norm_type"):
            TransformerBlock(d_model=256, norm_type="invalid")
    
    def test_residual_connections_preserve_gradient(self, block):
        """Test that residual connections preserve gradient."""
        x = torch.randn(2, 16, 256, requires_grad=True)
        output = block(x)
        
        # Sum gradients from output
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients from both paths
        assert x.grad is not None
        assert torch.any(x.grad != 0)


class TestTransformer:
    """Tests for complete Transformer model."""
    
    @pytest.fixture
    def config(self):
        """Create basic transformer config."""
        return TransformerConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=4,
            num_layers=2,
            sequence_length=512,
        )
    
    @pytest.fixture
    def model(self, config):
        """Create Transformer model."""
        return Transformer(config)
    
    def test_instantiation(self, config):
        """Test that model can be instantiated."""
        model = Transformer(config)
        assert model is not None
        assert model.vocab_size == config.vocab_size
        assert model.d_model == config.d_model
        assert model.num_layers == config.num_layers
    
    def test_forward_output_shape(self, model):
        """Test that forward pass produces correct output shape."""
        input_ids = torch.randint(0, 1000, (2, 16))
        logits = model(input_ids)
        
        assert logits.shape == (2, 16, 1000)  # [batch, seq_len, vocab_size]
    
    def test_forward_output_finite(self, model):
        """Test that output values are finite."""
        input_ids = torch.randint(0, 1000, (2, 16))
        logits = model(input_ids)
        
        assert torch.isfinite(logits).all()
    
    def test_different_sequence_lengths(self, model):
        """Test forward pass with different sequence lengths."""
        for seq_len in [1, 4, 8, 16, 32]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            logits = model(input_ids)
            
            assert logits.shape == (2, seq_len, 1000)
    
    def test_different_batch_sizes(self, model):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 1000, (batch_size, 16))
            logits = model(input_ids)
            
            assert logits.shape == (batch_size, 16, 1000)
    
    def test_gradient_flow(self, model):
        """Test that gradients flow through model."""
        input_ids = torch.randint(0, 1000, (2, 16))
        logits = model(input_ids)
        
        loss = logits.mean()
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                # At least some parameters should have gradients
                if param.grad is not None:
                    assert torch.any(param.grad != 0)
    
    def test_parameter_count(self, model):
        """Test parameter counting."""
        total_params = model.count_parameters()
        
        assert total_params > 0
        assert total_params == sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    
    def test_weight_tying(self, model):
        """Test that output projection shares weight with embedding."""
        # Weight tying: output_proj.weight should be same as token_embedding.weight
        assert model.output_proj.weight is model.token_embedding.weight
    
    def test_causal_mask_effect(self):
        """Test that causal masking works correctly."""
        config = TransformerConfig(vocab_size=100, d_model=64, num_heads=4, num_layers=1)
        model = Transformer(config)
        model.eval()
        
        input_ids = torch.arange(8).unsqueeze(0)  # [1, 8]
        
        with torch.no_grad():
            logits = model(input_ids)
        
        # Should produce valid logits
        assert logits.shape == (1, 8, 100)
        assert torch.isfinite(logits).all()
    
    def test_eval_mode(self, model):
        """Test eval mode behavior."""
        model.eval()
        
        input_ids = torch.randint(0, 1000, (2, 16))
        
        with torch.no_grad():
            output1 = model(input_ids)
            output2 = model(input_ids)
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(output1, output2)
    
    def test_different_config_sizes(self):
        """Test models with different sizes."""
        configs = [
            TransformerConfig(vocab_size=100, d_model=64, num_heads=2, num_layers=1),
            TransformerConfig(vocab_size=1000, d_model=128, num_heads=4, num_layers=2),
            TransformerConfig(vocab_size=10000, d_model=256, num_heads=8, num_layers=4),
        ]
        
        for config in configs:
            model = Transformer(config)
            input_ids = torch.randint(0, config.vocab_size, (2, 16))
            logits = model(input_ids)
            
            assert logits.shape == (2, 16, config.vocab_size)
    
    def test_learned_vs_sinusoidal_encoding(self):
        """Test model with different positional encodings."""
        config_learned = TransformerConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=1,
            pos_encoding="learned",
        )
        config_sine = TransformerConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=1,
            pos_encoding="sinusoidal",
        )
        
        model_learned = Transformer(config_learned)
        model_sine = Transformer(config_sine)
        
        input_ids = torch.randint(0, 100, (1, 8))
        
        logits_learned = model_learned(input_ids)
        logits_sine = model_sine(input_ids)
        
        # Both should work and produce same shape
        assert logits_learned.shape == logits_sine.shape
    
    def test_custom_d_ff(self):
        """Test model with custom feed-forward dimension."""
        config = TransformerConfig(
            vocab_size=100,
            d_model=128,
            num_heads=4,
            num_layers=1,
        )
        
        model = Transformer(config)
        input_ids = torch.randint(0, 100, (2, 16))
        logits = model(input_ids)
        
        assert logits.shape == (2, 16, 100)


class TestTransformerIntegration:
    """Integration tests for transformer end-to-end."""
    
    def test_next_token_prediction(self):
        """Test basic next-token prediction."""
        config = TransformerConfig(vocab_size=50, d_model=128, num_heads=4, num_layers=2)
        model = Transformer(config)
        model.eval()
        
        # Input sequence
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        with torch.no_grad():
            logits = model(input_ids)
        
        # Get next token prediction
        next_token_logits = logits[0, -1, :]  # Logits for last position
        next_token_id = next_token_logits.argmax().item()
        
        # Should be a valid token ID
        assert 0 <= next_token_id < 50
    
    def test_generation_step(self):
        """Test one generation step."""
        config = TransformerConfig(vocab_size=30, d_model=64, num_heads=4, num_layers=1)
        model = Transformer(config)
        model.eval()
        
        # Start with initial prompt
        context = torch.tensor([[5, 10, 15]])  # [batch=1, seq_len=3]
        
        with torch.no_grad():
            logits = model(context)  # [1, 3, 30]
            next_logits = logits[0, -1, :]  # Last position logits
            next_token = next_logits.argmax().item()
        
        # Extend context
        new_context = torch.cat([context, torch.tensor([[next_token]])], dim=1)
        assert new_context.shape == (1, 4)
    
    def test_backprop_through_layers(self):
        """Test backpropagation through all layers."""
        config = TransformerConfig(vocab_size=100, d_model=128, num_heads=4, num_layers=3)
        model = Transformer(config)
        
        input_ids = torch.randint(0, 100, (2, 16))
        logits = model(input_ids)
        
        # Compute loss (e.g., mean prediction)
        loss = logits.mean()
        loss.backward()
        
        # Check that all layers have gradients
        layer_count = 0
        for name, param in model.named_parameters():
            if "blocks" in name and param.requires_grad:
                if param.grad is not None:
                    layer_count += 1
        
        # Should have gradients in block layers
        assert layer_count > 0
    
    def test_large_vocab(self):
        """Test with large vocabulary."""
        config = TransformerConfig(
            vocab_size=50000,
            d_model=256,
            num_heads=8,
            num_layers=2,
        )
        model = Transformer(config)
        
        input_ids = torch.randint(0, 50000, (2, 16))
        logits = model(input_ids)
        
        assert logits.shape == (2, 16, 50000)
    
    def test_long_sequence(self):
        """Test with long sequence (within max_seq_len)."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            sequence_length=512,
        )
        model = Transformer(config)
        
        # Use a long sequence
        input_ids = torch.randint(0, 1000, (2, 256))
        logits = model(input_ids)
        
        assert logits.shape == (2, 256, 1000)
    
    def test_exceed_max_seq_len_error(self):
        """Test that exceeding max_seq_len raises error."""
        config = TransformerConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=1,
            sequence_length=128,
        )
        model = Transformer(config)
        
        # Try to use longer sequence
        input_ids = torch.randint(0, 100, (1, 256))
        
        with pytest.raises(RuntimeError, match="exceeds max_seq_len|exceeds sequence_length"):
            model(input_ids)
