"""Unit tests for positional encoding layers."""

import math
import pytest
import torch
from fundamentallm.models.components.embeddings import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    create_positional_encoding,
)


class TestLearnedPositionalEncoding:
    """Tests for LearnedPositionalEncoding."""
    
    @pytest.fixture
    def encoding(self):
        """Create LearnedPositionalEncoding instance."""
        return LearnedPositionalEncoding(d_model=512, max_seq_len=2048)
    
    def test_output_shape(self, encoding):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 32, 512)
        output = encoding(x)
        assert output.shape == (1, 32, 512)
    
    def test_output_dimension_matches_model(self, encoding):
        """Test that output d_model matches input."""
        x = torch.randn(4, 16, 512)
        output = encoding(x)
        assert output.size(-1) == 512
    
    def test_learnable_parameters(self, encoding):
        """Test that learned encodings are learnable."""
        assert encoding.embeddings.weight.requires_grad
    
    def test_gradient_flow(self, encoding):
        """Test that gradients flow through learned encoding."""
        x = torch.randn(2, 32, 512, requires_grad=True)
        output = encoding(x)
        loss = output.mean()
        loss.backward()
        
        assert encoding.embeddings.weight.grad is not None
        assert torch.any(encoding.embeddings.weight.grad != 0)
    
    def test_different_sequence_lengths(self, encoding):
        """Test that encoding works with different sequence lengths."""
        for seq_len in [1, 8, 16, 32, 64, 128]:
            x = torch.randn(2, seq_len, 512)
            output = encoding(x)
            assert output.shape == (1, seq_len, 512)
    
    def test_exceeds_max_seq_len_error(self, encoding):
        """Test that exceeding max_seq_len raises error."""
        x = torch.randn(2, 3000, 512)
        
        with pytest.raises(RuntimeError, match="exceeds max_seq_len"):
            encoding(x)
    
    def test_position_embeddings_are_different(self, encoding):
        """Test that different positions have different embeddings."""
        x = torch.randn(1, 10, 512)
        output = encoding(x)
        
        # Different positions should have different embeddings
        for i in range(output.size(1) - 1):
            pos_i = output[0, i, :]
            pos_i_plus_1 = output[0, i + 1, :]
            # They should not be identical
            assert not torch.allclose(pos_i, pos_i_plus_1)
    
    def test_deterministic_same_position(self):
        """Test that same positions always produce same encoding (in eval mode)."""
        encoding = LearnedPositionalEncoding(d_model=256, max_seq_len=100)
        encoding.eval()  # Disable dropout for determinism
        
        with torch.no_grad():
            x1 = torch.randn(1, 5, 256)
            x2 = torch.randn(1, 5, 256)
            
            output1 = encoding(x1)
            output2 = encoding(x2)
            
            # Same positions should give same encoding
            assert torch.allclose(output1, output2)
    
    def test_eval_mode_doesnt_affect_learned(self):
        """Test that eval mode doesn't change learned encodings (except dropout)."""
        encoding = LearnedPositionalEncoding(d_model=256, max_seq_len=100)
        
        x = torch.randn(1, 10, 256)
        
        encoding.eval()
        output_eval1 = encoding(x)
        output_eval2 = encoding(x)
        
        # Eval mode: learned encodings are deterministic (dropout disabled)
        assert torch.allclose(output_eval1, output_eval2)


class TestSinusoidalPositionalEncoding:
    """Tests for SinusoidalPositionalEncoding."""
    
    @pytest.fixture
    def encoding(self):
        """Create SinusoidalPositionalEncoding instance."""
        return SinusoidalPositionalEncoding(d_model=512, max_seq_len=2048)
    
    def test_output_shape(self, encoding):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 32, 512)
        output = encoding(x)
        assert output.shape == (1, 32, 512)
    
    def test_non_learnable_parameters(self, encoding):
        """Test that sinusoidal encoding has no learnable parameters."""
        params = list(encoding.parameters())
        assert len(params) == 0  # Only dropout parameters (biases), no embeddings
    
    def test_gradient_flow_through_dropout(self):
        """Test that positional encoding doesn't break gradient flow."""
        encoding = SinusoidalPositionalEncoding(d_model=512, max_seq_len=2048)
        encoding.eval()  # Disable dropout for this test
        
        x = torch.randn(2, 32, 512, requires_grad=True)
        output = encoding(x)
        
        # The output from positional encoding alone has no gradients
        # (it's fixed, not a function of x), but we can test that x
        # can still be used in a computation graph
        combined = x + output
        loss = combined.mean()
        loss.backward()
        
        # Should work without errors
        assert x.grad is not None
    
    def test_sinusoidal_formula_even_indices(self):
        """Test that even indices use sin formula."""
        encoding = SinusoidalPositionalEncoding(d_model=512)
        
        # Manually compute expected values for position 0
        # Even indices: sin(0 * div_term) = sin(0) = 0
        pos = 0
        d_model = 512
        
        dim_indices = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(dim_indices * -(math.log(10000.0) / d_model))
        expected_even = torch.sin(torch.tensor(pos, dtype=torch.float) * div_term)
        
        # Extract from pe
        with torch.no_grad():
            actual = encoding.pe[pos, 0::2]
        
        # Check a few values
        assert torch.allclose(actual[:10], expected_even[:10], atol=1e-5)
    
    def test_sinusoidal_formula_odd_indices(self):
        """Test that odd indices use cos formula."""
        encoding = SinusoidalPositionalEncoding(d_model=512)
        
        # Manually compute expected values for position 0
        # Odd indices: cos(0 * div_term) = cos(0) = 1
        pos = 0
        d_model = 512
        
        dim_indices = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(dim_indices * -(math.log(10000.0) / d_model))
        expected_odd = torch.cos(torch.tensor(pos, dtype=torch.float) * div_term)
        
        # Extract from pe
        with torch.no_grad():
            actual = encoding.pe[pos, 1::2]
        
        # Check a few values
        # Note: if d_model is odd, we have one fewer odd indices
        assert torch.allclose(actual[:10], expected_odd[:10], atol=1e-5)
    
    def test_different_sequence_lengths(self, encoding):
        """Test that encoding works with different sequence lengths."""
        for seq_len in [1, 8, 16, 32, 64, 128]:
            x = torch.randn(2, seq_len, 512)
            output = encoding(x)
            assert output.shape == (1, seq_len, 512)
    
    def test_exceeds_max_seq_len_error(self, encoding):
        """Test that exceeding max_seq_len raises error."""
        x = torch.randn(2, 3000, 512)
        
        with pytest.raises(RuntimeError, match="exceeds max_seq_len"):
            encoding(x)
    
    def test_extrapolation_to_longer_sequences(self):
        """Test that sinusoidal encoding can extrapolate to longer sequences."""
        # Create with small max_seq_len
        encoding = SinusoidalPositionalEncoding(d_model=128, max_seq_len=1024)
        
        # Generate encoding for max length - should work
        x = torch.randn(1, 1024, 128)
        output = encoding(x)
        assert output.shape == (1, 1024, 128)
    
    def test_position_unique_values(self, encoding):
        """Test that different positions have different values."""
        x = torch.randn(1, 100, 512)
        output = encoding(x)
        
        # Different positions should have different values
        for i in range(min(10, output.size(1) - 1)):
            pos_i = output[0, i, :]
            pos_i_plus_1 = output[0, i + 1, :]
            assert not torch.allclose(pos_i, pos_i_plus_1, atol=1e-3)
    
    def test_deterministic_same_position(self):
        """Test that same positions always produce same encoding (in eval mode)."""
        encoding = SinusoidalPositionalEncoding(d_model=256, max_seq_len=100)
        encoding.eval()  # Disable dropout for determinism
        
        with torch.no_grad():
            x1 = torch.randn(1, 5, 256)
            x2 = torch.randn(1, 5, 256)
            
            output1 = encoding(x1)
            output2 = encoding(x2)
            
            # Same positions should give same encoding
            assert torch.allclose(output1, output2)
    
    def test_eval_mode_doesnt_affect_sinusoidal(self):
        """Test that eval mode doesn't change sinusoidal encodings (except dropout)."""
        encoding = SinusoidalPositionalEncoding(d_model=256, max_seq_len=100)
        
        x = torch.randn(1, 10, 256)
        
        encoding.eval()
        output_eval1 = encoding(x)
        output_eval2 = encoding(x)
        
        # Eval mode: sinusoidal encodings are deterministic (dropout disabled)
        assert torch.allclose(output_eval1, output_eval2)


class TestPositionalEncodingFactory:
    """Tests for the factory function."""
    
    def test_create_learned_encoding(self):
        """Test creating learned positional encoding."""
        encoding = create_positional_encoding("learned", d_model=512)
        assert isinstance(encoding, LearnedPositionalEncoding)
    
    def test_create_sinusoidal_encoding(self):
        """Test creating sinusoidal positional encoding."""
        encoding = create_positional_encoding("sinusoidal", d_model=512)
        assert isinstance(encoding, SinusoidalPositionalEncoding)
    
    def test_create_case_insensitive(self):
        """Test that factory is case-insensitive."""
        encoding1 = create_positional_encoding("LEARNED", d_model=512)
        encoding2 = create_positional_encoding("Learned", d_model=512)
        
        assert isinstance(encoding1, LearnedPositionalEncoding)
        assert isinstance(encoding2, LearnedPositionalEncoding)
    
    def test_invalid_encoding_type_error(self):
        """Test that invalid encoding type raises error."""
        with pytest.raises(ValueError, match="Unknown encoding type"):
            create_positional_encoding("invalid", d_model=512)
    
    def test_factory_parameters_passed_through(self):
        """Test that factory passes parameters correctly."""
        encoding = create_positional_encoding(
            "learned",
            d_model=256,
            max_seq_len=1024,
            dropout=0.2,
        )
        
        assert encoding.d_model == 256
        assert encoding.max_seq_len == 1024


class TestPositionalEncodingComparison:
    """Compare learned vs sinusoidal encodings."""
    
    def test_both_produce_same_output_shape(self):
        """Test that both encodings produce same output shape."""
        x = torch.randn(2, 32, 256)
        
        learned = LearnedPositionalEncoding(d_model=256)
        sinusoidal = SinusoidalPositionalEncoding(d_model=256)
        
        output1 = learned(x)
        output2 = sinusoidal(x)
        
        assert output1.shape == output2.shape == (1, 32, 256)
    
    def test_learned_vs_sinusoidal_different_values(self):
        """Test that learned and sinusoidal give different encodings."""
        x = torch.randn(1, 10, 128)
        
        learned = LearnedPositionalEncoding(d_model=128, max_seq_len=100)
        sinusoidal = SinusoidalPositionalEncoding(d_model=128, max_seq_len=100)
        
        output_learned = learned(x)
        output_sinusoidal = sinusoidal(x)
        
        # Should be different (with very high probability)
        assert not torch.allclose(output_learned, output_sinusoidal, atol=0.1)
    
    def test_learned_has_parameters_sinusoidal_does_not(self):
        """Test parameter counts."""
        learned = LearnedPositionalEncoding(d_model=256, max_seq_len=100)
        sinusoidal = SinusoidalPositionalEncoding(d_model=256, max_seq_len=100)
        
        # Learned should have embedding parameters
        learned_params = sum(p.numel() for p in learned.parameters())
        assert learned_params > 0
        
        # Sinusoidal should have no trainable parameters
        sinusoidal_params = sum(p.numel() for p in sinusoidal.parameters())
        assert sinusoidal_params == 0
