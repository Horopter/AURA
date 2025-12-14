"""
Comprehensive unit tests for training/_feature_cnn module.
Tests FeatureCNN1D with dummy feature arrays.
"""
import pytest
import torch
import torch.nn as nn
from lib.training._feature_cnn import FeatureCNN1D


class TestFeatureCNN1D:
    """Tests for FeatureCNN1D class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return torch.randn(10, 50)  # (batch, features)
    
    def test_initialization(self):
        """Test FeatureCNN1D initialization."""
        model = FeatureCNN1D(input_dim=50, conv_channels=[64, 128])
        assert isinstance(model, nn.Module)
        assert model.input_dim == 50
    
    def test_forward_pass(self, dummy_features):
        """Test FeatureCNN1D forward pass."""
        model = FeatureCNN1D(input_dim=50, conv_channels=[32, 64], num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)
    
    def test_forward_pass_different_pool_types(self, dummy_features):
        """Test FeatureCNN1D with different pooling types."""
        for pool_type in ["max", "avg", "adaptive"]:
            model = FeatureCNN1D(input_dim=50, pool_type=pool_type, num_classes=2)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_features)
            
            assert output.shape == (10, 2)
    
    def test_initialization_kernel_size_mismatch(self):
        """Test FeatureCNN1D raises error on kernel size mismatch."""
        with pytest.raises(ValueError, match="kernel_sizes must match"):
            FeatureCNN1D(
                input_dim=50,
                conv_channels=[32, 64],
                kernel_sizes=[3]  # Mismatch: should have 2 values
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

