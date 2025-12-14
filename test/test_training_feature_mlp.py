"""
Comprehensive unit tests for training/_feature_mlp module.
Tests FeatureMLP with dummy feature arrays.
"""
import pytest
import torch
import torch.nn as nn
from lib.training._feature_mlp import FeatureMLP


class TestFeatureMLP:
    """Tests for FeatureMLP class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return torch.randn(10, 50)  # (batch, features)
    
    def test_initialization(self):
        """Test FeatureMLP initialization."""
        model = FeatureMLP(input_dim=50, hidden_dims=[256, 128])
        assert isinstance(model, nn.Module)
        assert model.input_dim == 50
    
    def test_forward_pass(self, dummy_features):
        """Test FeatureMLP forward pass."""
        model = FeatureMLP(input_dim=50, hidden_dims=[32, 16], num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)
    
    def test_forward_pass_different_activations(self, dummy_features):
        """Test FeatureMLP with different activation functions."""
        for activation in ["relu", "gelu", "tanh"]:
            model = FeatureMLP(input_dim=50, activation=activation, num_classes=2)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_features)
            
            assert output.shape == (10, 2)
    
    def test_forward_pass_without_batch_norm(self, dummy_features):
        """Test FeatureMLP without batch normalization."""
        model = FeatureMLP(input_dim=50, batch_norm=False, num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

