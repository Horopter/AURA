"""
Comprehensive unit tests for training/feature_models module.
Tests feature-based models with dummy feature arrays.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.feature_models import (
    FeatureMLP,
    FeatureCNN1D,
    FeatureTransformer,
    FeatureLSTM,
    FeatureResNet,
    create_feature_model,
)


class TestFeatureMLP:
    """Tests for FeatureMLP class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return torch.randn(10, 50)  # (batch, features)
    
    def test_initialization(self):
        """Test FeatureMLP initialization."""
        model = FeatureMLP(input_dim=50, hidden_dims=[256, 128, 64])
        assert isinstance(model, nn.Module)
        assert model.input_dim == 50
    
    def test_forward_pass(self, dummy_features):
        """Test FeatureMLP forward pass."""
        model = FeatureMLP(input_dim=50, hidden_dims=[32, 16], num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)  # (batch, num_classes)
    
    def test_forward_pass_binary(self, dummy_features):
        """Test FeatureMLP with binary classification."""
        model = FeatureMLP(input_dim=50, num_classes=1)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 1)


class TestFeatureCNN1D:
    """Tests for FeatureCNN1D class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return torch.randn(10, 50)  # (batch, features)
    
    def test_initialization(self):
        """Test FeatureCNN1D initialization."""
        model = FeatureCNN1D(input_dim=50, num_filters=[64, 128])
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, dummy_features):
        """Test FeatureCNN1D forward pass."""
        model = FeatureCNN1D(input_dim=50, num_filters=[32, 64], num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)


class TestFeatureTransformer:
    """Tests for FeatureTransformer class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return torch.randn(10, 50)
    
    def test_initialization(self):
        """Test FeatureTransformer initialization."""
        model = FeatureTransformer(input_dim=50, d_model=128)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, dummy_features):
        """Test FeatureTransformer forward pass."""
        model = FeatureTransformer(input_dim=50, d_model=64, num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)


class TestFeatureLSTM:
    """Tests for FeatureLSTM class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return torch.randn(10, 50)
    
    def test_initialization(self):
        """Test FeatureLSTM initialization."""
        model = FeatureLSTM(input_dim=50, hidden_dim=64)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, dummy_features):
        """Test FeatureLSTM forward pass."""
        model = FeatureLSTM(input_dim=50, hidden_dim=32, num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)


class TestFeatureResNet:
    """Tests for FeatureResNet class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return torch.randn(10, 50)
    
    def test_initialization(self):
        """Test FeatureResNet initialization."""
        model = FeatureResNet(input_dim=50, num_blocks=2)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, dummy_features):
        """Test FeatureResNet forward pass."""
        model = FeatureResNet(input_dim=50, num_blocks=1, num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_features)
        
        assert output.shape == (10, 2)


class TestCreateFeatureModel:
    """Tests for create_feature_model function."""
    
    def test_create_feature_model_mlp(self):
        """Test create_feature_model for MLP."""
        model = create_feature_model("mlp", input_dim=50)
        assert isinstance(model, FeatureMLP)
    
    def test_create_feature_model_cnn(self):
        """Test create_feature_model for CNN."""
        model = create_feature_model("cnn", input_dim=50)
        assert isinstance(model, FeatureCNN1D)
    
    def test_create_feature_model_invalid(self):
        """Test create_feature_model with invalid type."""
        with pytest.raises(ValueError):
            create_feature_model("invalid", input_dim=50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

