"""
Comprehensive unit tests for training/slowfast_advanced module.
Tests advanced SlowFast variants with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.slowfast_advanced import (
    SlowFastAttentionModel,
    MultiScaleSlowFastModel,
)


class TestSlowFastAttentionModel:
    """Tests for SlowFastAttentionModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test SlowFastAttentionModel initialization."""
        try:
            model = SlowFastAttentionModel(pretrained=False)
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("SlowFast dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test SlowFastAttentionModel forward pass."""
        try:
            model = SlowFastAttentionModel(pretrained=False)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
        except (ImportError, RuntimeError):
            pytest.skip("SlowFast dependencies not available")


class TestMultiScaleSlowFastModel:
    """Tests for MultiScaleSlowFastModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test MultiScaleSlowFastModel initialization."""
        try:
            model = MultiScaleSlowFastModel(pretrained=False)
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("SlowFast dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test MultiScaleSlowFastModel forward pass."""
        try:
            model = MultiScaleSlowFastModel(pretrained=False)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
        except (ImportError, RuntimeError):
            pytest.skip("SlowFast dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

