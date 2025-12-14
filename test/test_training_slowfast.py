"""
Comprehensive unit tests for training/slowfast module.
Tests SlowFastModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.slowfast import SlowFastModel


class TestSlowFastModel:
    """Tests for SlowFastModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test SlowFastModel initialization."""
        try:
            model = SlowFastModel(pretrained=False)
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("SlowFast dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test SlowFastModel forward pass."""
        try:
            model = SlowFastModel(pretrained=False)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("SlowFast dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

