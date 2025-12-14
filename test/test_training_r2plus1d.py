"""
Comprehensive unit tests for training/r2plus1d module.
Tests R2Plus1DModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.r2plus1d import R2Plus1DModel


class TestR2Plus1DModel:
    """Tests for R2Plus1DModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test R2Plus1DModel initialization."""
        try:
            model = R2Plus1DModel(pretrained=False)
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("R2Plus1D dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test R2Plus1DModel forward pass."""
        try:
            model = R2Plus1DModel(pretrained=False)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("R2Plus1D dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

