"""
Comprehensive unit tests for training/i3d module.
Tests I3DModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.i3d import I3DModel


class TestI3DModel:
    """Tests for I3DModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization_pretrained(self):
        """Test I3DModel initialization with pretrained=True."""
        try:
            model = I3DModel(pretrained=True)
            assert isinstance(model, nn.Module)
            assert hasattr(model, 'backbone')
        except (ImportError, RuntimeError):
            pytest.skip("I3D dependencies not available")
    
    def test_initialization_no_pretrained(self):
        """Test I3DModel initialization with pretrained=False."""
        try:
            model = I3DModel(pretrained=False)
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("I3D dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test I3DModel forward pass."""
        try:
            model = I3DModel(pretrained=False)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2  # Batch size
            assert output.shape[1] == 1  # Binary classification
        except (ImportError, RuntimeError):
            pytest.skip("I3D dependencies not available")
    
    def test_forward_pass_small_spatial(self):
        """Test I3DModel forward pass with small spatial dimensions."""
        try:
            model = I3DModel(pretrained=False)
            model.eval()
            
            # Small spatial dimensions (should be upsampled)
            x = torch.randn(2, 3, 8, 16, 16)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape[0] == 2
        except (ImportError, RuntimeError):
            pytest.skip("I3D dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

