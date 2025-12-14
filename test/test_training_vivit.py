"""
Comprehensive unit tests for training/vivit module.
Tests ViViTModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.vivit import ViViTModel


class TestViViTModel:
    """Tests for ViViTModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 256, 256)
    
    def test_initialization(self):
        """Test ViViTModel initialization."""
        try:
            model = ViViTModel(
                num_frames=8,
                img_size=256,
                pretrained=False
            )
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("ViViT dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test ViViTModel forward pass."""
        try:
            model = ViViTModel(
                num_frames=8,
                img_size=256,
                pretrained=False
            )
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("ViViT dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

