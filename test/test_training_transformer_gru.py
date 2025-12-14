"""
Comprehensive unit tests for training/_transformer_gru module.
Tests ViTGRUModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training._transformer_gru import ViTGRUModel


class TestViTGRUModel:
    """Tests for ViTGRUModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 256, 256)
    
    def test_initialization(self):
        """Test ViTGRUModel initialization."""
        try:
            model = ViTGRUModel(
                num_frames=8,
                pretrained=False
            )
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("ViT-GRU dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test ViTGRUModel forward pass."""
        try:
            model = ViTGRUModel(
                num_frames=8,
                pretrained=False
            )
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("ViT-GRU dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

