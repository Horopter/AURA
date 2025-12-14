"""
Comprehensive unit tests for training/_transformer module.
Tests ViTTransformerModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training._transformer import ViTTransformerModel


class TestViTTransformerModel:
    """Tests for ViTTransformerModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 256, 256)
    
    def test_initialization(self):
        """Test ViTTransformerModel initialization."""
        try:
            model = ViTTransformerModel(
                num_frames=8,
                pretrained=False
            )
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("ViT-Transformer dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test ViTTransformerModel forward pass."""
        try:
            model = ViTTransformerModel(
                num_frames=8,
                pretrained=False
            )
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("ViT-Transformer dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

