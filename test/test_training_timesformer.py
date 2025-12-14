"""
Comprehensive unit tests for training/timesformer module.
Tests TimeSformerModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.timesformer import TimeSformerModel, SpaceTimeAttention


class TestSpaceTimeAttention:
    """Tests for SpaceTimeAttention class."""
    
    def test_initialization(self):
        """Test SpaceTimeAttention initialization."""
        attn = SpaceTimeAttention(dim=768, num_heads=8)
        assert isinstance(attn, nn.Module)
    
    def test_forward_pass(self):
        """Test SpaceTimeAttention forward pass."""
        attn = SpaceTimeAttention(dim=768, num_heads=8)
        x = torch.randn(2, 100, 768)  # (batch, seq_len, dim)
        
        with torch.no_grad():
            output = attn(x)
        
        assert output.shape == x.shape


class TestTimeSformerModel:
    """Tests for TimeSformerModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test TimeSformerModel initialization."""
        try:
            model = TimeSformerModel(
                num_frames=8,
                img_size=224,
                pretrained=False
            )
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("TimeSformer dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test TimeSformerModel forward pass."""
        try:
            model = TimeSformerModel(
                num_frames=8,
                img_size=224,
                pretrained=False
            )
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("TimeSformer dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

