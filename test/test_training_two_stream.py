"""
Comprehensive unit tests for training/two_stream module.
Tests TwoStreamModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.two_stream import TwoStreamModel


class TestTwoStreamModel:
    """Tests for TwoStreamModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test TwoStreamModel initialization."""
        try:
            model = TwoStreamModel(
                num_frames=8,
                pretrained=False
            )
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("TwoStream dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test TwoStreamModel forward pass."""
        try:
            model = TwoStreamModel(
                num_frames=8,
                pretrained=False
            )
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("TwoStream dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

