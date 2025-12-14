"""
Comprehensive unit tests for training/_cnn module.
Tests NaiveCNNBaseline with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training._cnn import NaiveCNNBaseline


class TestNaiveCNNBaseline:
    """Tests for NaiveCNNBaseline class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test NaiveCNNBaseline initialization."""
        model = NaiveCNNBaseline(num_frames=8, num_classes=2)
        assert model.num_frames == 8
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test forward pass with dummy video tensor."""
        model = NaiveCNNBaseline(num_frames=8, num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_video_tensor)
        
        assert output.shape == (2, 2)  # (batch, num_classes)
    
    def test_forward_pass_different_batch_size(self):
        """Test forward pass with different batch sizes."""
        model = NaiveCNNBaseline(num_frames=8, num_classes=2)
        model.eval()
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 8, 224, 224)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 2)
    
    def test_forward_pass_different_frames(self):
        """Test forward pass with different number of frames."""
        model = NaiveCNNBaseline(num_frames=16, num_classes=2)
        model.eval()
        
        x = torch.randn(2, 3, 16, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 2)
    
    def test_training_mode(self, dummy_video_tensor):
        """Test model in training mode."""
        model = NaiveCNNBaseline(num_frames=8, num_classes=2)
        model.train()
        
        output = model(dummy_video_tensor)
        assert output.shape == (2, 2)
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = NaiveCNNBaseline(num_frames=8, num_classes=2)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

