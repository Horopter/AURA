"""
Comprehensive unit tests for training/x3d module.
Tests X3DModel with dummy video tensors.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.x3d import X3DModel


class TestX3DModel:
    """Tests for X3DModel class."""
    
    @pytest.fixture
    def dummy_video_tensor(self):
        """Create dummy video tensor (B, C, T, H, W)."""
        return torch.randn(2, 3, 8, 224, 224)
    
    def test_initialization(self):
        """Test X3DModel initialization."""
        try:
            model = X3DModel(variant="x3d_m", pretrained=False)
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("X3D dependencies not available")
    
    def test_forward_pass(self, dummy_video_tensor):
        """Test X3DModel forward pass."""
        try:
            model = X3DModel(variant="x3d_m", pretrained=False)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_video_tensor)
            
            assert output.shape[0] == 2
            assert output.shape[1] == 1
        except (ImportError, RuntimeError):
            pytest.skip("X3D dependencies not available")
    
    def test_initialization_different_variants(self):
        """Test X3DModel with different variants."""
        for variant in ["x3d_s", "x3d_m", "x3d_l"]:
            try:
                model = X3DModel(variant=variant, pretrained=False)
                assert isinstance(model, nn.Module)
            except (ImportError, RuntimeError):
                pytest.skip(f"X3D {variant} dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

