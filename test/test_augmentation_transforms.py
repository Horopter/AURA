"""
Comprehensive unit tests for augmentation/transforms module.
Tests augmentation transforms with dummy frames.
"""
import pytest
import torch
from lib.augmentation.transforms import (
    RandomRotation,
    RandomAffine,
    RandomGaussianNoise,
    RandomGaussianBlur,
    RandomCutout,
    LetterboxResize,
    temporal_frame_drop,
    temporal_frame_duplicate,
    temporal_reverse,
    apply_simple_augmentation,
    build_comprehensive_frame_transforms,
    apply_temporal_augmentations,
)


class TestRandomRotation:
    """Tests for RandomRotation class."""
    
    def test_random_rotation_forward(self):
        """Test RandomRotation forward pass."""
        transform = RandomRotation(degrees=15)
        frame = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        
        result = transform(frame)
        
        assert result.shape == frame.shape


class TestRandomAffine:
    """Tests for RandomAffine class."""
    
    def test_random_affine_forward(self):
        """Test RandomAffine forward pass."""
        transform = RandomAffine(degrees=10, translate=(0.1, 0.1))
        frame = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        
        result = transform(frame)
        
        assert result.shape == frame.shape


class TestRandomGaussianNoise:
    """Tests for RandomGaussianNoise class."""
    
    def test_random_gaussian_noise_forward(self):
        """Test RandomGaussianNoise forward pass."""
        transform = RandomGaussianNoise(std=0.1)
        frame = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8).float()
        
        result = transform(frame)
        
        assert result.shape == frame.shape


class TestRandomGaussianBlur:
    """Tests for RandomGaussianBlur class."""
    
    def test_random_gaussian_blur_forward(self):
        """Test RandomGaussianBlur forward pass."""
        transform = RandomGaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        frame = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        
        result = transform(frame)
        
        assert result.shape == frame.shape


class TestRandomCutout:
    """Tests for RandomCutout class."""
    
    def test_random_cutout_forward(self):
        """Test RandomCutout forward pass."""
        transform = RandomCutout(size=(50, 50))
        frame = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        
        result = transform(frame)
        
        assert result.shape == frame.shape


class TestLetterboxResize:
    """Tests for LetterboxResize class."""
    
    def test_letterbox_resize_forward(self):
        """Test LetterboxResize forward pass."""
        transform = LetterboxResize(target_size=(256, 256))
        frame = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        
        result = transform(frame)
        
        assert result.shape == (3, 256, 256)


class TestTemporalAugmentations:
    """Tests for temporal augmentation functions."""
    
    def test_temporal_frame_drop(self):
        """Test temporal_frame_drop function."""
        frames = [torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8) for _ in range(10)]
        
        result = temporal_frame_drop(frames, drop_prob=0.1)
        
        assert isinstance(result, list)
        assert len(result) <= len(frames)
    
    def test_temporal_frame_duplicate(self):
        """Test temporal_frame_duplicate function."""
        frames = [torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8) for _ in range(10)]
        
        result = temporal_frame_duplicate(frames, dup_prob=0.1)
        
        assert isinstance(result, list)
        assert len(result) >= len(frames)
    
    def test_temporal_reverse(self):
        """Test temporal_reverse function."""
        frames = [torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8) for _ in range(10)]
        
        result = temporal_reverse(frames, reverse_prob=0.1)
        
        assert isinstance(result, list)
        assert len(result) == len(frames)


class TestApplySimpleAugmentation:
    """Tests for apply_simple_augmentation function."""
    
    def test_apply_simple_augmentation_basic(self):
        """Test apply_simple_augmentation with dummy frames."""
        frames = [torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8) for _ in range(10)]
        
        result = apply_simple_augmentation(frames, rotation_degrees=15)
        
        assert isinstance(result, list)
        assert len(result) == len(frames)


class TestBuildComprehensiveFrameTransforms:
    """Tests for build_comprehensive_frame_transforms function."""
    
    def test_build_comprehensive_frame_transforms_basic(self):
        """Test build_comprehensive_frame_transforms creates transforms."""
        transforms = build_comprehensive_frame_transforms(
            rotation_degrees=15,
            noise_std=0.1
        )
        
        assert isinstance(transforms, list)
        assert len(transforms) > 0


class TestApplyTemporalAugmentations:
    """Tests for apply_temporal_augmentations function."""
    
    def test_apply_temporal_augmentations_basic(self):
        """Test apply_temporal_augmentations with dummy frames."""
        frames = [torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8) for _ in range(10)]
        
        result = apply_temporal_augmentations(frames, drop_prob=0.1)
        
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
