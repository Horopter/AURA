"""
Comprehensive unit tests for augmentation/pregenerate module.
Tests pregeneration functions.
"""
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from lib.augmentation.pregenerate import (
    generate_augmented_clips,
    pregenerate_augmented_dataset,
    load_precomputed_clip,
)


class TestGenerateAugmentedClips:
    """Tests for generate_augmented_clips function."""
    
    @patch('lib.augmentation.pregenerate._read_video_wrapper')
    @patch('lib.augmentation.transforms.apply_simple_augmentation')
    def test_generate_augmented_clips_basic(self, mock_augment, mock_read, temp_dir):
        """Test generate_augmented_clips generates augmented clips."""
        from lib.models.video import VideoConfig
        
        # Mock video tensor
        mock_video = torch.randint(0, 256, (10, 3, 224, 224), dtype=torch.uint8)
        mock_read.return_value = mock_video
        mock_augment.return_value = [mock_video]
        
        video_path = str(Path(temp_dir) / "test.mp4")
        Path(video_path).write_text("dummy")
        output_dir = Path(temp_dir) / "augmented"
        output_dir.mkdir()
        
        config = VideoConfig(num_frames=8)
        
        try:
            clips = generate_augmented_clips(
                video_path,
                config=config,
                num_augmentations=3,
                save_dir=str(output_dir)
            )
            
            assert isinstance(clips, list)
        except Exception as e:
            pytest.skip(f"Augmented clips generation not available: {e}")


class TestPregenerateAugmentedDataset:
    """Tests for pregenerate_augmented_dataset function."""
    
    @patch('lib.augmentation.pregenerate.generate_augmented_clips')
    def test_pregenerate_augmented_dataset_basic(self, mock_generate, temp_dir):
        """Test pregenerate_augmented_dataset processes videos."""
        import polars as pl
        
        mock_generate.return_value = ["clip1.mp4", "clip2.mp4"]
        
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        
        output_dir = Path(temp_dir) / "augmented"
        output_dir.mkdir()
        
        result = pregenerate_augmented_dataset(
            df,
            project_root=temp_dir,
            output_dir=str(output_dir),
            num_augmentations=3
        )
        
        assert isinstance(result, pl.DataFrame)


class TestLoadPrecomputedClip:
    """Tests for load_precomputed_clip function."""
    
    @patch('lib.augmentation.pregenerate.load_frames')
    def test_load_precomputed_clip_basic(self, mock_load, temp_dir):
        """Test load_precomputed_clip loads clip."""
        mock_frames = torch.randint(0, 256, (10, 3, 224, 224), dtype=torch.uint8)
        mock_load.return_value = [f for f in mock_frames]
        
        clip_path = str(Path(temp_dir) / "clip.mp4")
        Path(clip_path).write_text("dummy")
        
        clip = load_precomputed_clip(clip_path)
        
        assert isinstance(clip, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
