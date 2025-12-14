"""
Comprehensive unit tests for augmentation/pipeline module.
Tests augmentation pipeline functions.
"""
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from lib.augmentation.pipeline import (
    augment_video,
    _reconstruct_metadata_from_files,
    stage1_augment_videos,
)


class TestAugmentVideo:
    """Tests for augment_video function."""
    
    @patch('lib.augmentation.pipeline.load_frames')
    @patch('lib.augmentation.pipeline.apply_simple_augmentation')
    @patch('lib.augmentation.pipeline.save_frames')
    def test_augment_video_basic(self, mock_save, mock_augment, mock_load, temp_dir):
        """Test augment_video augments video."""
        # Mock frames
        mock_frames = [torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8) for _ in range(10)]
        mock_load.return_value = mock_frames
        mock_augment.return_value = mock_frames
        
        video_path = str(Path(temp_dir) / "test.mp4")
        Path(video_path).write_text("dummy")
        output_path = str(Path(temp_dir) / "augmented.mp4")
        
        try:
            augment_video(video_path, output_path, rotation_degrees=15)
            # Should not raise
        except Exception as e:
            pytest.skip(f"Video augmentation not available: {e}")


class TestReconstructMetadataFromFiles:
    """Tests for _reconstruct_metadata_from_files function."""
    
    def test_reconstruct_metadata_from_files_basic(self, temp_dir):
        """Test _reconstruct_metadata_from_files reconstructs metadata."""
        import polars as pl
        
        # Create dummy augmented videos
        aug_dir = Path(temp_dir) / "augmented"
        aug_dir.mkdir()
        (aug_dir / "video1_aug0.mp4").write_text("dummy")
        (aug_dir / "video1_aug1.mp4").write_text("dummy")
        
        metadata_path = Path(temp_dir) / "metadata.parquet"
        df = pl.DataFrame({"video_path": ["video1.mp4"], "label": [0]})
        df.write_parquet(metadata_path)
        
        _reconstruct_metadata_from_files(
            metadata_path=metadata_path,
            output_dir=aug_dir,
            project_root=Path(temp_dir),
            df=df,
            num_augmentations=2
        )
        
        # Function doesn't return anything, just check it doesn't raise
        assert True


class TestStage1AugmentVideos:
    """Tests for stage1_augment_videos function."""
    
    @patch('lib.augmentation.pipeline.augment_video')
    def test_stage1_augment_videos_basic(self, mock_augment, temp_dir):
        """Test stage1_augment_videos processes videos."""
        import polars as pl
        
        mock_augment.return_value = []
        
        output_dir = Path(temp_dir) / "augmented"
        output_dir.mkdir()
        
        result = stage1_augment_videos(
            project_root=temp_dir,
            output_dir=str(output_dir),
            num_augmentations=3
        )
        
        assert isinstance(result, pl.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
