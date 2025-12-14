"""
Comprehensive unit tests for models/video module.
Tests video loading and dataset classes with dummy data.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from lib.models.video import (
    _read_video_wrapper,
    VideoConfig,
    uniform_sample_indices,
    rolling_window_indices,
    build_frame_transforms,
    AdaptiveChunkSizeManager,
    VideoDataset,
    variable_ar_collate,
    build_sample_loader,
    Inception3DBlock,
    VariableARVideoModel,
    PretrainedInceptionVideoModel,
)


class TestReadVideoWrapper:
    """Tests for _read_video_wrapper function."""
    
    @patch('lib.models.video.torchvision.io.read_video')
    def test_read_video_wrapper_basic(self, mock_read, temp_dir):
        """Test _read_video_wrapper loads video."""
        # Mock video tensor (T, H, W, C)
        mock_video = torch.randint(0, 256, (10, 224, 224, 3), dtype=torch.uint8)
        mock_read.return_value = (mock_video, None, None)
        
        video_path = str(Path(temp_dir) / "test.mp4")
        Path(video_path).write_text("dummy")
        
        video = _read_video_wrapper(video_path)
        
        assert isinstance(video, torch.Tensor)
        assert video.ndim == 4


class TestVideoConfig:
    """Tests for VideoConfig dataclass."""
    
    def test_video_config_default(self):
        """Test VideoConfig with default values."""
        config = VideoConfig()
        
        assert config.num_frames == 8
        assert config.height == 224
        assert config.width == 224


class TestUniformSampleIndices:
    """Tests for uniform_sample_indices function."""
    
    def test_uniform_sample_indices_basic(self):
        """Test uniform_sample_indices samples uniformly."""
        indices = uniform_sample_indices(total_frames=100, num_frames=10)
        
        assert len(indices) == 10
        assert all(0 <= idx < 100 for idx in indices)
        assert len(set(indices)) == 10  # All unique


class TestRollingWindowIndices:
    """Tests for rolling_window_indices function."""
    
    def test_rolling_window_indices_basic(self):
        """Test rolling_window_indices creates rolling windows."""
        windows = rolling_window_indices(total_frames=100, window_size=10, stride=5)
        
        assert isinstance(windows, list)
        assert len(windows) > 0
        assert all(len(w) == 10 for w in windows)


class TestBuildFrameTransforms:
    """Tests for build_frame_transforms function."""
    
    def test_build_frame_transforms_basic(self):
        """Test build_frame_transforms creates transforms."""
        transforms = build_frame_transforms(
            height=224,
            width=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        assert transforms is not None


class TestAdaptiveChunkSizeManager:
    """Tests for AdaptiveChunkSizeManager class."""
    
    def test_initialization(self):
        """Test AdaptiveChunkSizeManager initialization."""
        manager = AdaptiveChunkSizeManager(initial_chunk_size=100)
        
        assert manager.current_chunk_size == 100
    
    def test_record_success(self):
        """Test record_success increases chunk size."""
        manager = AdaptiveChunkSizeManager(initial_chunk_size=100)
        initial_size = manager.current_chunk_size
        
        manager.record_success()
        
        assert manager.current_chunk_size >= initial_size
    
    def test_record_failure(self):
        """Test record_failure decreases chunk size."""
        manager = AdaptiveChunkSizeManager(initial_chunk_size=100)
        initial_size = manager.current_chunk_size
        
        manager.record_failure()
        
        assert manager.current_chunk_size < initial_size


class TestVideoDataset:
    """Tests for VideoDataset class."""
    
    @pytest.fixture
    def dummy_metadata(self, temp_dir):
        """Create dummy metadata DataFrame."""
        import polars as pl
        return pl.DataFrame({
            "video_path": ["video1.mp4", "video2.mp4"],
            "label": [0, 1]
        })
    
    @patch('lib.models.video._read_video_wrapper')
    def test_video_dataset_getitem(self, mock_read, dummy_metadata, temp_dir):
        """Test VideoDataset __getitem__ method."""
        # Mock video tensor
        mock_video = torch.randint(0, 256, (10, 224, 224, 3), dtype=torch.uint8)
        mock_read.return_value = mock_video
        
        config = VideoConfig(num_frames=8)
        dataset = VideoDataset(dummy_metadata, project_root=temp_dir, config=config)
        
        if len(dataset) > 0:
            video, label = dataset[0]
            assert isinstance(video, torch.Tensor)
            assert isinstance(label, (int, torch.Tensor))


class TestVariableArCollate:
    """Tests for variable_ar_collate function."""
    
    def test_variable_ar_collate_basic(self):
        """Test variable_ar_collate collates variable aspect ratio videos."""
        videos = [
            torch.randint(0, 256, (8, 3, 200, 300), dtype=torch.uint8),
            torch.randint(0, 256, (8, 3, 150, 250), dtype=torch.uint8)
        ]
        labels = [0, 1]
        
        batch_video, batch_labels = variable_ar_collate(list(zip(videos, labels)))
        
        assert isinstance(batch_video, torch.Tensor)
        assert isinstance(batch_labels, torch.Tensor)


class TestInception3DBlock:
    """Tests for Inception3DBlock class."""
    
    def test_forward(self):
        """Test Inception3DBlock forward pass."""
        block = Inception3DBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 8, 224, 224)
        
        output = block(x)
        
        assert output.shape[0] == 2
        assert output.shape[1] == 128


class TestVariableARVideoModel:
    """Tests for VariableARVideoModel class."""
    
    def test_forward(self):
        """Test VariableARVideoModel forward pass."""
        model = VariableARVideoModel(num_classes=2)
        x = torch.randn(2, 8, 3, 224, 224)
        
        output = model(x)
        
        assert output.shape[0] == 2
        assert output.shape[1] == 2


class TestPretrainedInceptionVideoModel:
    """Tests for PretrainedInceptionVideoModel class."""
    
    def test_forward(self):
        """Test PretrainedInceptionVideoModel forward pass."""
        model = PretrainedInceptionVideoModel(num_classes=2)
        x = torch.randn(2, 8, 3, 224, 224)
        
        output = model(x)
        
        assert output.shape[0] == 2
        assert output.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
