"""
Comprehensive unit tests for utils/paths module.
Tests path resolution and validation functions.
"""
import pytest
import polars as pl
import tempfile
from pathlib import Path
from lib.utils.paths import (
    resolve_video_path,
    get_video_path_candidates,
    check_video_path_exists,
    find_metadata_file,
    load_metadata_flexible,
    validate_metadata_columns,
    validate_video_file,
    calculate_adaptive_num_frames,
    write_metadata_atomic,
    get_video_metadata_cache_path,
)


class TestResolveVideoPath:
    """Tests for resolve_video_path function."""
    
    @pytest.fixture
    def temp_project(self, temp_dir):
        """Create temporary project structure."""
        project_root = Path(temp_dir)
        videos_dir = project_root / "videos"
        videos_dir.mkdir()
        (videos_dir / "video1.mp4").write_text("dummy")
        return str(project_root)
    
    def test_resolve_video_path_with_videos_prefix(self, temp_project):
        """Test resolve_video_path with videos/ prefix strategy."""
        path = resolve_video_path("video1.mp4", temp_project)
        assert "video1.mp4" in path
        assert Path(path).exists()
    
    def test_resolve_video_path_direct(self, temp_dir):
        """Test resolve_video_path with direct path."""
        project_root = Path(temp_dir)
        (project_root / "video1.mp4").write_text("dummy")
        
        path = resolve_video_path("video1.mp4", str(project_root))
        assert Path(path).exists()
    
    def test_resolve_video_path_empty(self, temp_dir):
        """Test resolve_video_path raises error for empty path."""
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_video_path("", temp_dir)


class TestGetVideoPathCandidates:
    """Tests for get_video_path_candidates function."""
    
    def test_get_video_path_candidates_basic(self, temp_dir):
        """Test get_video_path_candidates returns list."""
        candidates = get_video_path_candidates("video.mp4", temp_dir)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
    
    def test_get_video_path_candidates_empty(self, temp_dir):
        """Test get_video_path_candidates with empty path."""
        candidates = get_video_path_candidates("", temp_dir)
        assert candidates == []


class TestCheckVideoPathExists:
    """Tests for check_video_path_exists function."""
    
    def test_check_video_path_exists_true(self, temp_dir):
        """Test check_video_path_exists returns True for existing file."""
        project_root = Path(temp_dir)
        video_file = project_root / "videos" / "video.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.write_text("dummy")
        
        assert check_video_path_exists("video.mp4", str(project_root)) is True
    
    def test_check_video_path_exists_false(self, temp_dir):
        """Test check_video_path_exists returns False for non-existing file."""
        assert check_video_path_exists("nonexistent.mp4", temp_dir) is False


class TestFindMetadataFile:
    """Tests for find_metadata_file function."""
    
    def test_find_metadata_file_exists(self, temp_dir):
        """Test find_metadata_file finds existing file."""
        base_path = Path(temp_dir)
        metadata_file = base_path / "metadata.parquet"
        metadata_file.write_text("dummy")
        
        found = find_metadata_file(base_path, "metadata")
        assert found == metadata_file
    
    def test_find_metadata_file_not_exists(self, temp_dir):
        """Test find_metadata_file returns None for non-existing file."""
        base_path = Path(temp_dir)
        found = find_metadata_file(base_path, "nonexistent")
        assert found is None


class TestLoadMetadataFlexible:
    """Tests for load_metadata_flexible function."""
    
    def test_load_metadata_flexible_parquet(self, temp_dir):
        """Test load_metadata_flexible loads parquet file."""
        metadata_path = Path(temp_dir) / "metadata.parquet"
        df = pl.DataFrame({
            "video_path": ["video1.mp4", "video2.mp4"],
            "label": [0, 1]
        })
        df.write_parquet(metadata_path)
        
        loaded = load_metadata_flexible(str(metadata_path))
        assert loaded is not None
        assert loaded.height == 2
    
    def test_load_metadata_flexible_not_exists(self, temp_dir):
        """Test load_metadata_flexible returns None for non-existing file."""
        loaded = load_metadata_flexible(str(Path(temp_dir) / "nonexistent.parquet"))
        assert loaded is None


class TestValidateMetadataColumns:
    """Tests for validate_metadata_columns function."""
    
    def test_validate_metadata_columns_valid(self):
        """Test validate_metadata_columns with valid columns."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        
        # Should not raise
        validate_metadata_columns(df, ["video_path", "label"])
    
    def test_validate_metadata_columns_missing(self):
        """Test validate_metadata_columns raises error for missing columns."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4"]
        })
        
        with pytest.raises(ValueError, match="Missing required column"):
            validate_metadata_columns(df, ["video_path", "label"])


class TestValidateVideoFile:
    """Tests for validate_video_file function."""
    
    def test_validate_video_file_not_exists(self, temp_dir):
        """Test validate_video_file returns False for non-existing file."""
        video_path = Path(temp_dir) / "nonexistent.mp4"
        is_valid, reason = validate_video_file(str(video_path))
        assert is_valid is False
        assert "not found" in reason.lower()


class TestCalculateAdaptiveNumFrames:
    """Tests for calculate_adaptive_num_frames function."""
    
    def test_calculate_adaptive_num_frames_basic(self):
        """Test calculate_adaptive_num_frames returns valid number."""
        num_frames = calculate_adaptive_num_frames(
            total_frames=100,
            target_duration_sec=5.0,
            fps=30.0
        )
        assert isinstance(num_frames, int)
        assert num_frames > 0


class TestWriteMetadataAtomic:
    """Tests for write_metadata_atomic function."""
    
    def test_write_metadata_atomic_parquet(self, temp_dir):
        """Test write_metadata_atomic writes parquet file."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        output_path = Path(temp_dir) / "output.parquet"
        
        write_metadata_atomic(df, output_path)
        
        assert output_path.exists()
        loaded = pl.read_parquet(output_path)
        assert loaded.height == 1


class TestGetVideoMetadataCachePath:
    """Tests for get_video_metadata_cache_path function."""
    
    def test_get_video_metadata_cache_path_basic(self, temp_dir):
        """Test get_video_metadata_cache_path returns path."""
        cache_path = get_video_metadata_cache_path(Path(temp_dir))
        assert cache_path is not None or isinstance(cache_path, Path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
