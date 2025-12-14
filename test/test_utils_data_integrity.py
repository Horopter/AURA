"""
Comprehensive unit tests for utils/data_integrity module.
Tests DataIntegrityChecker with dummy data.
"""
import pytest
import polars as pl
import numpy as np
from pathlib import Path
from lib.utils.data_integrity import DataIntegrityChecker


class TestDataIntegrityChecker:
    """Tests for DataIntegrityChecker class."""
    
    @pytest.fixture
    def dummy_metadata_file(self, temp_dir):
        """Create dummy metadata file."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4", "video2.mp4"],
            "label": [0, 1]
        })
        metadata_path = Path(temp_dir) / "metadata.parquet"
        df.write_parquet(metadata_path)
        return metadata_path
    
    def test_validate_metadata_file_valid(self, dummy_metadata_file):
        """Test validate_metadata_file with valid file."""
        is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
            dummy_metadata_file,
            required_columns={"video_path", "label"},
            min_rows=1
        )
        
        assert is_valid is True
        assert error == "OK"
        assert df is not None
        assert df.height == 2
    
    def test_validate_metadata_file_missing_columns(self, dummy_metadata_file):
        """Test validate_metadata_file with missing columns."""
        is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
            dummy_metadata_file,
            required_columns={"video_path", "label", "missing_col"},
            min_rows=1
        )
        
        assert is_valid is False
        assert "Missing required columns" in error
    
    def test_validate_metadata_file_empty(self, temp_dir):
        """Test validate_metadata_file with empty file."""
        empty_df = pl.DataFrame({"video_path": [], "label": []})
        metadata_path = Path(temp_dir) / "empty.parquet"
        empty_df.write_parquet(metadata_path)
        
        is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
            metadata_path,
            allow_empty=False
        )
        
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_feature_file_exists(self, temp_dir):
        """Test validate_feature_file with existing file."""
        feature_file = Path(temp_dir) / "features.npy"
        features = np.random.randn(10, 50)
        np.save(feature_file, features)
        
        is_valid, error, features_array = DataIntegrityChecker.validate_feature_file(
            feature_file
        )
        
        assert is_valid is True
        assert features_array is not None
        assert features_array.shape == (10, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
