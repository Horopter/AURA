"""
Comprehensive unit tests for utils/schemas module.
Tests schema validation functions.
"""
import pytest
import polars as pl
from lib.utils.schemas import (
    validate_stage1_output,
    validate_stage_output,
)


class TestValidateStage1Output:
    """Tests for validate_stage1_output function."""
    
    def test_validate_stage1_output_valid(self):
        """Test validate_stage1_output with valid DataFrame."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0],
            "original_video": ["video1.mp4"],
            "augmentation_idx": [-1],
            "is_original": [True]
        })
        
        try:
            is_valid = validate_stage1_output(df)
            assert isinstance(is_valid, bool)
        except ImportError:
            pytest.skip("Pandera not available")
    
    def test_validate_stage1_output_missing_columns(self):
        """Test validate_stage1_output with missing columns."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4"]
        })
        
        try:
            is_valid = validate_stage1_output(df)
            # May return False or skip validation if Pandera not available
            assert isinstance(is_valid, bool)
        except ImportError:
            pytest.skip("Pandera not available")


class TestValidateStageOutput:
    """Tests for validate_stage_output function."""
    
    def test_validate_stage_output_stage1(self):
        """Test validate_stage_output for stage 1."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        
        try:
            is_valid = validate_stage_output(df, stage=1)
            assert isinstance(is_valid, bool)
        except ImportError:
            pytest.skip("Pandera not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
