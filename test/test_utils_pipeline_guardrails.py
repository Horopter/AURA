"""
Comprehensive unit tests for utils/pipeline_guardrails module.
Tests pipeline guardrails with dummy data.
"""
import pytest
from pathlib import Path
from lib.utils.pipeline_guardrails import PipelineGuardrails


class TestPipelineGuardrails:
    """Tests for PipelineGuardrails class."""
    
    @pytest.fixture
    def pipeline_guardrails(self, temp_dir):
        """Create PipelineGuardrails instance."""
        return PipelineGuardrails(project_root=temp_dir, strict_mode=False)
    
    def test_initialization(self, temp_dir):
        """Test PipelineGuardrails initialization."""
        guardrails = PipelineGuardrails(project_root=temp_dir)
        assert guardrails.project_root == Path(temp_dir).resolve()
        assert guardrails.monitor is not None
    
    def test_validate_stage1_output(self, pipeline_guardrails, temp_dir):
        """Test validate_stage1_output."""
        # Create dummy stage1 output
        aug_dir = Path(temp_dir) / "data" / "augmented_videos"
        aug_dir.mkdir(parents=True, exist_ok=True)
        
        import polars as pl
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        df.write_parquet(aug_dir / "augmented_metadata.parquet")
        
        is_valid, errors, info = pipeline_guardrails.validate_stage1_output()
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        assert isinstance(info, dict)
    
    def test_validate_stage2_output(self, pipeline_guardrails, temp_dir):
        """Test validate_stage2_output."""
        # Create dummy stage2 output
        features_dir = Path(temp_dir) / "data" / "features_stage2"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        import polars as pl
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0],
            "features_path": ["features1.npy"]
        })
        df.write_parquet(features_dir / "features_metadata.parquet")
        
        is_valid, errors, info = pipeline_guardrails.validate_stage2_output()
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
