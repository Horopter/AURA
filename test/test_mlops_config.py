"""
Comprehensive unit tests for mlops/config module.
Tests MLOps configuration classes.
"""
import pytest
from pathlib import Path
from lib.mlops.config import (
    RunConfig,
    ExperimentTracker,
    CheckpointManager,
    DataVersionManager,
    create_run_directory,
)


class TestRunConfig:
    """Tests for RunConfig class."""
    
    def test_run_config_default(self):
        """Test RunConfig with default values."""
        config = RunConfig(run_id="test_run", experiment_name="test_exp")
        
        assert config.run_id == "test_run"
        assert config.experiment_name == "test_exp"


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""
    
    def test_initialization(self, temp_dir):
        """Test ExperimentTracker initialization."""
        run_dir = str(Path(temp_dir) / "runs" / "test_run")
        tracker = ExperimentTracker(run_dir=run_dir)
        assert tracker is not None
        assert tracker.run_dir.exists()


class TestCheckpointManager:
    """Tests for CheckpointManager class."""
    
    def test_initialization(self, temp_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
            run_id="test_run"
        )
        assert manager.checkpoint_dir is not None


class TestDataVersionManager:
    """Tests for DataVersionManager class."""
    
    def test_initialization(self, temp_dir):
        """Test DataVersionManager initialization."""
        manager = DataVersionManager(data_dir=str(Path(temp_dir) / "data"))
        assert manager.data_dir is not None


class TestCreateRunDirectory:
    """Tests for create_run_directory function."""
    
    def test_create_run_directory_basic(self, temp_dir):
        """Test create_run_directory creates directory."""
        run_dir, run_id = create_run_directory(
            base_dir=str(temp_dir),
            experiment_name="test_exp"
        )
        
        assert Path(run_dir).exists()
        assert run_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
