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
        config = RunConfig()
        
        assert config.project_root is not None
        assert config.experiment_name is not None


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""
    
    def test_initialization(self):
        """Test ExperimentTracker initialization."""
        tracker = ExperimentTracker()
        assert tracker is not None


class TestCheckpointManager:
    """Tests for CheckpointManager class."""
    
    def test_initialization(self, temp_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(checkpoint_dir=str(Path(temp_dir) / "checkpoints"))
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
        run_dir = create_run_directory(
            base_dir=temp_dir,
            experiment_name="test_exp"
        )
        
        assert Path(run_dir).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
