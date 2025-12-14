"""
Comprehensive unit tests for mlops/cleanup module.
Tests cleanup functions.
"""
import pytest
from pathlib import Path
from lib.mlops.cleanup import (
    cleanup_runs_and_logs,
    cleanup_intermediate_files,
)


class TestCleanupRunsAndLogs:
    """Tests for cleanup_runs_and_logs function."""
    
    def test_cleanup_runs_and_logs_basic(self, temp_dir):
        """Test cleanup_runs_and_logs doesn't raise errors."""
        cleanup_runs_and_logs(
            project_root=temp_dir,
            keep_models=False,
            keep_intermediate_data=False
        )
        # Should not raise


class TestCleanupIntermediateFiles:
    """Tests for cleanup_intermediate_files function."""
    
    def test_cleanup_intermediate_files_basic(self, temp_dir):
        """Test cleanup_intermediate_files doesn't raise errors."""
        cleanup_intermediate_files(
            project_root=temp_dir,
            keep_augmentations=True
        )
        # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
