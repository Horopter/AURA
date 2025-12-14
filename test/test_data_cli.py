"""
Comprehensive unit tests for data/cli module.
Tests CLI functions.
"""
import pytest
from unittest.mock import Mock, patch
from lib.data.cli import run_default_prep


class TestRunDefaultPrep:
    """Tests for run_default_prep function."""
    
    @patch('lib.data.cli.build_video_index')
    @patch('lib.data.cli.FVCConfig')
    def test_run_default_prep_basic(self, mock_config_class, mock_build_index):
        """Test run_default_prep calls build_video_index."""
        mock_config = Mock()
        mock_config.root_dir = "/tmp/test"
        mock_config.metadata_dir = "/tmp/test/metadata"
        mock_config.data_dir = "/tmp/test/data"
        mock_config_class.return_value = mock_config
        
        run_default_prep()
        
        mock_build_index.assert_called_once()
        call_args = mock_build_index.call_args
        assert call_args[1]['drop_duplicates'] is False
        assert call_args[1]['compute_stats'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
