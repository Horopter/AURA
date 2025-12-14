"""
Comprehensive unit tests for data/config module.
Tests FVCConfig class.
"""
import pytest
import os
from pathlib import Path
from lib.data.config import FVCConfig


class TestFVCConfig:
    """Tests for FVCConfig class."""
    
    def test_default_initialization(self, temp_dir):
        """Test FVCConfig with default values."""
        # Override root_dir to use temp directory
        config = FVCConfig(root_dir=temp_dir)
        
        assert config.root_dir == temp_dir
        assert config.subsets == ("FVC1", "FVC2", "FVC3")
        assert config.main_metadata_filename == "FVC.csv"
        assert config.dup_metadata_filename == "FVC_dup.csv"
    
    def test_post_init_creates_directories(self, temp_dir):
        """Test __post_init__ creates necessary directories."""
        config = FVCConfig(root_dir=temp_dir)
        
        assert Path(config.data_dir).exists()
        assert config.videos_dir is not None
        assert config.metadata_dir is not None
        assert config.data_dir is not None
    
    def test_custom_metadata_filename(self, temp_dir):
        """Test FVCConfig with custom metadata filename."""
        config = FVCConfig(
            root_dir=temp_dir,
            main_metadata_filename="custom.csv"
        )
        
        assert config.main_metadata_filename == "custom.csv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
