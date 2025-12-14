"""
Comprehensive unit tests for training/video_training_pipeline module.
Tests video training pipeline functions.
"""
import pytest
import polars as pl
from lib.training.video_training_pipeline import (
    is_feature_based,
    is_video_based,
    train_video_model,
)


class TestIsFeatureBased:
    """Tests for is_feature_based function."""
    
    def test_is_feature_based_true(self):
        """Test is_feature_based returns True for feature-based models."""
        assert is_feature_based("logistic_regression") is True
        assert is_feature_based("svm") is True
        assert is_feature_based("logistic_regression_stage2") is True
    
    def test_is_feature_based_false(self):
        """Test is_feature_based returns False for video-based models."""
        assert is_feature_based("slowfast") is False
        assert is_feature_based("x3d") is False
        assert is_feature_based("naive_cnn") is False


class TestIsVideoBased:
    """Tests for is_video_based function."""
    
    def test_is_video_based_true(self):
        """Test is_video_based returns True for video-based models."""
        assert is_video_based("slowfast") is True
        assert is_video_based("x3d") is True
        assert is_video_based("naive_cnn") is True
    
    def test_is_video_based_false(self):
        """Test is_video_based returns False for feature-based models."""
        assert is_video_based("logistic_regression") is False
        assert is_video_based("svm") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

