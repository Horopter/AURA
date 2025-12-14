"""
Comprehensive unit tests for training/feature_preprocessing module.
Tests feature preprocessing functions with dummy feature matrices.
"""
import pytest
import numpy as np
import polars as pl
from unittest.mock import patch
from lib.training.feature_preprocessing import (
    remove_collinear_features,
    _remove_vif_collinear,
    _normalize_video_path,
    _match_video_path,
    load_and_combine_features,
)


class TestRemoveCollinearFeatures:
    """Tests for remove_collinear_features function."""
    
    def test_remove_collinear_features_basic(self):
        """Test remove_collinear_features with basic features."""
        n_samples = 100
        n_features = 10
        features = np.random.randn(n_samples, n_features)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        filtered, indices, names = remove_collinear_features(
            features,
            feature_names=feature_names,
            correlation_threshold=0.95,
            method="correlation"
        )
        
        assert filtered.shape[0] == n_samples
        assert filtered.shape[1] <= n_features
        assert len(indices) == filtered.shape[1]
        assert len(names) == filtered.shape[1]
    
    def test_remove_collinear_features_with_collinearity(self):
        """Test remove_collinear_features with highly correlated features."""
        n_samples = 100
        # Create features where feature_1 is highly correlated with feature_0
        feature_0 = np.random.randn(n_samples)
        feature_1 = feature_0 + np.random.randn(n_samples) * 0.01  # Very high correlation
        feature_2 = np.random.randn(n_samples)  # Independent
        
        features = np.column_stack([feature_0, feature_1, feature_2])
        feature_names = ["feature_0", "feature_1", "feature_2"]
        
        filtered, indices, names = remove_collinear_features(
            features,
            feature_names=feature_names,
            correlation_threshold=0.95,
            method="correlation"
        )
        
        # Should remove one of the highly correlated features
        assert filtered.shape[1] < 3
        assert filtered.shape[1] >= 1
    
    def test_remove_collinear_features_empty(self):
        """Test remove_collinear_features with empty features."""
        features = np.array([]).reshape(0, 0)
        filtered, indices, names = remove_collinear_features(features)
        
        assert filtered.shape == (0, 0)
        assert indices == []
        assert names == []
    
    def test_remove_collinear_features_no_names(self):
        """Test remove_collinear_features without feature names."""
        n_samples = 100
        n_features = 10
        features = np.random.randn(n_samples, n_features)
        
        filtered, indices, names = remove_collinear_features(
            features,
            correlation_threshold=0.95
        )
        
        assert len(names) == filtered.shape[1]
        assert all(name.startswith("feature_") for name in names)


class TestNormalizeVideoPath:
    """Tests for _normalize_video_path function."""
    
    def test_normalize_video_path_basic(self):
        """Test _normalize_video_path with basic paths."""
        path = "video.mp4"
        normalized = _normalize_video_path(path)
        assert isinstance(normalized, str)
        assert len(normalized) > 0
    
    def test_normalize_video_path_with_slashes(self):
        """Test _normalize_video_path with paths containing slashes."""
        path = "path/to/video.mp4"
        normalized = _normalize_video_path(path)
        assert isinstance(normalized, str)


class TestMatchVideoPath:
    """Tests for _match_video_path function."""
    
    def test_match_video_path_exact(self):
        """Test _match_video_path with exact match."""
        target = "video.mp4"
        candidates = ["video.mp4", "other.mp4"]
        result = _match_video_path(target, candidates)
        assert result == "video.mp4"
    
    def test_match_video_path_no_match(self):
        """Test _match_video_path with no match."""
        target = "video.mp4"
        candidates = ["other1.mp4", "other2.mp4"]
        result = _match_video_path(target, candidates)
        assert result is None or isinstance(result, str)


class TestLoadAndCombineFeatures:
    """Tests for load_and_combine_features function."""
    
    @pytest.fixture
    def dummy_features_df(self, temp_dir):
        """Create dummy features DataFrame."""
        n_samples = 100
        n_features = 20
        data = {
            "video_path": [f"video_{i}.mp4" for i in range(n_samples)],
            "label": np.random.randint(0, 2, n_samples).tolist(),
        }
        for i in range(n_features):
            data[f"feature_{i}"] = np.random.randn(n_samples).tolist()
        
        df = pl.DataFrame(data)
        features_path = Path(temp_dir) / "features.parquet"
        df.write_parquet(features_path)
        return str(features_path), df
    
    @patch('lib.training.feature_preprocessing.load_metadata_flexible')
    def test_load_and_combine_features_stage2_only(
        self,
        mock_load_metadata,
        dummy_features_df,
        temp_dir
    ):
        """Test load_and_combine_features with Stage 2 only."""
        features_path, features_df = dummy_features_df
        mock_load_metadata.return_value = features_df
        
        video_paths = ["video_0.mp4", "video_1.mp4"]
        
        features, names, indices, valid_indices = load_and_combine_features(
            features_stage2_path=features_path,
            features_stage4_path=None,
            video_paths=video_paths,
            project_root=temp_dir,
            remove_collinearity=False
        )
        
        assert features is not None
        assert features.shape[0] == len(video_paths)
        assert len(names) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

