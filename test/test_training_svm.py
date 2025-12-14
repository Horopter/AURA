"""
Comprehensive unit tests for training/_svm module.
Tests SVMBaseline with dummy features.
"""
import pytest
import numpy as np
import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from lib.training._svm import SVMBaseline


class TestSVMBaseline:
    """Tests for SVMBaseline class."""
    
    @pytest.fixture
    def dummy_features_df(self, temp_dir):
        """Create dummy features DataFrame."""
        n_samples = 100
        n_features = 50
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
    
    @pytest.fixture
    def dummy_train_df(self):
        """Create dummy training DataFrame."""
        return pl.DataFrame({
            "video_path": [f"video_{i}.mp4" for i in range(80)],
            "label": np.random.randint(0, 2, 80).tolist()
        })
    
    def test_initialization(self):
        """Test SVMBaseline initialization."""
        model = SVMBaseline(
            features_stage2_path="/tmp/features.parquet",
            use_stage2_only=True
        )
        assert model.use_stage2_only is True
        assert model.is_fitted is False
        assert model.model is not None
        assert model.scaler is not None
    
    @patch('lib.training._svm.load_metadata_flexible')
    @patch('lib.training._svm.load_and_combine_features')
    def test_fit_with_dummy_features(
        self,
        mock_load_features,
        mock_load_metadata,
        dummy_features_df,
        dummy_train_df,
        temp_dir
    ):
        """Test fit() with dummy features."""
        features_path, features_df = dummy_features_df
        mock_load_metadata.return_value = features_df
        
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
        dummy_features = features_df.select(feature_cols).to_numpy()
        dummy_labels = features_df["label"].to_numpy()
        
        def mock_load_combine(video_paths, stage2_path, stage4_path, project_root):
            indices = [features_df["video_path"].to_list().index(vp) for vp in video_paths]
            return dummy_features[indices], dummy_labels[indices]
        
        mock_load_features.side_effect = mock_load_combine
        
        model = SVMBaseline(
            features_stage2_path=features_path,
            use_stage2_only=True
        )
        
        model.fit(dummy_train_df, temp_dir)
        
        assert model.is_fitted is True
    
    def test_fit_missing_stage2_path(self, dummy_train_df, temp_dir):
        """Test fit() raises error when Stage 2 path is missing."""
        model = SVMBaseline(features_stage2_path=None)
        
        with pytest.raises(ValueError, match="Stage 2 features path is REQUIRED"):
            model.fit(dummy_train_df, temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

