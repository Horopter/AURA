"""
Comprehensive unit tests for training/_linear module.
Tests LogisticRegressionBaseline with dummy features.
"""
import pytest
import numpy as np
import polars as pl
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from lib.training._linear import LogisticRegressionBaseline


class TestLogisticRegressionBaseline:
    """Tests for LogisticRegressionBaseline class."""
    
    @pytest.fixture
    def dummy_features_df(self, temp_dir):
        """Create dummy features DataFrame."""
        n_samples = 100
        n_features = 50
        data = {
            "video_path": [f"video_{i}.mp4" for i in range(n_samples)],
            "label": np.random.randint(0, 2, n_samples).tolist(),
        }
        # Add feature columns
        for i in range(n_features):
            data[f"feature_{i}"] = np.random.randn(n_samples).tolist()
        
        df = pl.DataFrame(data)
        # Save to temp file
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
        """Test LogisticRegressionBaseline initialization."""
        model = LogisticRegressionBaseline(
            features_stage2_path="/tmp/features.parquet",
            use_stage2_only=True
        )
        assert model.num_frames == 1000  # Default
        assert model.use_stage2_only is True
        assert model.is_fitted is False
        assert model.model is not None
        assert model.scaler is not None
    
    def test_initialization_with_stage4(self):
        """Test initialization with Stage 4 features."""
        model = LogisticRegressionBaseline(
            features_stage2_path="/tmp/stage2.parquet",
            features_stage4_path="/tmp/stage4.parquet",
            use_stage2_only=False
        )
        assert model.use_stage2_only is False
        assert model.features_stage4_path == "/tmp/stage4.parquet"
    
    @patch('lib.training._linear.load_metadata_flexible')
    @patch('lib.training._linear.load_and_combine_features')
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
        
        # Mock metadata loading
        mock_load_metadata.return_value = features_df
        
        # Mock feature loading
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
        dummy_features = features_df.select(feature_cols).to_numpy()
        dummy_labels = features_df["label"].to_numpy()
        
        def mock_load_combine(video_paths, stage2_path, stage4_path, project_root):
            # Return features matching video_paths
            indices = [features_df["video_path"].to_list().index(vp) for vp in video_paths]
            return dummy_features[indices], dummy_labels[indices]
        
        mock_load_features.side_effect = mock_load_combine
        
        model = LogisticRegressionBaseline(
            features_stage2_path=features_path,
            use_stage2_only=True
        )
        
        # Test fit
        model.fit(dummy_train_df, temp_dir)
        
        assert model.is_fitted is True
        assert model.model is not None
        assert model.scaler is not None
    
    def test_fit_missing_stage2_path(self, dummy_train_df, temp_dir):
        """Test fit() raises error when Stage 2 path is missing."""
        model = LogisticRegressionBaseline(
            features_stage2_path=None,
            use_stage2_only=True
        )
        
        with pytest.raises(ValueError, match="Stage 2 features path is REQUIRED"):
            model.fit(dummy_train_df, temp_dir)
    
    @patch('lib.training._linear.load_metadata_flexible')
    @patch('lib.training._linear.load_and_combine_features')
    def test_predict_with_dummy_features(
        self,
        mock_load_features,
        mock_load_metadata,
        dummy_features_df,
        dummy_train_df,
        temp_dir
    ):
        """Test predict() with dummy features."""
        features_path, features_df = dummy_features_df
        
        # Mock metadata loading
        mock_load_metadata.return_value = features_df
        
        # Mock feature loading
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
        dummy_features = features_df.select(feature_cols).to_numpy()
        dummy_labels = features_df["label"].to_numpy()
        
        def mock_load_combine(video_paths, stage2_path, stage4_path, project_root):
            indices = [features_df["video_path"].to_list().index(vp) for vp in video_paths]
            return dummy_features[indices], dummy_labels[indices]
        
        mock_load_features.side_effect = mock_load_combine
        
        model = LogisticRegressionBaseline(
            features_stage2_path=features_path,
            use_stage2_only=True
        )
        
        # Fit first
        model.fit(dummy_train_df, temp_dir)
        
        # Test predict
        test_df = pl.DataFrame({
            "video_path": ["video_0.mp4", "video_1.mp4"],
            "label": [0, 1]
        })
        
        probs = model.predict(test_df, temp_dir)
        
        assert probs.shape[0] == 2
        assert probs.shape[1] == 2  # Binary classification
        assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_predict_not_fitted(self, temp_dir):
        """Test predict() raises error when model not fitted."""
        model = LogisticRegressionBaseline(
            features_stage2_path="/tmp/features.parquet"
        )
        
        test_df = pl.DataFrame({
            "video_path": ["video_0.mp4"],
            "label": [0]
        })
        
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(test_df, temp_dir)
    
    @patch('lib.training._linear.load_metadata_flexible')
    @patch('lib.training._linear.load_and_combine_features')
    def test_save_load(
        self,
        mock_load_features,
        mock_load_metadata,
        dummy_features_df,
        dummy_train_df,
        temp_dir
    ):
        """Test save() and load() methods."""
        features_path, features_df = dummy_features_df
        
        mock_load_metadata.return_value = features_df
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
        dummy_features = features_df.select(feature_cols).to_numpy()
        dummy_labels = features_df["label"].to_numpy()
        
        def mock_load_combine(video_paths, stage2_path, stage4_path, project_root):
            indices = [features_df["video_path"].to_list().index(vp) for vp in video_paths]
            return dummy_features[indices], dummy_labels[indices]
        
        mock_load_features.side_effect = mock_load_combine
        
        # Create and fit model
        model = LogisticRegressionBaseline(
            features_stage2_path=features_path,
            use_stage2_only=True
        )
        model.fit(dummy_train_df, temp_dir)
        
        # Save
        save_dir = Path(temp_dir) / "saved_model"
        model.save(str(save_dir))
        
        assert (save_dir / "model.joblib").exists()
        assert (save_dir / "scaler.joblib").exists()
        assert (save_dir / "metadata.json").exists()
        
        # Load
        loaded_model = LogisticRegressionBaseline(
            features_stage2_path=features_path
        )
        loaded_model.load(str(save_dir))
        
        assert loaded_model.is_fitted is True
        assert loaded_model.model is not None
        assert loaded_model.scaler is not None
        assert loaded_model.feature_indices == model.feature_indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

