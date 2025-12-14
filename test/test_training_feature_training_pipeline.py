"""
Comprehensive unit tests for training/feature_training_pipeline module.
Tests feature training pipeline with dummy features.
"""
import pytest
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import Mock, patch
from lib.training.feature_training_pipeline import (
    load_features_for_training,
    train_feature_model,
)


class TestLoadFeaturesForTraining:
    """Tests for load_features_for_training function."""
    
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
    
    @patch('lib.training.feature_training_pipeline.load_metadata_flexible')
    @patch('lib.training.feature_training_pipeline.load_and_combine_features')
    def test_load_features_for_training_basic(
        self,
        mock_load_features,
        mock_load_metadata,
        dummy_features_df,
        temp_dir
    ):
        """Test load_features_for_training with basic setup."""
        features_path, features_df = dummy_features_df
        mock_load_metadata.return_value = features_df
        
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
        dummy_features = features_df.select(feature_cols).to_numpy()
        dummy_labels = features_df["label"].to_numpy()
        
        def mock_load_combine(video_paths, stage2_path, stage4_path, project_root):
            indices = [features_df["video_path"].to_list().index(vp) for vp in video_paths]
            return dummy_features[indices], dummy_labels[indices]
        
        mock_load_features.side_effect = mock_load_combine
        
        train_df = pl.DataFrame({
            "video_path": ["video_0.mp4", "video_1.mp4"],
            "label": [0, 1]
        })
        
        features, labels, feature_names = load_features_for_training(
            train_df=train_df,
            features_stage2_path=features_path,
            features_stage4_path=None,
            project_root=temp_dir
        )
        
        assert features is not None
        assert features.shape[0] == 2
        assert len(labels) == 2
        assert len(feature_names) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

