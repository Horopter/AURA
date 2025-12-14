"""
Comprehensive unit tests for training/feature_pipeline module.
Tests feature pipeline functions with dummy feature arrays.
"""
import pytest
import numpy as np
import torch
from lib.training.feature_pipeline import (
    FeatureDataset,
    FeaturePreprocessor,
    create_stratified_splits,
    train_model_with_cv,
    evaluate_model,
    plot_roc_pr_curves,
)


class TestFeatureDataset:
    """Tests for FeatureDataset class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return np.random.randn(100, 50)
    
    @pytest.fixture
    def dummy_labels(self):
        """Create dummy labels."""
        return np.random.randint(0, 2, 100)
    
    def test_initialization(self, dummy_features, dummy_labels):
        """Test FeatureDataset initialization."""
        feature_names = [f"feature_{i}" for i in range(50)]
        dataset = FeatureDataset(dummy_features, dummy_labels, feature_names)
        
        assert len(dataset) == 100
        assert dataset.features.shape == (100, 50)
        assert dataset.labels.shape == (100,)
    
    def test_getitem(self, dummy_features, dummy_labels):
        """Test FeatureDataset __getitem__."""
        feature_names = [f"feature_{i}" for i in range(50)]
        dataset = FeatureDataset(dummy_features, dummy_labels, feature_names)
        
        features, label = dataset[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert features.shape == (50,)


class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor class."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array with some NaN values."""
        features = np.random.randn(100, 50)
        # Add some NaN values
        features[0, 0] = np.nan
        features[1, 1] = np.nan
        return features
    
    def test_initialization(self):
        """Test FeaturePreprocessor initialization."""
        preprocessor = FeaturePreprocessor(
            imputation_strategy="mean",
            scaling_method="standard",
            normalize=True
        )
        assert preprocessor.is_fitted is False
        assert preprocessor.imputer is not None
        assert preprocessor.scaler is not None
    
    def test_fit_transform(self, dummy_features):
        """Test FeaturePreprocessor fit_transform."""
        preprocessor = FeaturePreprocessor(
            imputation_strategy="mean",
            scaling_method="standard",
            normalize=True
        )
        
        transformed = preprocessor.fit_transform(dummy_features)
        
        assert transformed.shape == dummy_features.shape
        assert not np.any(np.isnan(transformed))
        assert preprocessor.is_fitted is True
    
    def test_transform(self, dummy_features):
        """Test FeaturePreprocessor transform after fit."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(dummy_features)
        
        test_features = np.random.randn(20, 50)
        transformed = preprocessor.transform(test_features)
        
        assert transformed.shape == test_features.shape


class TestCreateStratifiedSplits:
    """Tests for create_stratified_splits function."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        return np.random.randn(100, 50)
    
    @pytest.fixture
    def dummy_labels(self):
        """Create dummy labels."""
        return np.random.randint(0, 2, 100)
    
    def test_create_stratified_splits_basic(self, dummy_features, dummy_labels):
        """Test create_stratified_splits with basic data."""
        splits = create_stratified_splits(
            features=dummy_features,
            labels=dummy_labels,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42
        )
        
        assert len(splits) == 3
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        
        # Check shapes
        assert splits["train"]["features"].shape[0] + splits["val"]["features"].shape[0] + splits["test"]["features"].shape[0] == 100


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create dummy model."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    @pytest.fixture
    def dummy_loader(self):
        """Create dummy DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(20, 50)
        y = torch.randint(0, 2, (20,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=4)
    
    def test_evaluate_model_basic(self, dummy_model, dummy_loader):
        """Test evaluate_model with basic setup."""
        metrics = evaluate_model(dummy_model, dummy_loader, device="cpu")
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

