"""
Comprehensive unit tests for training/metrics_utils module.
Tests classification metrics computation with dummy data.
"""
import pytest
import numpy as np
from lib.training.metrics_utils import compute_classification_metrics


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics function."""
    
    def test_compute_classification_metrics_basic(self):
        """Test compute_classification_metrics with basic predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.6])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "val_acc" in metrics
        assert "val_f1" in metrics
        assert "val_precision" in metrics
        assert "val_recall" in metrics
        assert "val_f1_class0" in metrics
        assert "val_precision_class0" in metrics
        assert "val_recall_class0" in metrics
        assert "val_f1_class1" in metrics
        assert "val_precision_class1" in metrics
        assert "val_recall_class1" in metrics
        
        assert 0 <= metrics["val_acc"] <= 1
        assert 0 <= metrics["val_f1"] <= 1
        assert 0 <= metrics["val_precision"] <= 1
        assert 0 <= metrics["val_recall"] <= 1
    
    def test_compute_classification_metrics_perfect_predictions(self):
        """Test compute_classification_metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])  # Perfect
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.1])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert metrics["val_acc"] == 1.0
        assert metrics["val_f1"] == 1.0
        assert metrics["val_precision"] == 1.0
        assert metrics["val_recall"] == 1.0
    
    def test_compute_classification_metrics_without_probs(self):
        """Test compute_classification_metrics without probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs=None)
        
        assert np.isnan(metrics["val_loss"])  # Should be NaN when no probs
        assert metrics["val_acc"] == 1.0
    
    def test_compute_classification_metrics_with_2d_probs(self):
        """Test compute_classification_metrics with 2D probability array."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_probs = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9]
        ])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert not np.isnan(metrics["val_loss"])
        assert metrics["val_acc"] == 1.0
    
    def test_compute_classification_metrics_imbalanced(self):
        """Test compute_classification_metrics with imbalanced classes."""
        y_true = np.array([0, 0, 0, 0, 1])  # 4 class 0, 1 class 1
        y_pred = np.array([0, 0, 0, 0, 1])
        y_probs = np.array([0.1, 0.2, 0.1, 0.2, 0.9])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert metrics["val_acc"] == 1.0
        assert "val_f1_class0" in metrics
        assert "val_f1_class1" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

