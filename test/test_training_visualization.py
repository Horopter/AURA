"""
Comprehensive unit tests for training/visualization module.
Tests plotting functions with dummy data.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from lib.training.visualization import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_cv_fold_comparison,
    plot_hyperparameter_search,
    generate_all_plots,
)


class TestPlotLearningCurves:
    """Tests for plot_learning_curves function."""
    
    @pytest.fixture
    def dummy_losses(self):
        """Create dummy loss values."""
        return {
            "train_losses": [1.0, 0.8, 0.6, 0.4, 0.3],
            "val_losses": [1.2, 0.9, 0.7, 0.5, 0.4],
            "train_accs": [0.5, 0.6, 0.7, 0.8, 0.9],
            "val_accs": [0.4, 0.5, 0.6, 0.7, 0.8]
        }
    
    def test_plot_learning_curves_loss_only(self, dummy_losses, temp_dir):
        """Test plot_learning_curves with loss only."""
        save_path = Path(temp_dir) / "learning_curves.png"
        
        plot_learning_curves(
            train_losses=dummy_losses["train_losses"],
            val_losses=dummy_losses["val_losses"],
            save_path=save_path
        )
        
        assert save_path.exists()
    
    def test_plot_learning_curves_with_accuracy(self, dummy_losses, temp_dir):
        """Test plot_learning_curves with loss and accuracy."""
        save_path = Path(temp_dir) / "learning_curves_full.png"
        
        plot_learning_curves(
            train_losses=dummy_losses["train_losses"],
            val_losses=dummy_losses["val_losses"],
            train_accs=dummy_losses["train_accs"],
            val_accs=dummy_losses["val_accs"],
            save_path=save_path
        )
        
        assert save_path.exists()


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""
    
    def test_plot_confusion_matrix_basic(self, temp_dir):
        """Test plot_confusion_matrix with basic data."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1])
        class_names = ["Class 0", "Class 1"]
        save_path = Path(temp_dir) / "confusion_matrix.png"
        
        plot_confusion_matrix(y_true, y_pred, class_names, save_path)
        
        assert save_path.exists()


class TestPlotRocCurve:
    """Tests for plot_roc_curve function."""
    
    def test_plot_roc_curve_basic(self, temp_dir):
        """Test plot_roc_curve with basic data."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        save_path = Path(temp_dir) / "roc_curve.png"
        
        plot_roc_curve(y_true, y_probs, save_path)
        
        assert save_path.exists()


class TestPlotPrecisionRecallCurve:
    """Tests for plot_precision_recall_curve function."""
    
    def test_plot_precision_recall_curve_basic(self, temp_dir):
        """Test plot_precision_recall_curve with basic data."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        save_path = Path(temp_dir) / "pr_curve.png"
        
        plot_precision_recall_curve(y_true, y_probs, save_path)
        
        assert save_path.exists()


class TestPlotCvFoldComparison:
    """Tests for plot_cv_fold_comparison function."""
    
    def test_plot_cv_fold_comparison_basic(self, temp_dir):
        """Test plot_cv_fold_comparison with basic data."""
        fold_metrics = {
            "fold_0": {"f1": 0.8, "accuracy": 0.85},
            "fold_1": {"f1": 0.82, "accuracy": 0.87},
            "fold_2": {"f1": 0.79, "accuracy": 0.84}
        }
        save_path = Path(temp_dir) / "cv_comparison.png"
        
        plot_cv_fold_comparison(fold_metrics, save_path)
        
        assert save_path.exists()


class TestPlotHyperparameterSearch:
    """Tests for plot_hyperparameter_search function."""
    
    def test_plot_hyperparameter_search_basic(self, temp_dir):
        """Test plot_hyperparameter_search with basic data."""
        results = [
            {"params": {"C": 0.1}, "score": 0.8},
            {"params": {"C": 1.0}, "score": 0.9},
            {"params": {"C": 10.0}, "score": 0.85}
        ]
        save_path = Path(temp_dir) / "hyperparameter_search.png"
        
        plot_hyperparameter_search(results, save_path)
        
        assert save_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

