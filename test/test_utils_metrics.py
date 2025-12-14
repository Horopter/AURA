"""
Comprehensive unit tests for utils/metrics module.
Tests metrics computation functions with dummy data.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lib.utils.metrics import (
    collect_logits_and_labels,
    basic_classification_metrics,
    confusion_matrix,
    roc_auc,
)


class TestCollectLogitsAndLabels:
    """Tests for collect_logits_and_labels function."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def dummy_loader(self):
        """Create a dummy DataLoader."""
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=4)
    
    def test_collect_logits_and_labels_basic(self, dummy_model, dummy_loader):
        """Test collect_logits_and_labels with basic setup."""
        logits, labels = collect_logits_and_labels(
            dummy_model,
            dummy_loader,
            device="cpu"
        )
        
        assert logits.shape[0] == 20
        assert labels.shape[0] == 20
        assert logits.shape[1] == 2  # Binary classification (2 logits)


class TestBasicClassificationMetrics:
    """Tests for basic_classification_metrics function."""
    
    def test_basic_classification_metrics_binary(self):
        """Test basic_classification_metrics with binary logits."""
        logits = torch.tensor([-1.0, 1.0, -0.5, 0.5])
        labels = torch.tensor([0, 1, 0, 1])
        
        metrics = basic_classification_metrics(logits, labels)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_basic_classification_metrics_multiclass(self):
        """Test basic_classification_metrics with multiclass logits."""
        logits = torch.tensor([
            [2.0, 0.5],
            [0.5, 2.0],
            [2.0, 0.5],
            [0.5, 2.0]
        ])
        labels = torch.tensor([0, 1, 0, 1])
        
        metrics = basic_classification_metrics(logits, labels)
        
        assert metrics["accuracy"] == 1.0  # Perfect predictions


class TestConfusionMatrix:
    """Tests for confusion_matrix function."""
    
    def test_confusion_matrix_binary(self):
        """Test confusion_matrix with binary classification."""
        logits = torch.tensor([-1.0, 1.0, -0.5, 0.5])
        labels = torch.tensor([0, 1, 0, 1])
        
        cm = confusion_matrix(logits, labels)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == 4  # Total samples


class TestRocAuc:
    """Tests for roc_auc function."""
    
    def test_roc_auc_basic(self):
        """Test roc_auc with basic predictions."""
        logits = torch.tensor([-1.0, 1.0, -0.5, 0.5])
        labels = torch.tensor([0, 1, 0, 1])
        
        auc = roc_auc(logits, labels)
        
        assert 0 <= auc <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
