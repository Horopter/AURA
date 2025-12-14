"""
Comprehensive unit tests for training/ensemble module.
Tests ensemble model creation and prediction with dummy data.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from lib.training.ensemble import (
    EnsembleMetaLearner,
    PredictionDataset,
    load_trained_model,
    get_predictions_from_model,
    train_ensemble_model,
)


class TestEnsembleMetaLearner:
    """Tests for EnsembleMetaLearner class."""
    
    def test_initialization(self):
        """Test EnsembleMetaLearner initialization."""
        model = EnsembleMetaLearner(num_models=3, hidden_dim=64)
        assert isinstance(model, nn.Module)
        assert model.fc1.in_features == 6  # 3 models * 2 classes
    
    def test_forward_pass(self):
        """Test EnsembleMetaLearner forward pass."""
        model = EnsembleMetaLearner(num_models=2, hidden_dim=32)
        model.eval()
        
        # Input: (batch, num_models * 2) = (4, 4)
        x = torch.randn(4, 4)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 1)  # Binary classification output
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = EnsembleMetaLearner(num_models=3, hidden_dim=32)
        model.eval()
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 6)  # 3 models * 2
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 1)


class TestPredictionDataset:
    """Tests for PredictionDataset class."""
    
    def test_initialization(self):
        """Test PredictionDataset initialization."""
        predictions = {
            "model1": np.random.rand(10, 2),
            "model2": np.random.rand(10, 2)
        }
        labels = np.random.randint(0, 2, 10)
        
        dataset = PredictionDataset(predictions, labels)
        
        assert len(dataset) == 10
        assert dataset.predictions.shape == (10, 4)  # 2 models * 2 classes
    
    def test_getitem(self):
        """Test PredictionDataset __getitem__."""
        predictions = {
            "model1": np.random.rand(10, 2),
            "model2": np.random.rand(10, 2)
        }
        labels = np.random.randint(0, 2, 10)
        
        dataset = PredictionDataset(predictions, labels)
        
        pred, label = dataset[0]
        assert pred.shape == (4,)  # 2 models * 2 classes
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long


class TestLoadTrainedModel:
    """Tests for load_trained_model function."""
    
    @pytest.fixture
    def dummy_model_config(self):
        """Create dummy model config."""
        config = Mock()
        config.num_frames = 8
        config.img_size = 224
        config.fixed_size = 224
        return config
    
    @pytest.fixture
    def dummy_fold_dir(self, temp_dir):
        """Create dummy fold directory with checkpoint."""
        fold_dir = Path(temp_dir) / "fold_0"
        fold_dir.mkdir()
        return fold_dir
    
    @patch('lib.training.ensemble.create_model')
    @patch('lib.training.ensemble.is_pytorch_model')
    def test_load_trained_model_pytorch(
        self,
        mock_is_pytorch,
        mock_create_model,
        dummy_model_config,
        dummy_fold_dir,
        temp_dir
    ):
        """Test load_trained_model for PyTorch model."""
        mock_is_pytorch.return_value = True
        
        # Create dummy model and checkpoint
        model = nn.Sequential(nn.Linear(10, 2))
        checkpoint_path = dummy_fold_dir / "checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
        mock_create_model.return_value = model
        
        loaded = load_trained_model(
            "naive_cnn",
            dummy_fold_dir,
            temp_dir,
            dummy_model_config
        )
        
        assert loaded is not None
        assert isinstance(loaded, nn.Module)
    
    @patch('lib.training.ensemble.is_pytorch_model')
    def test_load_trained_model_sklearn(
        self,
        mock_is_pytorch,
        dummy_fold_dir,
        temp_dir
    ):
        """Test load_trained_model for sklearn model."""
        mock_is_pytorch.return_value = False
        
        # Create dummy sklearn model files
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        model = LogisticRegression()
        model.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        joblib.dump(model, dummy_fold_dir / "model.joblib")
        joblib.dump(Mock(), dummy_fold_dir / "scaler.joblib")
        
        # Create metadata
        import json
        with open(dummy_fold_dir / "metadata.json", 'w') as f:
            json.dump({"feature_indices": [0, 1, 2]}, f)
        
        loaded = load_trained_model(
            "logistic_regression",
            dummy_fold_dir,
            temp_dir,
            Mock()
        )
        
        assert loaded is not None


class TestGetPredictionsFromModel:
    """Tests for get_predictions_from_model function."""
    
    @patch('lib.training.ensemble.is_pytorch_model')
    @patch('lib.training.ensemble.VideoDataset')
    def test_get_predictions_from_model_pytorch(
        self,
        mock_video_dataset,
        mock_is_pytorch,
        temp_dir
    ):
        """Test get_predictions_from_model for PyTorch model."""
        mock_is_pytorch.return_value = True
        
        # Create dummy model
        model = nn.Sequential(nn.Linear(10, 2))
        model.eval()
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([
            (torch.randn(2, 10), torch.randint(0, 2, (2,)))
            for _ in range(5)
        ]))
        mock_video_dataset.return_value = mock_dataset
        
        # Mock DataLoader
        with patch('lib.training.ensemble.DataLoader', return_value=mock_loader):
            predictions = get_predictions_from_model(
                model,
                "naive_cnn",
                mock_dataset,
                "cpu",
                temp_dir
            )
        
        assert predictions.shape[0] == 10
        assert predictions.shape[1] == 2  # Binary classification


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

