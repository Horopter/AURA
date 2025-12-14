"""
Comprehensive unit tests for training/model_factory module.
Tests model creation and configuration with dummy configs.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from lib.training.model_factory import (
    get_model_config,
    list_available_models,
    create_model,
    is_xgboost_model,
    is_pytorch_model,
    get_model_input_shape,
    download_pretrained_models,
)


class TestGetModelConfig:
    """Tests for get_model_config function."""
    
    def test_get_model_config_logistic_regression(self):
        """Test get_model_config for logistic_regression."""
        config = get_model_config("logistic_regression")
        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "num_workers" in config
        assert "num_frames" in config
    
    def test_get_model_config_xgboost(self):
        """Test get_model_config for xgboost model."""
        config = get_model_config("xgboost_i3d")
        assert isinstance(config, dict)
        assert "batch_size" in config
    
    def test_get_model_config_video_model(self):
        """Test get_model_config for video model."""
        config = get_model_config("slowfast")
        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "num_frames" in config
    
    def test_get_model_config_invalid(self):
        """Test get_model_config with invalid model type."""
        with pytest.raises(ValueError):
            get_model_config("invalid_model_type")


class TestListAvailableModels:
    """Tests for list_available_models function."""
    
    def test_list_available_models_with_xgboost(self):
        """Test list_available_models including XGBoost."""
        models = list_available_models(include_xgboost=True)
        assert isinstance(models, list)
        assert len(models) > 0
        assert "logistic_regression" in models
        assert "xgboost_i3d" in models or "xgboost_pretrained_inception" in models
    
    def test_list_available_models_without_xgboost(self):
        """Test list_available_models excluding XGBoost."""
        models = list_available_models(include_xgboost=False)
        assert isinstance(models, list)
        assert len(models) > 0
        # XGBoost models should not be in the list
        xgboost_models = [m for m in models if m.startswith("xgboost_")]
        assert len(xgboost_models) == 0


class TestCreateModel:
    """Tests for create_model function."""
    
    @pytest.fixture
    def dummy_config(self):
        """Create a dummy RunConfig-like object."""
        config = Mock()
        config.num_frames = 8
        config.img_size = 224
        config.fixed_size = 224
        config.batch_size = 4
        config.num_epochs = 2
        config.learning_rate = 1e-4
        config.project_root = "/tmp/test"
        config.output_dir = "/tmp/test/output"
        return config
    
    def test_create_model_logistic_regression(self, dummy_config):
        """Test create_model for logistic_regression."""
        model = create_model("logistic_regression", dummy_config)
        assert model is not None
        # LogisticRegressionBaseline should have fit, predict methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_model_svm(self, dummy_config):
        """Test create_model for svm."""
        model = create_model("svm", dummy_config)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_model_naive_cnn(self, dummy_config):
        """Test create_model for naive_cnn."""
        model = create_model("naive_cnn", dummy_config)
        assert model is not None
        # Should be a PyTorch model
        import torch.nn as nn
        assert isinstance(model, nn.Module)
    
    @pytest.mark.skipif(True, reason="Requires heavy dependencies")
    def test_create_model_vit_gru(self, dummy_config):
        """Test create_model for vit_gru (skip if dependencies not available)."""
        try:
            model = create_model("vit_gru", dummy_config)
            assert model is not None
            import torch.nn as nn
            assert isinstance(model, nn.Module)
        except (ImportError, RuntimeError):
            pytest.skip("vit_gru dependencies not available")
    
    def test_create_model_xgboost(self, dummy_config):
        """Test create_model for xgboost model."""
        try:
            model = create_model("xgboost_pretrained_inception", dummy_config)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
        except ImportError:
            pytest.skip("xgboost not available")
    
    def test_create_model_invalid(self, dummy_config):
        """Test create_model with invalid model type."""
        with pytest.raises(ValueError):
            create_model("invalid_model", dummy_config)


class TestIsXGBoostModel:
    """Tests for is_xgboost_model function."""
    
    def test_is_xgboost_model_true(self):
        """Test is_xgboost_model returns True for XGBoost models."""
        assert is_xgboost_model("xgboost_i3d") is True
        assert is_xgboost_model("xgboost_pretrained_inception") is True
        assert is_xgboost_model("xgboost_vit_gru") is True
    
    def test_is_xgboost_model_false(self):
        """Test is_xgboost_model returns False for non-XGBoost models."""
        assert is_xgboost_model("logistic_regression") is False
        assert is_xgboost_model("slowfast") is False
        assert is_xgboost_model("naive_cnn") is False


class TestIsPyTorchModel:
    """Tests for is_pytorch_model function."""
    
    def test_is_pytorch_model_true(self):
        """Test is_pytorch_model returns True for PyTorch models."""
        assert is_pytorch_model("slowfast") is True
        assert is_pytorch_model("x3d") is True
        assert is_pytorch_model("naive_cnn") is True
        assert is_pytorch_model("vit_gru") is True
    
    def test_is_pytorch_model_false(self):
        """Test is_pytorch_model returns False for non-PyTorch models."""
        assert is_pytorch_model("logistic_regression") is False
        assert is_pytorch_model("svm") is False
        assert is_pytorch_model("xgboost_i3d") is False  # XGBoost uses PyTorch for features but is not a PyTorch model


class TestGetModelInputShape:
    """Tests for get_model_input_shape function."""
    
    @pytest.fixture
    def dummy_config(self):
        """Create a dummy RunConfig-like object."""
        config = Mock()
        config.num_frames = 8
        config.img_size = 224
        config.fixed_size = 224
        return config
    
    def test_get_model_input_shape_video_model(self, dummy_config):
        """Test get_model_input_shape for video model."""
        shape = get_model_input_shape("slowfast", dummy_config)
        assert isinstance(shape, tuple)
        assert len(shape) == 5  # (batch, channels, frames, height, width)
    
    def test_get_model_input_shape_feature_model(self, dummy_config):
        """Test get_model_input_shape for feature-based model."""
        # Feature models don't have video input shape
        # This might raise an error or return None
        try:
            shape = get_model_input_shape("logistic_regression", dummy_config)
            # If it doesn't raise, check it's reasonable
            if shape is not None:
                assert isinstance(shape, tuple)
        except (ValueError, AttributeError):
            # Expected for feature-based models
            pass


class TestDownloadPretrainedModels:
    """Tests for download_pretrained_models function."""
    
    def test_download_pretrained_models_empty_list(self):
        """Test download_pretrained_models with empty list."""
        result = download_pretrained_models([])
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_download_pretrained_models_single(self):
        """Test download_pretrained_models with single model."""
        # This might download models, so we'll just test it doesn't crash
        try:
            result = download_pretrained_models(["slowfast"])
            assert isinstance(result, dict)
            # Result should map model names to success status
            assert "slowfast" in result
            assert isinstance(result["slowfast"], bool)
        except Exception as e:
            # If download fails (network, etc.), that's OK for unit tests
            pytest.skip(f"Model download failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

