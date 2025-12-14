"""
Comprehensive unit tests for training/trainer module.
Tests all functions with dummy tensors to avoid GPU/video dependencies.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch
from lib.training.trainer import (
    OptimConfig,
    TrainConfig,
    EarlyStopping,
    freeze_all,
    unfreeze_all,
    freeze_backbone_unfreeze_head,
    trainable_params,
    compute_class_counts,
    make_class_weights,
    make_weighted_sampler,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    evaluate,
    fit,
)


class TestOptimConfig:
    """Tests for OptimConfig dataclass."""
    
    def test_default_values(self):
        """Test OptimConfig with default values."""
        config = OptimConfig()
        assert config.lr == 1e-4
        assert config.weight_decay == 1e-4
        assert config.betas == (0.9, 0.999)
        assert config.backbone_lr is None
        assert config.head_lr is None
        assert config.max_grad_norm == 1.0
    
    def test_custom_values(self):
        """Test OptimConfig with custom values."""
        config = OptimConfig(
            lr=1e-3,
            weight_decay=1e-5,
            betas=(0.95, 0.999),
            backbone_lr=1e-5,
            head_lr=1e-3,
            max_grad_norm=2.0
        )
        assert config.lr == 1e-3
        assert config.weight_decay == 1e-5
        assert config.betas == (0.95, 0.999)
        assert config.backbone_lr == 1e-5
        assert config.head_lr == 1e-3
        assert config.max_grad_norm == 2.0


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""
    
    def test_default_values(self):
        """Test TrainConfig with default values."""
        config = TrainConfig()
        assert config.num_epochs == 20
        assert config.log_interval == 10
        assert config.use_class_weights is True
        assert config.use_amp is True
        assert config.early_stopping_patience == 5
        assert config.scheduler_type == "cosine"
        assert config.warmup_epochs == 2
    
    def test_custom_values(self):
        """Test TrainConfig with custom values."""
        config = TrainConfig(
            num_epochs=50,
            log_interval=5,
            use_class_weights=False,
            use_amp=False,
            early_stopping_patience=10,
            scheduler_type="step",
            warmup_epochs=5
        )
        assert config.num_epochs == 50
        assert config.log_interval == 5
        assert config.use_class_weights is False
        assert config.use_amp is False
        assert config.early_stopping_patience == 10
        assert config.scheduler_type == "step"
        assert config.warmup_epochs == 5


class TestEarlyStopping:
    """Tests for EarlyStopping class."""
    
    def test_initialization(self):
        """Test EarlyStopping initialization."""
        es = EarlyStopping(patience=5, mode="max")
        assert es.patience == 5
        assert es.mode == "max"
        assert es.best is None
        assert es.counter == 0
        assert es.should_stop is False
    
    def test_step_first_value(self):
        """Test step() with first value."""
        es = EarlyStopping(patience=5, mode="max")
        es.step(0.8)
        assert es.best == 0.8
        assert es.counter == 0
        assert es.should_stop is False
    
    def test_step_improvement_max(self):
        """Test step() with improvement in max mode."""
        es = EarlyStopping(patience=5, mode="max")
        es.step(0.8)
        es.step(0.9)  # Improvement
        assert es.best == 0.9
        assert es.counter == 0
        assert es.should_stop is False
    
    def test_step_no_improvement_max(self):
        """Test step() with no improvement in max mode."""
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.8)
        es.step(0.7)  # No improvement
        assert es.best == 0.8
        assert es.counter == 1
        assert es.should_stop is False
        
        es.step(0.6)  # Still no improvement
        assert es.counter == 2
        es.step(0.5)  # Still no improvement - should stop
        assert es.counter == 3
        assert es.should_stop is True
    
    def test_step_improvement_min(self):
        """Test step() with improvement in min mode."""
        es = EarlyStopping(patience=5, mode="min")
        es.step(0.8)
        es.step(0.7)  # Improvement (lower is better)
        assert es.best == 0.7
        assert es.counter == 0
    
    def test_step_no_improvement_min(self):
        """Test step() with no improvement in min mode."""
        es = EarlyStopping(patience=2, mode="min")
        es.step(0.8)
        es.step(0.9)  # No improvement (higher is worse)
        assert es.best == 0.8
        assert es.counter == 1
        es.step(1.0)  # Still no improvement
        assert es.counter == 2
        assert es.should_stop is True


class TestFreezeUnfreeze:
    """Tests for freeze/unfreeze functions."""
    
    def test_freeze_all(self):
        """Test freeze_all function."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        freeze_all(model)
        for param in model.parameters():
            assert param.requires_grad is False
    
    def test_unfreeze_all(self):
        """Test unfreeze_all function."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        freeze_all(model)
        unfreeze_all(model)
        for param in model.parameters():
            assert param.requires_grad is True
    
    def test_freeze_backbone_unfreeze_head_with_backbone_head(self):
        """Test freeze_backbone_unfreeze_head with backbone and head attributes."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(10, 5))
                self.head = nn.Sequential(nn.Linear(5, 2))
        
        model = MockModel()
        freeze_backbone_unfreeze_head(model)
        
        for param in model.backbone.parameters():
            assert param.requires_grad is False
        for param in model.head.parameters():
            assert param.requires_grad is True
    
    def test_freeze_backbone_unfreeze_head_with_classifier(self):
        """Test freeze_backbone_unfreeze_head with classifier attribute."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(10, 5))
                self.classifier = nn.Sequential(nn.Linear(5, 2))
        
        model = MockModel()
        freeze_backbone_unfreeze_head(model)
        
        for param in model.backbone.parameters():
            assert param.requires_grad is False
        for param in model.classifier.parameters():
            assert param.requires_grad is True
    
    def test_freeze_backbone_unfreeze_head_with_fc(self):
        """Test freeze_backbone_unfreeze_head with fc attribute."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(10, 5))
                self.fc = nn.Sequential(nn.Linear(5, 2))
        
        model = MockModel()
        freeze_backbone_unfreeze_head(model)
        
        for param in model.backbone.parameters():
            assert param.requires_grad is False
        for param in model.fc.parameters():
            assert param.requires_grad is True


class TestTrainableParams:
    """Tests for trainable_params function."""
    
    def test_trainable_params_all_unfrozen(self):
        """Test trainable_params with all parameters unfrozen."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        trainable = list(trainable_params(model))
        assert len(trainable) > 0
        assert all(p.requires_grad for p in trainable)
    
    def test_trainable_params_all_frozen(self):
        """Test trainable_params with all parameters frozen."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        freeze_all(model)
        trainable = list(trainable_params(model))
        assert len(trainable) == 0
    
    def test_trainable_params_partial_frozen(self):
        """Test trainable_params with some parameters frozen."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        
        trainable = list(trainable_params(model))
        # Should have parameters from second layer only
        assert len(trainable) > 0
        assert all(p.requires_grad for p in trainable)


class TestComputeClassCounts:
    """Tests for compute_class_counts function."""
    
    def test_compute_class_counts_binary(self):
        """Test compute_class_counts with binary classification."""
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        dataset = TensorDataset(torch.randn(6, 10), labels)
        loader = DataLoader(dataset, batch_size=2)
        
        counts = compute_class_counts(loader, num_classes=2)
        assert counts.shape == (2,)
        assert counts[0] == 3  # Three class 0
        assert counts[1] == 3  # Three class 1
    
    def test_compute_class_counts_multiclass(self):
        """Test compute_class_counts with multiclass."""
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        dataset = TensorDataset(torch.randn(6, 10), labels)
        loader = DataLoader(dataset, batch_size=2)
        
        counts = compute_class_counts(loader, num_classes=3)
        assert counts.shape == (3,)
        assert counts[0] == 2
        assert counts[1] == 2
        assert counts[2] == 2
    
    def test_compute_class_counts_imbalanced(self):
        """Test compute_class_counts with imbalanced classes."""
        labels = torch.tensor([0, 0, 0, 0, 1])  # 4 class 0, 1 class 1
        dataset = TensorDataset(torch.randn(5, 10), labels)
        loader = DataLoader(dataset, batch_size=2)
        
        counts = compute_class_counts(loader, num_classes=2)
        assert counts[0] == 4
        assert counts[1] == 1


class TestMakeClassWeights:
    """Tests for make_class_weights function."""
    
    def test_make_class_weights_balanced(self):
        """Test make_class_weights with balanced classes."""
        counts = torch.tensor([5, 5])
        weights = make_class_weights(counts)
        assert weights.shape == (2,)
        # Should be approximately equal for balanced classes
        assert torch.allclose(weights, torch.ones(2), atol=0.1)
    
    def test_make_class_weights_imbalanced(self):
        """Test make_class_weights with imbalanced classes."""
        counts = torch.tensor([10, 2])  # Class 0 has 10, class 1 has 2
        weights = make_class_weights(counts)
        assert weights.shape == (2,)
        # Class 1 should have higher weight (inverse frequency)
        assert weights[1] > weights[0]
    
    def test_make_class_weights_zero_count(self):
        """Test make_class_weights with zero count (should not crash)."""
        counts = torch.tensor([10, 0])
        weights = make_class_weights(counts)
        assert weights.shape == (2,)
        # Should handle zero count gracefully


class TestMakeWeightedSampler:
    """Tests for make_weighted_sampler function."""
    
    def test_make_weighted_sampler(self):
        """Test make_weighted_sampler creation."""
        labels = torch.tensor([0, 0, 0, 1, 1])  # Imbalanced
        sampler = make_weighted_sampler(labels)
        assert sampler is not None
        assert sampler.num_samples == len(labels)
    
    def test_make_weighted_sampler_balanced(self):
        """Test make_weighted_sampler with balanced classes."""
        labels = torch.tensor([0, 0, 1, 1])
        sampler = make_weighted_sampler(labels)
        assert sampler.num_samples == 4


class TestBuildOptimizer:
    """Tests for build_optimizer function."""
    
    def test_build_optimizer_standard(self):
        """Test build_optimizer with standard config."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        config = OptimConfig(lr=1e-3, weight_decay=1e-4)
        optimizer = build_optimizer(model, config, use_differential_lr=False)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4
    
    def test_build_optimizer_differential_lr_with_backbone_fc(self):
        """Test build_optimizer with differential LR (backbone + fc)."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(10, 5))
                self.fc = nn.Sequential(nn.Linear(5, 2))
        
        model = MockModel()
        config = OptimConfig(
            lr=1e-4,
            backbone_lr=1e-5,
            head_lr=1e-3
        )
        optimizer = build_optimizer(model, config, use_differential_lr=True)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2
        # Check that backbone and head have different LRs
        lrs = [g['lr'] for g in optimizer.param_groups]
        assert len(set(lrs)) == 2  # Two different LRs
    
    def test_build_optimizer_differential_lr_with_backbone_head(self):
        """Test build_optimizer with differential LR (backbone + head)."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(10, 5))
                self.head = nn.Sequential(nn.Linear(5, 2))
        
        model = MockModel()
        config = OptimConfig(lr=1e-4, backbone_lr=1e-5)
        optimizer = build_optimizer(model, config, use_differential_lr=True)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2


class TestBuildScheduler:
    """Tests for build_scheduler function."""
    
    def test_build_scheduler_cosine(self):
        """Test build_scheduler with cosine scheduler."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, scheduler_type="cosine", num_epochs=10)
        
        assert scheduler is not None
        # Step scheduler
        scheduler.step()
        assert optimizer.param_groups[0]['lr'] < 1e-3  # Should decrease
    
    def test_build_scheduler_step(self):
        """Test build_scheduler with step scheduler."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, scheduler_type="step", num_epochs=10, step_size=2)
        
        assert scheduler is not None
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        scheduler.step()
        # After step_size steps, LR should decrease
        assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_build_scheduler_none(self):
        """Test build_scheduler with none scheduler."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, scheduler_type="none")
        
        assert scheduler is not None
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        # LR should remain constant
        assert optimizer.param_groups[0]['lr'] == initial_lr


class TestTrainOneEpoch:
    """Tests for train_one_epoch function."""
    
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
    
    def test_train_one_epoch_basic(self, dummy_model, dummy_loader):
        """Test train_one_epoch with basic setup."""
        optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
        device = "cpu"
        
        loss = train_one_epoch(
            dummy_model,
            dummy_loader,
            optimizer,
            device=device,
            use_class_weights=False,
            use_amp=False,
            epoch=1,
            log_interval=10
        )
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_train_one_epoch_with_class_weights(self, dummy_model, dummy_loader):
        """Test train_one_epoch with class weights."""
        optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
        device = "cpu"
        
        loss = train_one_epoch(
            dummy_model,
            dummy_loader,
            optimizer,
            device=device,
            use_class_weights=True,
            use_amp=False,
            epoch=1
        )
        
        assert isinstance(loss, float)
    
    def test_train_one_epoch_gradient_accumulation(self, dummy_model, dummy_loader):
        """Test train_one_epoch with gradient accumulation."""
        optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
        device = "cpu"
        
        loss = train_one_epoch(
            dummy_model,
            dummy_loader,
            optimizer,
            device=device,
            gradient_accumulation_steps=2,
            use_amp=False,
            epoch=1
        )
        
        assert isinstance(loss, float)


class TestEvaluate:
    """Tests for evaluate function."""
    
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
    
    def test_evaluate_basic(self, dummy_model, dummy_loader):
        """Test evaluate with basic setup."""
        device = "cpu"
        metrics = evaluate(dummy_model, dummy_loader, device=device)
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "per_class" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
    
    def test_evaluate_per_class_metrics(self, dummy_model, dummy_loader):
        """Test evaluate per-class metrics."""
        device = "cpu"
        metrics = evaluate(dummy_model, dummy_loader, device=device)
        
        per_class = metrics["per_class"]
        assert "0" in per_class
        assert "1" in per_class
        assert "precision" in per_class["0"]
        assert "recall" in per_class["0"]
        assert "f1" in per_class["0"]


class TestFit:
    """Tests for fit function."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def dummy_train_loader(self):
        """Create a dummy training DataLoader."""
        X = torch.randn(40, 10)
        y = torch.randint(0, 2, (40,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=4)
    
    @pytest.fixture
    def dummy_val_loader(self):
        """Create a dummy validation DataLoader."""
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=4)
    
    def test_fit_basic(self, dummy_model, dummy_train_loader, dummy_val_loader):
        """Test fit with basic setup."""
        optim_cfg = OptimConfig(lr=1e-3)
        train_cfg = TrainConfig(
            num_epochs=2,
            device="cpu",
            use_amp=False,
            early_stopping_patience=0  # Disable early stopping for short test
        )
        
        trained_model = fit(
            dummy_model,
            dummy_train_loader,
            dummy_val_loader,
            optim_cfg,
            train_cfg,
            use_differential_lr=False
        )
        
        assert trained_model is not None
        assert trained_model is dummy_model  # Should return same model
    
    def test_fit_with_early_stopping(self, dummy_model, dummy_train_loader, dummy_val_loader):
        """Test fit with early stopping."""
        optim_cfg = OptimConfig(lr=1e-3)
        train_cfg = TrainConfig(
            num_epochs=10,
            device="cpu",
            use_amp=False,
            early_stopping_patience=2
        )
        
        trained_model = fit(
            dummy_model,
            dummy_train_loader,
            dummy_val_loader,
            optim_cfg,
            train_cfg
        )
        
        assert trained_model is not None
    
    def test_fit_without_validation(self, dummy_model, dummy_train_loader):
        """Test fit without validation loader."""
        optim_cfg = OptimConfig(lr=1e-3)
        train_cfg = TrainConfig(
            num_epochs=2,
            device="cpu",
            use_amp=False
        )
        
        trained_model = fit(
            dummy_model,
            dummy_train_loader,
            None,  # No validation
            optim_cfg,
            train_cfg
        )
        
        assert trained_model is not None
    
    def test_fit_with_scheduler(self, dummy_model, dummy_train_loader, dummy_val_loader):
        """Test fit with learning rate scheduler."""
        optim_cfg = OptimConfig(lr=1e-3)
        train_cfg = TrainConfig(
            num_epochs=3,
            device="cpu",
            use_amp=False,
            scheduler_type="cosine",
            warmup_epochs=1
        )
        
        trained_model = fit(
            dummy_model,
            dummy_train_loader,
            dummy_val_loader,
            optim_cfg,
            train_cfg
        )
        
        assert trained_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

