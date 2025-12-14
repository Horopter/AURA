"""
Comprehensive unit tests for training/cleanup_utils module.
Tests memory cleanup functions.
"""
import pytest
import torch
import torch.nn as nn
from lib.training.cleanup_utils import (
    cleanup_model_and_memory,
    cleanup_resources,
)


class TestCleanupModelAndMemory:
    """Tests for cleanup_model_and_memory function."""
    
    def test_cleanup_model_and_memory_with_model(self):
        """Test cleanup_model_and_memory with model."""
        model = nn.Sequential(nn.Linear(10, 2))
        device = torch.device("cpu")
        
        # Should not raise errors
        cleanup_model_and_memory(model=model, device=device, clear_cuda=False)
    
    def test_cleanup_model_and_memory_no_model(self):
        """Test cleanup_model_and_memory without model."""
        cleanup_model_and_memory(model=None, clear_cuda=False)
        # Should not raise errors
    
    def test_cleanup_model_and_memory_cuda(self):
        """Test cleanup_model_and_memory with CUDA (if available)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            cleanup_model_and_memory(model=None, device=device, clear_cuda=True)
        else:
            pytest.skip("CUDA not available")


class TestCleanupResources:
    """Tests for cleanup_resources function."""
    
    def test_cleanup_resources_basic(self):
        """Test cleanup_resources with basic setup."""
        model = nn.Sequential(nn.Linear(10, 2))
        device = torch.device("cpu")
        
        cleanup_resources(
            model=model,
            device=device,
            clear_cuda=False
        )
        # Should not raise errors
    
    def test_cleanup_resources_with_mlflow_tracker(self):
        """Test cleanup_resources with MLflow tracker."""
        model = nn.Sequential(nn.Linear(10, 2))
        mock_tracker = Mock()
        mock_tracker.end_run = Mock()
        
        cleanup_resources(
            model=model,
            mlflow_tracker=mock_tracker,
            clear_cuda=False
        )
        
        mock_tracker.end_run.assert_called_once()
    
    def test_cleanup_resources_with_aggressive_gc(self):
        """Test cleanup_resources with aggressive GC function."""
        model = nn.Sequential(nn.Linear(10, 2))
        mock_gc = Mock()
        
        cleanup_resources(
            model=model,
            aggressive_gc_func=mock_gc,
            clear_cuda=False
        )
        
        mock_gc.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

