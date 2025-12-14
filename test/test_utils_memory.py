"""
Comprehensive unit tests for utils/memory module.
Tests memory management functions.
"""
import pytest
import torch
from lib.utils.memory import (
    aggressive_gc,
    get_memory_stats,
    log_memory_stats,
    check_oom_error,
    handle_oom_error,
    safe_execute,
)


class TestAggressiveGc:
    """Tests for aggressive_gc function."""
    
    def test_aggressive_gc_basic(self):
        """Test aggressive_gc doesn't raise errors."""
        aggressive_gc(clear_cuda=False)
        # Should not raise
    
    def test_aggressive_gc_with_cuda(self):
        """Test aggressive_gc with CUDA clearing."""
        if torch.cuda.is_available():
            aggressive_gc(clear_cuda=True)
        else:
            aggressive_gc(clear_cuda=False)
        # Should not raise


class TestGetMemoryStats:
    """Tests for get_memory_stats function."""
    
    def test_get_memory_stats_basic(self):
        """Test get_memory_stats returns valid stats."""
        stats = get_memory_stats()
        
        assert isinstance(stats, dict)
        assert "cpu_memory_mb" in stats
        assert "cpu_memory_gb" in stats
        assert "gpu_allocated_gb" in stats
        assert stats["cpu_memory_mb"] > 0
    
    def test_get_memory_stats_gpu_available(self):
        """Test get_memory_stats includes GPU stats if available."""
        stats = get_memory_stats()
        
        if torch.cuda.is_available():
            assert stats["gpu_total_gb"] > 0
        else:
            assert stats["gpu_total_gb"] == 0.0


class TestLogMemoryStats:
    """Tests for log_memory_stats function."""
    
    def test_log_memory_stats_basic(self):
        """Test log_memory_stats doesn't raise errors."""
        log_memory_stats()
        # Should not raise
    
    def test_log_memory_stats_with_context(self):
        """Test log_memory_stats with context."""
        log_memory_stats(context="test_context")
        # Should not raise
    
    def test_log_memory_stats_detailed(self):
        """Test log_memory_stats with detailed=True."""
        log_memory_stats(detailed=True)
        # Should not raise


class TestCheckOomError:
    """Tests for check_oom_error function."""
    
    def test_check_oom_error_true(self):
        """Test check_oom_error returns True for OOM errors."""
        error = RuntimeError("CUDA out of memory")
        assert check_oom_error(error) is True
        
        error = RuntimeError("out of memory")
        assert check_oom_error(error) is True
    
    def test_check_oom_error_false(self):
        """Test check_oom_error returns False for non-OOM errors."""
        error = ValueError("Invalid input")
        assert check_oom_error(error) is False


class TestHandleOomError:
    """Tests for handle_oom_error function."""
    
    def test_handle_oom_error_basic(self):
        """Test handle_oom_error doesn't raise errors."""
        error = RuntimeError("CUDA out of memory")
        handle_oom_error(error, context="test")
        # Should not raise


class TestSafeExecute:
    """Tests for safe_execute function."""
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        def func():
            return 42
        
        result = safe_execute(func, context="test")
        assert result == 42
    
    def test_safe_execute_oom_error(self):
        """Test safe_execute handles OOM errors."""
        def func():
            raise RuntimeError("CUDA out of memory")
        
        result = safe_execute(func, context="test")
        assert result is None
    
    def test_safe_execute_other_error(self):
        """Test safe_execute re-raises non-OOM errors."""
        def func():
            raise ValueError("Invalid input")
        
        with pytest.raises(ValueError):
            safe_execute(func, context="test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
