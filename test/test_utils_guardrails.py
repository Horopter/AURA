"""
Comprehensive unit tests for utils/guardrails module.
Tests guardrail functions with dummy data.
"""
import pytest
import time
from lib.utils.guardrails import (
    GuardrailError,
    ResourceExhaustedError,
    TimeoutError as GuardrailTimeoutError,
    DataIntegrityError,
    HealthCheckStatus,
    ResourceLimits,
    RetryConfig,
    HealthCheckResult,
    ResourceMonitor,
    TimeoutHandler,
    retry_with_backoff,
    validate_file_integrity,
    validate_directory,
    resource_guard,
    guarded_execution,
    guarded_decorator,
)


class TestGuardrailErrors:
    """Tests for guardrail exception classes."""
    
    def test_guardrail_error(self):
        """Test GuardrailError can be raised."""
        with pytest.raises(GuardrailError):
            raise GuardrailError("Test error")
    
    def test_resource_exhausted_error(self):
        """Test ResourceExhaustedError."""
        with pytest.raises(ResourceExhaustedError):
            raise ResourceExhaustedError("Resource exhausted")
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        with pytest.raises(GuardrailTimeoutError):
            raise GuardrailTimeoutError("Operation timed out")
    
    def test_data_integrity_error(self):
        """Test DataIntegrityError."""
        with pytest.raises(DataIntegrityError):
            raise DataIntegrityError("Data integrity check failed")


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""
    
    def test_default_values(self):
        """Test ResourceLimits with default values."""
        limits = ResourceLimits()
        assert limits.max_memory_gb == 200.0
        assert limits.max_disk_gb == 1000.0
        assert limits.max_cpu_percent == 95.0
    
    def test_custom_values(self):
        """Test ResourceLimits with custom values."""
        limits = ResourceLimits(
            max_memory_gb=100.0,
            max_disk_gb=500.0,
            max_cpu_percent=80.0
        )
        assert limits.max_memory_gb == 100.0
        assert limits.max_disk_gb == 500.0
        assert limits.max_cpu_percent == 80.0


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""
    
    def test_default_values(self):
        """Test RetryConfig with default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
    
    def test_custom_values(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0
        )
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0


class TestResourceMonitor:
    """Tests for ResourceMonitor class."""
    
    def test_initialization(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor()
        assert monitor.limits is not None
        assert monitor.process is not None
    
    def test_check_memory(self):
        """Test ResourceMonitor.check_memory."""
        monitor = ResourceMonitor()
        is_ok, metrics = monitor.check_memory()
        
        assert isinstance(is_ok, bool)
        assert isinstance(metrics, dict)
        assert "process_memory_gb" in metrics
    
    def test_check_disk(self, temp_dir):
        """Test ResourceMonitor.check_disk."""
        monitor = ResourceMonitor()
        is_ok, metrics = monitor.check_disk(temp_dir)
        
        assert isinstance(is_ok, bool)
        assert isinstance(metrics, dict)


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""
    
    def test_retry_with_backoff_success(self):
        """Test retry_with_backoff with successful function."""
        from lib.utils.guardrails import RetryConfig
        call_count = [0]
        
        def func():
            call_count[0] += 1
            return 42
        
        config = RetryConfig(max_retries=3)
        result = retry_with_backoff(func, config=config)
        assert result == 42
        assert call_count[0] == 1
    
    def test_retry_with_backoff_retries(self):
        """Test retry_with_backoff retries on failure."""
        from lib.utils.guardrails import RetryConfig
        call_count = [0]
        
        def func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise OSError("Temporary error")
            return 42
        
        config = RetryConfig(max_retries=3)
        result = retry_with_backoff(func, config=config)
        assert result == 42
        assert call_count[0] == 2


class TestValidateFileIntegrity:
    """Tests for validate_file_integrity function."""
    
    def test_validate_file_integrity_exists(self, temp_dir):
        """Test validate_file_integrity with existing file."""
        from pathlib import Path
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        is_valid, reason = validate_file_integrity(str(test_file))
        assert is_valid is True
    
    def test_validate_file_integrity_not_exists(self, temp_dir):
        """Test validate_file_integrity with non-existing file."""
        from pathlib import Path
        test_file = Path(temp_dir) / "nonexistent.txt"
        
        is_valid, reason = validate_file_integrity(str(test_file))
        assert is_valid is False


class TestValidateDirectory:
    """Tests for validate_directory function."""
    
    def test_validate_directory_exists(self, temp_dir):
        """Test validate_directory with existing directory."""
        is_valid, reason = validate_directory(temp_dir)
        assert is_valid is True
    
    def test_validate_directory_not_exists(self, temp_dir):
        """Test validate_directory with non-existing directory."""
        from pathlib import Path
        nonexistent = Path(temp_dir) / "nonexistent"
        is_valid, reason = validate_directory(str(nonexistent))
        assert is_valid is False


class TestGuardedExecution:
    """Tests for guarded_execution function."""
    
    def test_guarded_execution_success(self):
        """Test guarded_execution with successful function."""
        def func():
            return 42
        
        result = guarded_execution(func)
        assert result == 42
    
    def test_guarded_execution_with_timeout(self):
        """Test guarded_execution with timeout."""
        def slow_func():
            time.sleep(2)
            return 42
        
        # Should timeout if timeout is set to 0.1 seconds
        result = guarded_execution(slow_func, timeout_seconds=0.1)
        # May return None or raise timeout error depending on implementation
        assert result is None or isinstance(result, GuardrailTimeoutError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
