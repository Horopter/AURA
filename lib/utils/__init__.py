"""
Utility functions module.

Provides:
- Memory management (aggressive GC, memory profiling)
- OOM handling and safe execution
- Video path resolution
- Video data loading and splitting
- Video metrics
"""

from .mlops_utils import (
    aggressive_gc,
    log_memory_stats,
    get_memory_stats,
    check_oom_error,
    handle_oom_error,
    safe_execute,
)

__all__ = [
    "aggressive_gc",
    "log_memory_stats",
    "get_memory_stats",
    "check_oom_error",
    "handle_oom_error",
    "safe_execute",
]

