"""
MLOps infrastructure module.

Provides:
- Experiment tracking and configuration
- Checkpoint management
- Data versioning
- Pipeline orchestration (single-split, k-fold, multi-model)
- Cleanup utilities
"""

from .mlops_core import (
    RunConfig,
    ExperimentTracker,
    CheckpointManager,
    DataVersionManager,
    create_run_directory,
)
from .mlops_pipeline import (
    PipelineStage,
    MLOpsPipeline,
    build_mlops_pipeline,
    fit_with_tracking,
)
from .mlops_pipeline_kfold import build_kfold_pipeline
from .mlops_pipeline_multimodel import build_multimodel_pipeline
from .cleanup_utils import (
    cleanup_runs_and_logs,
    cleanup_intermediate_files,
)

__all__ = [
    "RunConfig",
    "ExperimentTracker",
    "CheckpointManager",
    "DataVersionManager",
    "create_run_directory",
    "PipelineStage",
    "MLOpsPipeline",
    "build_mlops_pipeline",
    "fit_with_tracking",
    "build_kfold_pipeline",
    "build_multimodel_pipeline",
    "cleanup_runs_and_logs",
    "cleanup_intermediate_files",
]

