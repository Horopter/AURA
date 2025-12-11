"""
MLflow integration for enhanced experiment tracking.

This module provides MLflow integration to complement the existing ExperimentTracker,
enabling better experiment management, artifact tracking, and model registry.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")

# Import RunConfig only for type hints to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lib.mlops.config import RunConfig


class MLflowTracker:
    """
    MLflow wrapper that enhances ExperimentTracker with MLflow capabilities.
    
    This class can be used alongside or instead of ExperimentTracker to provide
    enhanced experiment tracking with MLflow's UI, model registry, and artifact management.
    """
    
    def __init__(
        self,
        experiment_name: str = "fvc_binary_classifier",
        tracking_uri: Optional[str] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking URI (default: local file store)
            run_id: Optional run ID (creates new run if None)
            tags: Optional tags for the run
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(
                "Failed to get/create experiment: %s. Using default.", e
            )
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                pass  # Will use default experiment
        
        # Start run
        if run_id:
            self.run = mlflow.start_run(run_id=run_id)
        else:
            self.run = mlflow.start_run()
        
        self.run_id = self.run.info.run_id
        self.experiment_name = experiment_name
        
        # Set tags
        if tags:
            mlflow.set_tags(tags)
        
        logger.info(f"MLflow run started: {self.run_id} (experiment: {experiment_name})")
    
    def log_config(self, config) -> None:
        """Log RunConfig or dict to MLflow."""
        # Handle both RunConfig objects and dicts
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        # Log as parameters (flatten nested dicts)
        for key, value in config_dict.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (str, int, float, bool)):
                        mlflow.log_param(f"{key}.{sub_key}", sub_value)
            elif isinstance(value, list):
                mlflow.log_param(key, json.dumps(value))
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to MLflow."""
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log a directory of artifacts to MLflow."""
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_model(self, model: Any, artifact_path: str = "model", **kwargs) -> None:
        """Log a PyTorch model to MLflow."""
        mlflow.pytorch.log_model(model, artifact_path, **kwargs)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags on the current run."""
        mlflow.set_tags(tags)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info(f"MLflow run ended: {self.run_id}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
        return False


def create_mlflow_tracker(
    experiment_name: str = "fvc_binary_classifier",
    tracking_uri: Optional[str] = None,
    use_mlflow: bool = True
) -> Optional[MLflowTracker]:
    """
    Create an MLflow tracker if available and requested.
    
    Args:
        experiment_name: Name of MLflow experiment
        tracking_uri: MLflow tracking URI
        use_mlflow: Whether to use MLflow (default: True)
    
    Returns:
        MLflowTracker instance or None
    """
    if not use_mlflow or not MLFLOW_AVAILABLE:
        return None
    
    try:
        return MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)
    except Exception as e:
        logger.warning(f"Failed to create MLflow tracker: {e}")
        return None


__all__ = ["MLflowTracker", "create_mlflow_tracker", "MLFLOW_AVAILABLE"]

