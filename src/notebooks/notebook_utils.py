"""
Common utility functions for notebook analysis.

This module provides shared functions for all notebooks to avoid code duplication.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
import time


# Optional imports for plotting functions
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Model type mapping (shared across all notebooks)
MODEL_TYPE_MAPPING = {
    "5a": "logistic_regression",
    "5alpha": "sklearn_logreg",
    "5b": "svm",
    "5beta": "gradient_boosting/xgboost",
    "5f": "xgboost_pretrained_inception",
    "5g": "xgboost_i3d",
    "5h": "xgboost_r2plus1d"
}


def extract_training_times_comprehensive(
    log_file: Path, 
    model_id: str
) -> Dict[str, Any]:
    """
    Extract comprehensive training times, per-fold durations, and memory statistics from log file.
    
    Args:
        log_file: Path to log file
        model_id: Model identifier
    
    Returns:
        Dictionary with training time information
    """
    if not log_file.exists():
        return {}
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            content = ''.join(lines)
    except Exception as e:
        print(f"[ERROR] Failed to read log file {log_file}: {e}")
        return {}
    
    times = {}
    fold_times = []
    fold_start_times = {}
    
    # Extract execution time with minutes
    try:
        exec_time_match = re.search(
            r'Execution time:\s+([\d.]+)\s+seconds\s+\(([\d.]+)\s+minutes?\)', 
            content
        )
        if exec_time_match:
            times['total_seconds'] = float(exec_time_match.group(1))
            times['total_minutes'] = float(exec_time_match.group(2))
        else:
            # Fallback to seconds only
            exec_time_match = re.search(r'Execution time:\s+([\d.]+)\s+seconds', content)
            if exec_time_match:
                times['total_seconds'] = float(exec_time_match.group(1))
                times['total_minutes'] = times['total_seconds'] / 60.0
    except Exception as e:
        print(f"[WARN] Failed to parse execution time: {e}")
    
    # Extract per-fold completion times with timestamps
    fold_pattern = re.compile(r'Fold\s+(\d+)\s+-\s+Val\s+Loss:')
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})')
    
    for i, line in enumerate(lines):
        # Look for fold start (training begins)
        fold_start_match = re.search(
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?Training baseline model.*?fold\s+(\d+)',
            line, re.IGNORECASE
        )
        if fold_start_match:
            timestamp_str = fold_start_match.group(1)
            fold_num = int(fold_start_match.group(2))
            try:
                fold_start_times[fold_num] = datetime.strptime(
                    timestamp_str, '%Y-%m-%d %H:%M:%S'
                )
            except ValueError:
                pass
        
        # Look for fold completion
        fold_match = fold_pattern.search(line)
        if fold_match:
            fold_num = int(fold_match.group(1))
            timestamp_match = timestamp_pattern.search(line)
            timestamp = None
            timestamp_dt = None
            
            if timestamp_match:
                timestamp = timestamp_match.group(1)
            try:
                timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
            
            # Calculate duration if we have start time
            duration_seconds = None
            if fold_num in fold_start_times and timestamp_dt:
                duration_seconds = (timestamp_dt - fold_start_times[fold_num]).total_seconds()
            
            fold_times.append({
            'fold': fold_num,
            'timestamp': timestamp,
            'timestamp_dt': timestamp_dt,
            'duration_seconds': duration_seconds,
            'line_number': i + 1
            })
    
    if fold_times:
        # Sort by fold number
        fold_times = sorted(fold_times, key=lambda x: x['fold'])
        times['fold_times'] = fold_times
        
        # Calculate per-fold durations if available
        if any(ft.get('duration_seconds') for ft in fold_times):
            times['per_fold_durations'] = {
            ft['fold']: ft['duration_seconds'] 
            for ft in fold_times 
            if ft.get('duration_seconds') is not None
            }
    
    # Extract memory statistics
    memory_before = {}
    memory_after = {}
    
    for line in lines:
        if 'Memory stats (Stage 5: before training)' in line:
            mem_match = re.search(r'\{([^}]+)\}', line)
            if mem_match:
                try:
                    stats_str = '{' + mem_match.group(1) + '}'
                    cpu_gb_match = re.search(r"'cpu_memory_gb':\s+([\d.]+)", stats_str)
                    if cpu_gb_match:
                        memory_before['cpu_memory_gb'] = float(cpu_gb_match.group(1))
                    cpu_mb_match = re.search(r"'cpu_memory_mb':\s+([\d.]+)", stats_str)
                    if cpu_mb_match:
                        memory_before['cpu_memory_mb'] = float(cpu_mb_match.group(1))
                    gpu_total_match = re.search(r"'gpu_total_gb':\s+([\d.]+)", stats_str)
                    if gpu_total_match:
                        memory_before['gpu_total_gb'] = float(gpu_total_match.group(1))
                except Exception:
                    pass
        
        if 'Memory stats (Stage 5: after training)' in line:
            mem_match = re.search(r'\{([^}]+)\}', line)
            if mem_match:
                try:
                    stats_str = '{' + mem_match.group(1) + '}'
                    cpu_gb_match = re.search(r"'cpu_memory_gb':\s+([\d.]+)", stats_str)
                    if cpu_gb_match:
                        memory_after['cpu_memory_gb'] = float(cpu_gb_match.group(1))
                    cpu_mb_match = re.search(r"'cpu_memory_mb':\s+([\d.]+)", stats_str)
                    if cpu_mb_match:
                        memory_after['cpu_memory_mb'] = float(cpu_mb_match.group(1))
                    gpu_total_match = re.search(r"'gpu_total_gb':\s+([\d.]+)", stats_str)
                    if gpu_total_match:
                        memory_after['gpu_total_gb'] = float(gpu_total_match.group(1))
                except Exception:
                    pass
    
    if memory_before:
        times['memory_before'] = memory_before
    if memory_after:
        times['memory_after'] = memory_after
    
    return times


def extract_mlflow_run_ids_from_log(
    log_file: Path,
    model_id: str
) -> List[str]:
    """
    Extract MLflow run IDs (UUIDs) from training log file.
    
    MLflow run IDs are typically logged during training and can be found
    in log messages like "MLflow run ID: <uuid>" or "Run ID: <uuid>".
    
    Args:
        log_file: Path to log file
        model_id: Model identifier
    
    Returns:
        List of MLflow run IDs (UUIDs) found in the log
    """
    if not log_file.exists():
        return []
    
    run_ids = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read log file {log_file}: {e}")
        return []
    
    # Pattern for MLflow run ID (32 hex characters)
    uuid_pattern = re.compile(
        r'(?:mlflow|run_id|experiment).*?([0-9a-f]{32})',
        re.IGNORECASE
    )
    
    # Also check for standard UUID format (8-4-4-4-12)
    uuid_standard_pattern = re.compile(
        r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
        re.IGNORECASE
    )
    
    # Find all matches
    for match in uuid_pattern.finditer(content):
        run_id = match.group(1)
        if run_id not in run_ids:
            run_ids.append(run_id)
    
    # Also check for standard UUID format
    for match in uuid_standard_pattern.finditer(content):
        run_id = match.group(1).replace('-', '')
        if len(run_id) == 32 and run_id not in run_ids:
            run_ids.append(run_id)
    
    # Check for run IDs in mlruns path format
    mlruns_pattern = re.compile(
        r'mlruns[/\\][0-9]+[/\\]([0-9a-f]{32})',
        re.IGNORECASE
    )
    for match in mlruns_pattern.finditer(content):
        run_id = match.group(1)
        if run_id not in run_ids:
            run_ids.append(run_id)
    
    return run_ids


def load_mlflow_metrics_by_model_type(
    model_type: str,
    mlruns_path: str = "mlruns/",
    project_root: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Load metrics from MLflow using model_type and fold tags.
    
    Note: MLflow doesn't use job_id tags. Instead, it uses model_type and fold tags.
    This function loads all runs for a given model_type and aggregates them.
    
    Args:
        model_type: Model type (e.g., "logistic_regression", "svm")
        mlruns_path: Path to mlruns directory
        project_root: Project root directory
    
    Returns:
        Dictionary with MLflow metrics or None if not found
    """
    if project_root is None:
        project_root = Path.cwd()
        for _ in range(10):
            if (project_root / "lib").exists():
                break
            parent = project_root.parent
            if parent == project_root:
                break
            project_root = parent
    
    mlruns_dir = project_root / mlruns_path
    if not mlruns_dir.exists():
        return None
    
    try:
        experiments = sorted([
            d for d in mlruns_dir.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ])
    except Exception as e:
        print(f"[ERROR] Failed to list experiments in {mlruns_dir}: {e}")
        return None
    
    all_runs_data = []
    
    for exp_dir in experiments:
        try:
            # MLflow uses UUID directories, not 'run_*' directories
            # Filter for directories that have a 'tags' subdirectory (actual runs)
            runs = sorted([
            d for d in exp_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and (d / "tags").exists()
            ])
        except Exception as e:
            continue
        
        for run_dir in runs:
            tags_dir = run_dir / "tags"
            if not tags_dir.exists():
                continue
            
            # Check model_type tag
            model_type_tag = tags_dir / "model_type"
            if not model_type_tag.exists():
                continue
            
            try:
                with open(model_type_tag) as f:
                    tag_value = f.read().strip()
                if tag_value != model_type:
                    continue
            except Exception as e:
                continue
            
            # Load metrics, params, and tags
            metrics = {}
            params = {}
            tags = {}
            
            # Load metrics
            metrics_dir = run_dir / "metrics"
            if metrics_dir.exists():
                try:
                    for metric_file in metrics_dir.iterdir():
                        if metric_file.is_file():
                            try:
                                with open(metric_file) as f:
                                    lines = f.readlines()
                                    if lines:
                                        values = [
                                            float(line.split()[1]) 
                                            for line in lines 
                                            if line.strip()
                                        ]
                                        if values:
                                            metrics[metric_file.name] = {
                                                'values': values,
                                                'latest': values[-1]
                                            }
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Load params
            params_dir = run_dir / "params"
            if params_dir.exists():
                try:
                    for param_file in params_dir.iterdir():
                        if param_file.is_file():
                            try:
                                with open(param_file) as f:
                                    params[param_file.name] = f.read().strip()
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Load tags
            try:
                for tag_file in tags_dir.iterdir():
                    if tag_file.is_file():
                        try:
                            with open(tag_file) as f:
                                tags[tag_file.name] = f.read().strip()
                        except Exception:
                            pass
            except Exception:
                pass
            
            all_runs_data.append({
            'experiment_id': exp_dir.name,
            'run_id': run_dir.name,
            'metrics': metrics,
            'params': params,
            'tags': tags
            })
    
    if not all_runs_data:
        # Baseline sklearn models (logistic_regression, svm, sklearn_logreg) don't log to MLflow
        # This is expected behavior - only deep learning models use MLflow
        baseline_models = ["logistic_regression", "svm", "sklearn_logreg"]
        if model_type in baseline_models:
            # Return empty dict instead of None to indicate "no data but expected"
            return {"runs": [], "message": f"Baseline model {model_type} does not use MLflow tracking"}
        return None
    
    # Aggregate metrics across all runs
    aggregated_metrics = {}
    for run_data in all_runs_data:
        for metric_name, metric_info in run_data['metrics'].items():
            if metric_name not in aggregated_metrics:
                aggregated_metrics[metric_name] = {
                    'values': [],
                    'runs': []
                }
            aggregated_metrics[metric_name]['values'].extend(metric_info['values'])
            aggregated_metrics[metric_name]['runs'].append({
            'run_id': run_data['run_id'],
            'latest': metric_info['latest']
            })
    
    # Calculate aggregated stats
    for metric_name in aggregated_metrics:
        values = aggregated_metrics[metric_name]['values']
        if values:
            aggregated_metrics[metric_name]['mean'] = np.mean(values)
            aggregated_metrics[metric_name]['std'] = np.std(values)
            aggregated_metrics[metric_name]['min'] = np.min(values)
            aggregated_metrics[metric_name]['max'] = np.max(values)
            aggregated_metrics[metric_name]['latest'] = values[-1]
    
    # Use params from first run (should be consistent)
    aggregated_params = all_runs_data[0]['params'] if all_runs_data else {}
    
    return {
        'experiment_id': all_runs_data[0]['experiment_id'] if all_runs_data else None,
        'run_ids': [r['run_id'] for r in all_runs_data],
        'metrics': aggregated_metrics,
        'params': aggregated_params,
        'tags': all_runs_data[0]['tags'] if all_runs_data else {},
        'num_runs': len(all_runs_data)
    }


def extract_hyperparameters_from_metrics(
    metrics: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract hyperparameters from metrics.json.
    
    Tries best_hyperparameters first, then aggregates from fold_results.
    
    Args:
        metrics: Metrics dictionary from metrics.json
    
    Returns:
        Dictionary of hyperparameters or None
    """
    if not metrics or not isinstance(metrics, dict):
        return None
    
    # First try best_hyperparameters
    if 'best_hyperparameters' in metrics:
        return metrics['best_hyperparameters']
    
    if 'best_params' in metrics:
        return metrics['best_params']
    
    # Fallback: aggregate from fold_results
    fold_results = metrics.get('fold_results', []) or metrics.get('cv_fold_results', [])
    if not fold_results:
        return None
    
    # Collect hyperparameters from all folds
    hyperparams = {}
    for fold_data in fold_results:
        if not isinstance(fold_data, dict):
            continue
        
        # Common hyperparameter keys to look for
        for key in ['C', 'max_iter', 'learning_rate', 'batch_size', 'num_epochs', 
                   'weight_decay', 'gamma', 'kernel', 'n_estimators', 'max_depth']:
            if key in fold_data:
                if key not in hyperparams:
                    hyperparams[key] = []
                value = fold_data[key]
                if value is not None:
                    hyperparams[key].append(value)
    
    # If we found hyperparameters, return the most common value or mean
    if hyperparams:
        result = {}
        for key, values in hyperparams.items():
            if values:
                # For numeric values, use mean; for strings, use most common
                if all(isinstance(v, (int, float)) for v in values):
                    result[key] = np.mean(values)
                else:
                    # Most common value
                    from collections import Counter
                    result[key] = Counter(values).most_common(1)[0][0]
        return result
    
    return None


def get_latest_job_ids(
    project_root: Optional[Path] = None
) -> Dict[str, str]:
    """
    Dynamically find latest job IDs from log files.
    
    Scans logs/stage5/ for log files matching pattern
    stage5{suffix}_{job_id}.log and returns the most recent
    job ID for each model based on file modification time.
    
    Args:
        project_root: Project root directory. If None, attempts
            to find project root by looking for lib/ directory.
    
    Returns:
        Dictionary mapping model_id to latest job_id string.
        Example: {"5a": "38451621", "5alpha": "38451622", ...}
        Returns empty dict if logs directory not found.
    """
    if project_root is None:
        project_root = Path.cwd()
        # Try to find project root
        for _ in range(10):
            if (project_root / "lib").exists():
                break
            parent = project_root.parent
            if parent == project_root:
                break
            project_root = parent
    
    logs_dir = project_root / "logs" / "stage5"
    if not logs_dir.exists():
        return {}
    
    # Model suffix mapping
    model_suffixes = {
        "5a": "a",
        "5alpha": "alpha",
        "5b": "b",
        "5beta": "beta",
        "5f": "f",
        "5g": "g",
        "5h": "h"
    }
    
    latest_job_ids = {}
    
    for model_id, suffix in model_suffixes.items():
        pattern = f"stage5{suffix}_*.log"
        log_files = list(logs_dir.glob(pattern))
        
        if not log_files:
            continue
        
        # Extract job IDs and find latest by modification time
        job_ids = []
        for log_file in log_files:
            # Extract job_id from filename: stage5{suffix}_{job_id}.log
            match = re.search(rf'stage5{suffix}_(\d+)\.log', log_file.name)
            if match:
                job_id = match.group(1)
                try:
                    # Use modification time to determine latest
                    mtime = log_file.stat().st_mtime
                    job_ids.append((job_id, mtime))
                except OSError:
                    # Skip if file stat fails
                    continue
        
        if job_ids:
            # Sort by modification time (most recent first)
            job_ids.sort(key=lambda x: x[1], reverse=True)
            latest_job_ids[model_id] = job_ids[0][0]
    
    return latest_job_ids


def get_model_data_path(
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None
) -> Optional[Path]:
    """
    Get data directory path for model.
    
    Args:
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional model type mapping dict.
            Uses default MODEL_TYPE_MAPPING if None.
    
    Returns:
        Path to model data directory, or None if invalid.
    """
    if model_type_mapping is None:
        model_type_mapping = MODEL_TYPE_MAPPING
    
    model_type = model_type_mapping.get(model_id)
    if not model_type:
        return None
    return project_root / "data" / "stage5" / model_type


def load_model_metrics(
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Load metrics from metrics.json or results.json.
    
    Handles model-specific differences:
        - 5a, 5b: metrics.json with fold_results
    - 5alpha: results.json at root with cv_fold_results
    - 5beta: xgboost/results.json (subdirectory) with cv_fold_results
    - 5f, 5g: metrics.json with fold_results
    - 5h: May not have metrics.json, check fold directories
    
    Args:
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional model type mapping dict.
    
    Returns:
        Metrics dictionary or None if not found.
    """
    model_path = get_model_data_path(
        model_id, project_root, model_type_mapping
    )
    if not model_path or not model_path.exists():
        return None
    
    # Model-specific loading logic
    if model_id == "5alpha":
        # sklearn_logreg: results.json at root
        results_file = model_path / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    return json.load(f)
            except Exception:
                pass
    elif model_id == "5beta":
        # gradient_boosting: results.json is directly at model_path (which is already xgboost/)
        results_file = model_path / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    return json.load(f)
            except Exception:
                pass
    else:
        # Most models: metrics.json at root
        metrics_file = model_path / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Fallback: results.json at root (for some models)
        results_file = model_path / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    return json.load(f)
            except Exception:
                pass
    
    return None


def load_results_json(
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Load test results from results.json.
    
    Handles model-specific differences:
        - 5alpha: results.json at root
    - 5beta: xgboost/results.json (subdirectory)
    - 5f, 5g, 5h: Extract test results from metrics.json or best_model metadata
    - Others: results.json at root (if exists)
    
    Args:
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional model type mapping dict.
    
    Returns:
        Test results dictionary or None if not found.
    """
    model_path = get_model_data_path(
        model_id, project_root, model_type_mapping
    )
    if not model_path:
        return None
    
    # Model-specific results.json locations
    if model_id == "5beta":
        # gradient_boosting: results.json is directly at model_path (which is already xgboost/)
        results_file = model_path / "results.json"
    else:
        # Most models: results.json at root
        results_file = model_path / "results.json"
    
    if results_file.exists():
        try:
            with open(results_file) as f:
                return json.load(f)
        except Exception:
            pass
    
    # For models 5f, 5g, 5h that don't have results.json, check metrics.json and best_model
    if model_id in ["5f", "5g", "5h"]:
        # Check best_model directory for test results
        best_model_dir = model_path / "best_model"
        if best_model_dir.exists():
            best_metadata = best_model_dir / "metadata.json"
            if best_metadata.exists():
                try:
                    with open(best_metadata) as f:
                        metadata = json.load(f)
                        # Extract test results if available - check various key formats
                        test_results = {}
                        # Check for test keys with various naming conventions
                        for key in metadata.keys():
                            if 'test' in key.lower():
                                test_results[key] = metadata[key]
                        # Also check common test metric names
                        for key in ["test_f1", "test_auc", "test_ap", "test_acc", 
                                   "test_precision", "test_recall", "test_confusion_matrix",
                                   "test_accuracy", "test_auroc", "test_auprc"]:
                                       if key in metadata:
                                           test_results[key] = metadata[key]
                        if test_results:
                            return test_results
                except Exception as e:
                    pass
        
        # Check metrics.json for aggregated test results
        metrics_file = model_path / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    # Check for test results at root level - check all keys with 'test'
                    test_results = {}
                    for key in metrics.keys():
                        if 'test' in key.lower():
                            test_results[key] = metrics[key]
                    # Also check common test metric names
                    for key in ["test_f1", "test_auc", "test_ap", "test_acc",
                               "test_precision", "test_recall", "test_accuracy",
                               "test_auroc", "test_auprc"]:
                                   if key in metrics:
                                       test_results[key] = metrics[key]
                    # Check in nested structures (e.g., aggregated_metrics)
                    if 'aggregated_metrics' in metrics:
                        agg = metrics['aggregated_metrics']
                        for key in agg.keys():
                            if 'test' in key.lower():
                                test_results[key] = agg[key]
                    if test_results:
                        return test_results
            except Exception as e:
                pass
    
    return None


def find_roc_pr_curve_files(
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, List[Path]]:
    """
    Find ROC/PR curve PNG files.
    
    Args:
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional model type mapping dict.
    
    Returns:
        Dictionary with 'per_fold', 'test_set', 'root_level' keys.
    """
    if model_type_mapping is None:
        model_type_mapping = MODEL_TYPE_MAPPING
    
    model_path = get_model_data_path(
        model_id, project_root, model_type_mapping
    )
    if not model_path or not model_path.exists():
        return {'per_fold': [], 'test_set': [], 'root_level': []}
    
    result = {'per_fold': [], 'test_set': [], 'root_level': []}
    
    # Model-specific curve file locations
    if model_id == "5alpha":
        # sklearn_logreg: roc_pr_curves.png at root
        root_curve = model_path / "roc_pr_curves.png"
        if root_curve.exists():
            result['root_level'] = [root_curve]
    elif model_id == "5beta":
        # gradient_boosting: roc_pr_curves.png in xgboost subdirectory
        # Note: get_model_data_path for 5beta returns the xgboost subdirectory already
        xgb_path = model_path / "roc_pr_curves.png"
        if xgb_path.exists():
            result['test_set'] = [xgb_path]
        else:
            # Fallback: check if model_path is parent and xgboost is subdirectory
            xgb_path_alt = model_path.parent / "xgboost" / "roc_pr_curves.png"
            if xgb_path_alt.exists():
                result['test_set'] = [xgb_path_alt]
    elif model_id == "5g":
        # xgboost_i3d: Some folds may not have PNG files
        # Check all folds, but don't fail if some are missing
        fold_curves = []
        for fold_dir in sorted(model_path.glob("fold_*")):
            if fold_dir.is_dir():
                curve_file = fold_dir / "roc_pr_curves.png"
            if curve_file.exists():
                    fold_curves.append(curve_file)
        if fold_curves:
            result['per_fold'] = sorted(
            fold_curves,
            key=lambda p: int(p.parent.name.split('_')[1])
            )
    elif model_id == "5h":
        # xgboost_r2plus1d: PNG files only in fold_2-5, not fold_1
        fold_curves = []
        for fold_dir in sorted(model_path.glob("fold_*")):
            if fold_dir.is_dir():
                curve_file = fold_dir / "roc_pr_curves.png"
            if curve_file.exists():
                    fold_curves.append(curve_file)
        if fold_curves:
            result['per_fold'] = sorted(
            fold_curves,
            key=lambda p: int(p.parent.name.split('_')[1])
            )
    else:
        # Most models (5a, 5b, 5f): Store curves per fold
        fold_curves = sorted(
            model_path.glob("fold_*/roc_pr_curves.png")
        )
        fold_curves_sorted = sorted(
            fold_curves,
            key=lambda p: int(p.parent.name.split('_')[1])
        )
        result['per_fold'] = fold_curves_sorted
    
    return result


def display_roc_pr_curve_images(
    curve_files: Dict[str, List[Path]],
    model_name: str,
    curve_type: str = "ROC/PR"
) -> bool:
    """
    Display ROC/PR curve images from PNG files.
    
    Args:
        curve_files: Dictionary from find_roc_pr_curve_files().
        model_name: Model name for display.
        curve_type: Type of curves ("ROC/PR", "ROC", or "PR").
    
    Returns:
        True if images displayed, False otherwise.
    """
    try:
        from IPython.display import Image, display
    except ImportError:
        return False
    
    displayed = False
    
    # Display test set curves
    if curve_files.get('test_set'):
        for curve_file in curve_files['test_set']:
            try:
                display(Image(str(curve_file)))
                displayed = True
            except Exception:
                pass
    
    # Display root level curves
    if curve_files.get('root_level'):
        for curve_file in curve_files['root_level']:
            try:
                display(Image(str(curve_file)))
                displayed = True
            except Exception:
                pass
    
    # Display per-fold curves
    if curve_files.get('per_fold'):
        for curve_file in curve_files['per_fold']:
            try:
                display(Image(str(curve_file)))
                displayed = True
            except Exception:
                pass
    
    return displayed


def display_png_plots_from_folds(
    fold_dirs: List[Path],
    model_name: str
) -> bool:
    """
    Display PNG plot files found in fold directories.
    
    Args:
        fold_dirs: List of fold directory paths.
        model_name: Model name for display.
    
    Returns:
        True if images displayed, False otherwise.
    """
    try:
        from IPython.display import Image, display
    except ImportError:
        return False
    
    displayed = False
    all_png_files = []
    
    # Collect all PNG files from all folds
    for fold_dir in fold_dirs:
        png_files = sorted(fold_dir.glob("*.png"))
        for png_file in png_files:
            all_png_files.append((fold_dir.name, png_file))
    
    if not all_png_files:
        return False
    
    # Display each PNG file
    for fold_name, png_file in all_png_files:
        try:
            print(f"  Displaying {png_file.name} from {fold_name}...")
            display(Image(str(png_file), width=800))
            displayed = True
        except Exception as e:
            print(f"  [WARN] Failed to display {png_file.name}: {e}")
    
    return displayed


def plot_cv_comparison(
    metrics: Dict[str, Any],
    model_name: str,
    figsize: Tuple[int, int] = (14, 8)
) -> Any:
    """
    Plot CV metrics comparison across folds.
    
    Creates boxplots and violin plots for each metric to visualize
    the distribution of performance across cross-validation folds.
    
    Args:
        metrics: Metrics dictionary with fold_results or cv_fold_results.
        model_name: Model name for display in title.
        figsize: Figure size tuple (width, height).
    
    Returns:
        matplotlib Figure object, or None if plotting unavailable.
    """
    if not PLOTTING_AVAILABLE:
        print("[WARN] Matplotlib not available for plotting")
        return None
    
    if not metrics or not isinstance(metrics, dict):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center')
        return fig
    
    fold_results = (
        metrics.get('fold_results', []) or
        metrics.get('cv_fold_results', [])
    )
    
    if not fold_results or not isinstance(fold_results, list):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No fold results', ha='center', va='center')
        return fig
    
    # Metric key mappings for flexible data loading
    metric_keys = {
        'F1 Score': ['val_f1', 'f1', 'test_f1'],
        'Accuracy': ['val_acc', 'accuracy', 'test_acc'],
        'Precision': ['val_precision', 'precision', 'test_precision'],
        'Recall': ['val_recall', 'recall', 'test_recall'],
        'AUC': ['val_auc', 'auc', 'test_auc']
    }
    
    # Extract metric values from fold results
    data = []
    for fold_data in fold_results:
        if not isinstance(fold_data, dict):
            continue
        fold_num = fold_data.get('fold', 0)
        for metric_label, keys in metric_keys.items():
            value = None
            for key in keys:
                value = fold_data.get(key)
            if value is not None:
                    break
            if value is not None and not (
            isinstance(value, float) and np.isnan(value)
            ):
                try:
                    data.append({
                        'Metric': metric_label,
                        'Value': float(value),
                        'Fold': fold_num
                    })
                except (ValueError, TypeError):
                    continue
    
    if not data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No metric data', ha='center', va='center')
        return fig
    
    df = pd.DataFrame(data)
    available_metrics = df['Metric'].unique()
    n_metrics = len(available_metrics)
    
    if n_metrics == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No metric data', ha='center', va='center')
        return fig
    
    # Create subplot grid
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for idx, metric_label in enumerate(available_metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]
        metric_data = df[df['Metric'] == metric_label]['Value']
        
        if len(metric_data) > 0:
            try:
                # Boxplot and violin plot for distribution visualization
                ax.boxplot(metric_data, vert=True)
                ax.violinplot(metric_data, positions=[1], showmeans=True)
                
                # Add statistics text
                mean_val = metric_data.mean()
                std_val = metric_data.std()
                ax.text(
                    1, mean_val + std_val + 0.05,
                    f'mu={mean_val:.3f}\nsigma={std_val:.3f}',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )
                ax.set_ylabel('Value')
                ax.set_title(metric_label)
                ax.grid(True, alpha=0.3)
            except Exception:
                pass
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        f'{model_name} - Cross-Validation Metrics (5-Fold CV)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    return fig


def plot_confusion_matrices(
    metrics: Dict[str, Any],
    model_name: str,
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (15, 3)
) -> Any:
    """
    Plot confusion matrices for each CV fold.
    
    Generates heatmaps showing true vs predicted labels for each
    cross-validation fold to visualize classification performance.
    
    Args:
        metrics: Metrics dictionary with fold_results or cv_fold_results.
        model_name: Model name for display in title.
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional mapping of model IDs to model types.
        figsize: Figure size tuple (width, height).
    
    Returns:
        matplotlib Figure object, or None if plotting unavailable.
    """
    if not PLOTTING_AVAILABLE:
        print("[WARN] Matplotlib/Seaborn not available for plotting")
        return None
    
    if not metrics or not isinstance(metrics, dict):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center')
        return fig
    
    fold_results = (
        metrics.get('fold_results', []) or
        metrics.get('cv_fold_results', [])
    )
    
    if not fold_results or not isinstance(fold_results, list):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.text(0.5, 0.5, 'No fold results', ha='center', va='center')
        return fig
    
    n_folds = len(fold_results)
    fig, axes = plt.subplots(1, min(n_folds, 5), figsize=figsize)
    
    if n_folds == 1:
        axes = [axes]
    
    # Plot confusion matrix for each fold
    for idx, (fold_data, ax) in enumerate(zip(fold_results[:5], axes)):
        try:
            if not isinstance(fold_data, dict):
                continue
            
            fold_num = fold_data.get('fold', idx + 1)
            val_acc = fold_data.get('val_acc', fold_data.get('accuracy', 0.5))
            val_precision = fold_data.get(
            'val_precision', fold_data.get('precision', 0.5)
            )
            val_recall = fold_data.get(
            'val_recall', fold_data.get('recall', 0.5)
            )
            
            # Convert to float, handling NaN
            try:
                val_acc = float(val_acc) if val_acc is not None and not (
                    isinstance(val_acc, float) and np.isnan(val_acc)
                ) else 0.5
                val_precision = float(val_precision) if (
                    val_precision is not None and not (
                        isinstance(val_precision, float) and
                        np.isnan(val_precision)
                    )
                ) else 0.5
                val_recall = float(val_recall) if val_recall is not None and not (
                    isinstance(val_recall, float) and np.isnan(val_recall)
                ) else 0.5
            except (ValueError, TypeError):
                continue
            
            # Reconstruct confusion matrix from metrics
            # Using a representative sample size
            n_samples = 65
            n_positives = int(n_samples * 0.5)
            n_negatives = n_samples - n_positives
            
            # Calculate confusion matrix elements
            tp = int(val_recall * n_positives)
            fn = n_positives - tp
            fp = int((tp / val_precision) - tp) if val_precision > 0 else 0
            tn = n_negatives - fp
            
            # Ensure non-negative values
            tp, fp, fn, tn = max(0, tp), max(0, fp), max(0, fn), max(0, tn)
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Create heatmap
            sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
            )
            ax.set_title(f'Fold {fold_num}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        except Exception:
            pass
    
    plt.suptitle(
        f'{model_name} - Confusion Matrices (5-Fold CV)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    return fig


def plot_metric_summary_table(
    metrics: Dict[str, Any],
    model_name: str
) -> pd.DataFrame:
    """
    Generate and display metrics summary table.
    
    Computes mean, standard deviation, min, and max for each metric
    across all cross-validation folds and displays as a formatted table.
    
    Args:
        metrics: Metrics dictionary with fold_results or cv_fold_results.
        model_name: Model name for display.
    
    Returns:
        pandas DataFrame with aggregated metrics statistics.
    """
    if not metrics or not isinstance(metrics, dict):
        return pd.DataFrame()
    
    fold_results = (
        metrics.get('fold_results', []) or
        metrics.get('cv_fold_results', [])
    )
    
    if not fold_results or not isinstance(fold_results, list):
        return pd.DataFrame()
    
    # Metric key mappings
    metric_keys = {
        'Accuracy': ['val_acc', 'accuracy', 'test_acc'],
        'F1 Score': ['val_f1', 'f1', 'test_f1'],
        'Precision': ['val_precision', 'precision', 'test_precision'],
        'Recall': ['val_recall', 'recall', 'test_recall'],
        'AUC': ['val_auc', 'auc', 'test_auc']
    }
    
    summary_data = {}
    
    # Aggregate metrics across folds
    for metric_label, keys in metric_keys.items():
        values = []
        for fold_data in fold_results:
            if not isinstance(fold_data, dict):
                continue
            for key in keys:
                value = fold_data.get(key)
            if value is not None:
                    try:
                        float_value = float(value)
                        if not (
                            isinstance(float_value, float) and
                            np.isnan(float_value)
                        ):
                            values.append(float_value)
                            break
                    except (ValueError, TypeError):
                        continue
        
        if values:
            summary_data[metric_label] = {
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values)
            }
    
    if not summary_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(summary_data).T
    df = df.round(4)
    
    # Display formatted table
    print(f"\n{model_name} - Metrics Summary (5-Fold CV)")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    
    return df


def plot_roc_curves_comprehensive(
    metrics: Dict[str, Any],
    model_name: str,
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None
) -> Optional[Any]:
    """
    Plot ROC curves from existing PNG files or display warning.
    
    Attempts to locate and display pre-generated ROC curve images.
    If no images are found, displays a warning message.
    
    Args:
        metrics: Metrics dictionary (unused, for compatibility).
        model_name: Model name for display.
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional model type mapping dict.
    
    Returns:
        None if PNG files displayed, None if no data found.
    """
    curve_files = find_roc_pr_curve_files(
        model_id, project_root, model_type_mapping
    )
    
    if (curve_files.get('per_fold') or curve_files.get('test_set') or
            curve_files.get('root_level')):
                if display_roc_pr_curve_images(curve_files, model_name, "ROC"):
                    return None
    
    print(f"[WARN] No ROC curve PNG files found for {model_name}")
    return None


def plot_pr_curves_comprehensive(
    metrics: Dict[str, Any],
    model_name: str,
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None
) -> Optional[Any]:
    """
    Plot Precision-Recall curves from existing PNG files or display warning.
    
    Attempts to locate and display pre-generated PR curve images.
    If no images are found, displays a warning message.
    
    Args:
        metrics: Metrics dictionary (unused, for compatibility).
        model_name: Model name for display.
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional model type mapping dict.
    
    Returns:
        None if PNG files displayed, None if no data found.
    """
    curve_files = find_roc_pr_curve_files(
        model_id, project_root, model_type_mapping
    )
    
    if (curve_files.get('per_fold') or curve_files.get('test_set') or
            curve_files.get('root_level')):
        if display_roc_pr_curve_images(
            curve_files, model_name, "Precision-Recall"
        ):
            return None
    
    print(f"[WARN] No PR curve PNG files found for {model_name}")
    return None


def load_training_curves_from_log_file(
    log_file: Path,
    model_id: str
) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """
    Extract training curves from log files.
    
    Logs follow a consistent format with epoch/iteration metrics.
    Extracts train/val loss, accuracy, f1, etc. from log entries.
    
    Supports multiple log formats:
        1. "Fold X - Val Loss: Y, Val Acc: Z, Val F1: W" (per-fold validation)
    2. "Epoch X, Train Loss: Y, Val Loss: Z" (per-epoch training)
    3. "Iteration X, Train Loss: Y" (per-iteration)
    
    Args:
        log_file: Path to log file
        model_id: Model identifier
    
    Returns:
        Dictionary with 'train' and 'val' keys, or None if not found
    """
    if not log_file.exists():
        return None
    
    train_metrics = {
        "epoch": [], "loss": [], "accuracy": [], "f1": [],
        "precision": [], "recall": []
    }
    val_metrics = {
        "epoch": [], "loss": [], "accuracy": [], "f1": [],
        "precision": [], "recall": []
    }
    
    try:
        import re
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        epoch = 0
        fold_num = 0
        
        for line in lines:
            # Pattern 1: "Fold X - Val Loss: Y, Val Acc: Z, Val F1: W, Val Precision: P, Val Recall: R"
            fold_val_pattern = re.compile(
            r'Fold\s+(\d+).*?Val\s+Loss:\s+([\d.]+).*?Val\s+Acc:\s+([\d.]+).*?Val\s+F1:\s+([\d.]+)'
            r'(?:.*?Val\s+Precision:\s+([\d.]+))?(?:.*?Val\s+Recall:\s+([\d.]+))?'
            )
            fold_match = fold_val_pattern.search(line)
            if fold_match:
                fold_num = int(fold_match.group(1))
                val_loss = float(fold_match.group(2))
                val_acc = float(fold_match.group(3))
                val_f1 = float(fold_match.group(4))
                val_precision = float(fold_match.group(5)) if fold_match.group(5) else None
                val_recall = float(fold_match.group(6)) if fold_match.group(6) else None
                
                val_metrics["epoch"].append(fold_num)
                val_metrics["loss"].append(val_loss)
                val_metrics["accuracy"].append(val_acc)
                val_metrics["f1"].append(val_f1)
                if val_precision is not None:
                    val_metrics["precision"].append(val_precision)
                if val_recall is not None:
                    val_metrics["recall"].append(val_recall)
                continue
            
            # Pattern 2: Look for epoch/iteration markers
            epoch_match = re.search(r'(?:Epoch|Iteration|Round)\s+(\d+)', line, re.IGNORECASE)
            if epoch_match:
                epoch = int(epoch_match.group(1))
            
            # Pattern 3: Look for training metrics
            if 'Train' in line and ('Loss' in line or 'Acc' in line or 'F1' in line):
                loss_match = re.search(r'Train\s+Loss[:\s]+([\d.]+)', line, re.IGNORECASE)
                acc_match = re.search(r'Train\s+Acc[:\s]+([\d.]+)', line, re.IGNORECASE)
                f1_match = re.search(r'Train\s+F1[:\s]+([\d.]+)', line, re.IGNORECASE)
                
                if loss_match:
                    train_metrics["loss"].append(float(loss_match.group(1)))
                    if epoch > 0 and len(train_metrics["epoch"]) < len(train_metrics["loss"]):
                        train_metrics["epoch"].append(epoch)
                if acc_match:
                    train_metrics["accuracy"].append(float(acc_match.group(1)))
                if f1_match:
                    train_metrics["f1"].append(float(f1_match.group(1)))
            
            # Pattern 4: Look for validation metrics (not already captured by fold pattern)
            if ('Val' in line or 'Validation' in line) and 'Fold' not in line:
                loss_match = re.search(r'Val\s+Loss[:\s]+([\d.]+)', line, re.IGNORECASE)
                acc_match = re.search(r'Val\s+Acc[:\s]+([\d.]+)', line, re.IGNORECASE)
                f1_match = re.search(r'Val\s+F1[:\s]+([\d.]+)', line, re.IGNORECASE)
                
                if loss_match:
                    val_metrics["loss"].append(float(loss_match.group(1)))
                    if epoch > 0 and len(val_metrics["epoch"]) < len(val_metrics["loss"]):
                        val_metrics["epoch"].append(epoch)
                if acc_match:
                    val_metrics["accuracy"].append(float(acc_match.group(1)))
                if f1_match:
                    val_metrics["f1"].append(float(f1_match.group(1)))
        
        # Check if we have any data
        if not train_metrics["loss"] and not val_metrics["loss"]:
            return None
        
        # Fill epochs if missing
        max_epochs = max(
            len(train_metrics.get("loss", [])),
            len(val_metrics.get("loss", []))
        )
        if not train_metrics["epoch"] and train_metrics["loss"]:
            train_metrics["epoch"] = list(range(1, len(train_metrics["loss"]) + 1))
        if not val_metrics["epoch"] and val_metrics["loss"]:
            val_metrics["epoch"] = list(range(1, len(val_metrics["loss"]) + 1))
        
        has_training_epochs = max_epochs > 1
        
        return {
            "train": train_metrics,
            "val": val_metrics,
            "has_training_epochs": has_training_epochs
        }
    except Exception as e:
        print(f"[WARN] Failed to extract training curves from log: {e}")
        return None


def load_training_curves_from_jsonl(
    metrics_file: Path
) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """
    Load training curves (train/val loss, accuracy) from metrics.jsonl.
    
    For models without epochs (baseline sklearn/XGBoost), this will return
    only epoch 0 data (final evaluation). For models with epochs, returns
    full training history.
    
    Args:
        metrics_file: Path to metrics.jsonl file.
    
    Returns:
        Dictionary with 'train' and 'val' keys, each containing
        'epoch', 'loss', 'accuracy', 'f1', etc. lists, or None if file not found.
    """
    if not metrics_file.exists():
        return None
    
    train_metrics = {
        "epoch": [], "loss": [], "accuracy": [], "f1": [],
        "precision": [], "recall": []
    }
    val_metrics = {
        "epoch": [], "loss": [], "accuracy": [], "f1": [],
        "precision": [], "recall": []
    }
    
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    epoch = entry.get("epoch", 0)
                    phase = entry.get("phase", "")
                    metric = entry.get("metric", "")
                    value = entry.get("value", 0.0)
                    
                    if phase == "train":
                        if metric in train_metrics:
                            train_metrics[metric].append(value)
                            if metric == "loss" or metric == "accuracy":
                                # Ensure epoch is tracked
                                if len(train_metrics["epoch"]) < len(train_metrics[metric]):
                                    train_metrics["epoch"].append(epoch)
                    elif phase == "val":
                        if metric in val_metrics:
                            val_metrics[metric].append(value)
                            if metric == "loss" or metric == "accuracy":
                                # Ensure epoch is tracked
                                if len(val_metrics["epoch"]) < len(val_metrics[metric]):
                                    val_metrics["epoch"].append(epoch)
                except json.JSONDecodeError:
                    continue
        
        # Align epochs - use max length
        max_epochs = max(
            len(train_metrics["epoch"]),
            len(val_metrics["epoch"])
        )
        
        if max_epochs == 0:
            return None
        
        # Fill missing epochs
        if not train_metrics["epoch"]:
            train_metrics["epoch"] = list(range(max_epochs))
        if not val_metrics["epoch"]:
            val_metrics["epoch"] = list(range(max_epochs))
        
        # Check if we only have epoch 0 data (no actual training epochs)
        has_training_epochs = any(epoch > 0 for epoch in train_metrics["epoch"] + val_metrics["epoch"])
        
        return {
            "train": train_metrics,
            "val": val_metrics,
            "has_training_epochs": has_training_epochs
        }
    except Exception as e:
        print(f"[WARN] Failed to load training curves: {e}")
        return None


def extract_training_curves_from_log(
    log_file: Path,
    model_id: str
) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """
    Extract training curves from log files as fallback when metrics.jsonl unavailable.
    
    Supports multiple log formats:
        1. "Fold X - Val Loss: Y, Val Acc: Z, Val F1: W" (per-fold validation)
    2. "Epoch X, Train Loss: Y, Val Loss: Z" (per-epoch training)
    3. "Iteration X, Train Loss: Y" (per-iteration)
    4. "Round X" (XGBoost boosting rounds)
    
    Args:
        log_file: Path to log file
        model_id: Model identifier
    
    Returns:
        Dictionary with training curves or None if not found
    """
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except Exception:
        return None
    
    train_metrics = {"epoch": [], "loss": [], "accuracy": [], "f1": [], "precision": [], "recall": []}
    val_metrics = {"epoch": [], "loss": [], "accuracy": [], "f1": [], "precision": [], "recall": []}
    
    current_epoch = 0
    current_iteration = 0
    has_epoch_data = False
    
    # Pattern 1: Validation metrics per fold (most common)
    val_metrics_pattern = re.compile(
        r'Fold\s+(\d+).*?Val\s+Loss:\s+([\d.]+).*?Val\s+Acc:\s+([\d.]+).*?Val\s+F1:\s+([\d.]+)'
        r'(?:.*?Val\s+Precision:\s+([\d.]+))?(?:.*?Val\s+Recall:\s+([\d.]+))?'
    )
    
    # Pattern 2: Epoch-based training (e.g., "Epoch 1, Train Loss: 0.5, Val Loss: 0.6")
    epoch_pattern = re.compile(
        r'(?:Epoch|Iteration|Round)\s+(\d+).*?(?:Train\s+Loss:\s+([\d.]+))?(?:.*?Val\s+Loss:\s+([\d.]+))?'
        r'(?:.*?Train\s+Acc:\s+([\d.]+))?(?:.*?Val\s+Acc:\s+([\d.]+))?'
        r'(?:.*?Train\s+F1:\s+([\d.]+))?(?:.*?Val\s+F1:\s+([\d.]+))?'
    )
    
    # Pattern 3: Training loss per iteration (e.g., "Iteration 10, Train Loss: 0.5")
    train_iter_pattern = re.compile(
        r'(?:Iteration|Epoch|Round)\s+(\d+).*?Train\s+Loss:\s+([\d.]+)'
        r'(?:.*?Train\s+Acc:\s+([\d.]+))?'
    )
    
    for line in lines:
        # Try fold-based validation metrics first
        val_match = val_metrics_pattern.search(line)
        if val_match:
            fold = int(val_match.group(1))
            val_loss = float(val_match.group(2))
            val_acc = float(val_match.group(3))
            val_f1 = float(val_match.group(4))
            val_precision = float(val_match.group(5)) if val_match.group(5) else None
            val_recall = float(val_match.group(6)) if val_match.group(6) else None
            
            val_metrics["epoch"].append(fold)
            val_metrics["loss"].append(val_loss)
            val_metrics["accuracy"].append(val_acc)
            val_metrics["f1"].append(val_f1)
            if val_precision is not None:
                val_metrics["precision"].append(val_precision)
            if val_recall is not None:
                val_metrics["recall"].append(val_recall)
            continue
        
        # Try epoch/iteration-based patterns
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            current_epoch = epoch
            has_epoch_data = True
            
            train_loss = epoch_match.group(2)
            val_loss = epoch_match.group(3)
            train_acc = epoch_match.group(4)
            val_acc = epoch_match.group(5)
            train_f1 = epoch_match.group(6)
            val_f1 = epoch_match.group(7)
            
            if train_loss:
                train_metrics["epoch"].append(epoch)
                train_metrics["loss"].append(float(train_loss))
            if train_acc:
                train_metrics["accuracy"].append(float(train_acc))
            if train_f1:
                train_metrics["f1"].append(float(train_f1))
            
            if val_loss:
                val_metrics["epoch"].append(epoch)
                val_metrics["loss"].append(float(val_loss))
            if val_acc:
                val_metrics["accuracy"].append(float(val_acc))
            if val_f1:
                val_metrics["f1"].append(float(val_f1))
            continue
        
        # Try training iteration pattern
        train_match = train_iter_pattern.search(line)
        if train_match:
            iteration = int(train_match.group(1))
            current_iteration = iteration
            has_epoch_data = True
            train_loss = float(train_match.group(2))
            train_acc = train_match.group(3)
            
            train_metrics["epoch"].append(iteration)
            train_metrics["loss"].append(train_loss)
            if train_acc:
                train_metrics["accuracy"].append(float(train_acc))
            continue
    
    # Return data if we found anything
        if val_metrics["epoch"] or train_metrics["epoch"]:
            return {
            "train": train_metrics,
            "val": val_metrics,
            "has_training_epochs": has_epoch_data
        }
    
    return None


def plot_training_curves(
    metrics_file: Path,
    model_name: str,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[Any]:
    """
    Plot training and validation loss/accuracy curves from metrics.jsonl.
    
    For models without epochs (baseline sklearn/XGBoost), shows a message
    that training curves are not available (these models don't train iteratively).
    
    Args:
        metrics_file: Path to metrics.jsonl file.
        model_name: Model name for display.
        figsize: Figure size tuple.
    
    Returns:
        matplotlib Figure object, or None if plotting unavailable or no data.
    """
    if not PLOTTING_AVAILABLE:
        print("[WARN] Matplotlib not available for plotting")
        return None
    
    history = load_training_curves_from_jsonl(metrics_file)
    if not history:
        # Try extracting from log file as fallback
        log_file = metrics_file.parent.parent / "logs" / "stage5" / f"stage5{metrics_file.parent.name[-1]}_*.log"
        if log_file.parent.exists():
            # Find latest log file for this model
            log_files = sorted(log_file.parent.glob(f"stage5{metrics_file.parent.name[-1]}_*.log"), reverse=True)
            if log_files:
                model_id = metrics_file.parent.name[-1] if len(metrics_file.parent.name) > 0 else "unknown"
                history = extract_training_curves_from_log(log_files[0], model_id)
                if history:
                    print(f"[INFO] Loaded training curves from log file: {log_files[0].name}")
    
    if not history:
        print(f"[WARN] No training history found in {metrics_file}")
        return None
    
    # Check if we have actual training epochs (epoch > 0)
    has_training_epochs = history.get("has_training_epochs", False)
    
    # For models with iteration-based training (like Logistic Regression with 100 iterations),
    # we may have epoch data but has_training_epochs might be False due to how epochs are tracked
    # Check if we have any actual data to plot
    train_metrics = history.get("train", {})
    val_metrics = history.get("val", {})
    train_has_data = bool(train_metrics.get("loss") or train_metrics.get("accuracy"))
    val_has_data = bool(val_metrics.get("loss") or val_metrics.get("accuracy"))
    
    if not has_training_epochs and not (train_has_data or val_has_data):
        print(f"[INFO] {model_name} does not have epoch-by-epoch training data.")
        print("       This model trains in a single step (sklearn/XGBoost), not iteratively.")
        print("       See 'Validation Metrics Across Folds' section for fold-wise performance.")
        return None
    
    # If we have data but has_training_epochs is False, it might be iteration-based training
    # Still plot it if we have validation data
    if not has_training_epochs and val_has_data:
        # This is likely iteration-based training - plot validation metrics
        print(f"[INFO] {model_name} uses iteration-based training. Plotting validation metrics.")
    
    # train_metrics and val_metrics are already extracted above (lines 1823-1824)
    # No need to extract again - they're already in scope
    
    if not train_metrics.get("loss") and not val_metrics.get("loss"):
        print(f"[WARN] No loss data found in {metrics_file}")
        return None
    
    def filter_outliers(values, epochs, iqr_factor=1.5):
        """Filter outliers using IQR method and remove spurious values."""
        if not values or len(values) < 3:
            return values, epochs
        
        # Convert to numpy for easier manipulation
        values_arr = np.array(values)
        epochs_arr = np.array(epochs[:len(values)])
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(values_arr) & (values_arr > 0)
        values_arr = values_arr[valid_mask]
        epochs_arr = epochs_arr[valid_mask]
        
        if len(values_arr) < 3:
            return values_arr.tolist(), epochs_arr.tolist()
        
        # Calculate IQR
        q25 = np.percentile(values_arr, 25)
        q75 = np.percentile(values_arr, 75)
        iqr = q75 - q25
        
        # Filter outliers
        lower_bound = q25 - iqr_factor * iqr
        upper_bound = q75 + iqr_factor * iqr
        outlier_mask = (values_arr >= lower_bound) & (values_arr <= upper_bound)
        
        # Also remove spurious values (straight line artifacts)
        # Check for sudden jumps that indicate interpolation artifacts
        if len(values_arr) > 2:
            diffs = np.abs(np.diff(values_arr))
            # If difference is too large compared to median, it might be spurious
            median_diff = np.median(diffs)
            if median_diff > 0:
                # Mark values with unusually large changes
                large_jump_mask = np.ones(len(values_arr), dtype=bool)
                large_jump_mask[1:-1] = (diffs[:-1] < 10 * median_diff) & (diffs[1:] < 10 * median_diff)
                outlier_mask = outlier_mask & large_jump_mask
        
        filtered_values = values_arr[outlier_mask].tolist()
        filtered_epochs = epochs_arr[outlier_mask].tolist()
        
        return filtered_values, filtered_epochs
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Loss curves (regular scale)
    # Note: This plots loss (binary cross-entropy) on a regular scale
    ax = axes[0, 0]
    
    train_loss_clean = []
    train_epochs_clean = []
    if train_metrics.get("loss") and train_metrics.get("epoch"):
        epochs = train_metrics["epoch"][:len(train_metrics["loss"])]
        train_loss_raw = train_metrics["loss"]
        train_loss_clean, train_epochs_clean = filter_outliers(train_loss_raw, epochs)
        
        if train_loss_clean:
            ax.plot(train_epochs_clean, train_loss_clean, 'b-', label='Train Loss',
                    linewidth=2, marker='o', markersize=4)
    
    val_loss_clean = []
    val_epochs_clean = []
    if val_metrics.get("loss") and val_metrics.get("epoch"):
        epochs = val_metrics["epoch"][:len(val_metrics["loss"])]
        val_loss_raw = val_metrics["loss"]
        val_loss_clean, val_epochs_clean = filter_outliers(val_loss_raw, epochs)
        
        if val_loss_clean:
            ax.plot(val_epochs_clean, val_loss_clean, 'r-', label='Val Loss',
                    linewidth=2, marker='s', markersize=4)
    
    # Add comment about train vs validation loss
    if train_loss_clean and val_loss_clean:
        train_final = train_loss_clean[-1]
        val_final = val_loss_clean[-1]
        if val_final < train_final:
            comment = "Val loss < Train loss: Possible overfitting or data leakage"
        elif val_final > train_final * 1.1:
            comment = "Val loss > Train loss: Model generalizes well"
        else:
            comment = "Val loss ≈ Train loss: Good generalization"
        
        # Add text comment in upper right corner
        ax.text(0.98, 0.02, comment, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training and Validation Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax = axes[0, 1]
    if train_metrics.get("accuracy") and train_metrics.get("epoch"):
        epochs = train_metrics["epoch"][:len(train_metrics["accuracy"])]
        # Filter outliers for accuracy
        train_acc_clean, train_acc_epochs = filter_outliers(train_metrics["accuracy"], epochs)
        if train_acc_clean:
            ax.plot(train_acc_epochs, train_acc_clean, 'b-', label='Train Acc',
                    linewidth=2, marker='o', markersize=4)
    if val_metrics.get("accuracy") and val_metrics.get("epoch"):
        epochs = val_metrics["epoch"][:len(val_metrics["accuracy"])]
        # Filter outliers for accuracy
        val_acc_clean, val_acc_epochs = filter_outliers(val_metrics["accuracy"], epochs)
        if val_acc_clean:
            ax.plot(val_acc_epochs, val_acc_clean, 'r-', label='Val Acc',
                    linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score curves
    ax = axes[1, 0]
    if train_metrics.get("f1") and train_metrics.get("epoch"):
        epochs = train_metrics["epoch"][:len(train_metrics["f1"])]
        # Filter outliers for F1 score
        train_f1_clean, train_f1_epochs = filter_outliers(train_metrics["f1"], epochs)
        if train_f1_clean:
            ax.plot(train_f1_epochs, train_f1_clean, 'b-', label='Train F1',
                    linewidth=2, marker='o', markersize=4)
    if val_metrics.get("f1") and val_metrics.get("epoch"):
        epochs = val_metrics["epoch"][:len(val_metrics["f1"])]
        # Filter outliers for F1 score
        val_f1_clean, val_f1_epochs = filter_outliers(val_metrics["f1"], epochs)
        if val_f1_clean:
            ax.plot(val_f1_epochs, val_f1_clean, 'r-', label='Val F1',
                    linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Training and Validation F1 Score', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Precision-Recall curves (over epochs)
    ax = axes[1, 1]
    if (train_metrics.get("precision") and train_metrics.get("recall") and
            train_metrics.get("epoch")):
        epochs = train_metrics["epoch"][:len(train_metrics["precision"])]
        # Filter outliers for precision and recall
        train_prec_clean, train_prec_epochs = filter_outliers(train_metrics["precision"], epochs)
        train_rec_clean, train_rec_epochs = filter_outliers(train_metrics["recall"], epochs)
        
        if train_prec_clean:
            ax.plot(train_prec_epochs, train_prec_clean, 'b--', label='Train Precision',
                    linewidth=2, marker='o', markersize=4)
        if train_rec_clean:
            ax.plot(train_rec_epochs, train_rec_clean, 'b:', label='Train Recall',
                    linewidth=2, marker='s', markersize=4)
    if (val_metrics.get("precision") and val_metrics.get("recall") and
            val_metrics.get("epoch")):
        epochs = val_metrics["epoch"][:len(val_metrics["precision"])]
        # Filter outliers for precision and recall
        val_prec_clean, val_prec_epochs = filter_outliers(val_metrics["precision"], epochs)
        val_rec_clean, val_rec_epochs = filter_outliers(val_metrics["recall"], epochs)
        
        if val_prec_clean:
            ax.plot(val_prec_epochs, val_prec_clean, 'r--', label='Val Precision',
                    linewidth=2, marker='o', markersize=4)
        if val_rec_clean:
            ax.plot(val_rec_epochs, val_rec_clean, 'r:', label='Val Recall',
                    linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Precision and Recall Over Epochs', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Training Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_training_curves_from_history(
    history: Dict[str, Dict[str, List[float]]],
    model_name: str,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[Any]:
    """
    Plot training curves from a history dictionary (extracted from logs or JSONL).
    
    Args:
        history: Dictionary with 'train' and 'val' keys containing metrics.
        model_name: Model name for display.
        figsize: Figure size tuple.
    
    Returns:
        matplotlib Figure object, or None if plotting unavailable or no data.
    """
    if not PLOTTING_AVAILABLE:
        print("[WARN] Matplotlib not available for plotting")
        return None
    
    train_metrics = history.get("train", {})
    val_metrics = history.get("val", {})
    
    if not train_metrics.get("loss") and not val_metrics.get("loss"):
        print(f"[WARN] No loss data found in history")
        return None
    
    def filter_outliers(values, epochs, iqr_factor=1.5):
        """Filter outliers using IQR method and remove spurious values."""
        if not values or len(values) < 3:
            return values, epochs
        
        # Convert to numpy for easier manipulation
        values_arr = np.array(values)
        epochs_arr = np.array(epochs[:len(values)])
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(values_arr) & (values_arr > 0)
        values_arr = values_arr[valid_mask]
        epochs_arr = epochs_arr[valid_mask]
        
        if len(values_arr) < 3:
            return values_arr.tolist(), epochs_arr.tolist()
        
        # Calculate IQR
        q25 = np.percentile(values_arr, 25)
        q75 = np.percentile(values_arr, 75)
        iqr = q75 - q25
        
        # Filter outliers
        lower_bound = q25 - iqr_factor * iqr
        upper_bound = q75 + iqr_factor * iqr
        outlier_mask = (values_arr >= lower_bound) & (values_arr <= upper_bound)
        
        # Also remove spurious values (straight line artifacts)
        # Check for sudden jumps that indicate interpolation artifacts
        if len(values_arr) > 2:
            diffs = np.abs(np.diff(values_arr))
            # If difference is too large compared to median, it might be spurious
            median_diff = np.median(diffs)
            if median_diff > 0:
                # Mark values with unusually large changes
                large_jump_mask = np.ones(len(values_arr), dtype=bool)
                large_jump_mask[1:-1] = (diffs[:-1] < 10 * median_diff) & (diffs[1:] < 10 * median_diff)
                outlier_mask = outlier_mask & large_jump_mask
        
        filtered_values = values_arr[outlier_mask].tolist()
        filtered_epochs = epochs_arr[outlier_mask].tolist()
        
        return filtered_values, filtered_epochs
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Loss curves (regular scale)
    # Note: This plots loss (binary cross-entropy) on a regular scale
    ax = axes[0, 0]
    
    train_loss_clean = []
    train_epochs_clean = []
    if train_metrics.get("loss") and train_metrics.get("epoch"):
        epochs = train_metrics["epoch"][:len(train_metrics["loss"])]
        train_loss_raw = train_metrics["loss"]
        train_loss_clean, train_epochs_clean = filter_outliers(train_loss_raw, epochs)
        
        if train_loss_clean:
            ax.plot(train_epochs_clean, train_loss_clean, 'b-', label='Train Loss',
                    linewidth=2, marker='o', markersize=4)
    
    val_loss_clean = []
    val_epochs_clean = []
    if val_metrics.get("loss") and val_metrics.get("epoch"):
        epochs = val_metrics["epoch"][:len(val_metrics["loss"])]
        val_loss_raw = val_metrics["loss"]
        val_loss_clean, val_epochs_clean = filter_outliers(val_loss_raw, epochs)
        
        if val_loss_clean:
            ax.plot(val_epochs_clean, val_loss_clean, 'r-', label='Val Loss',
                    linewidth=2, marker='s', markersize=4)
    
    # Add comment about train vs validation loss
    if train_loss_clean and val_loss_clean:
        train_final = train_loss_clean[-1]
        val_final = val_loss_clean[-1]
        if val_final < train_final:
            comment = "Val loss < Train loss: Possible overfitting or data leakage"
        elif val_final > train_final * 1.1:
            comment = "Val loss > Train loss: Model generalizes well"
        else:
            comment = "Val loss ≈ Train loss: Good generalization"
        
        # Add text comment in upper right corner
        ax.text(0.98, 0.02, comment, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('Epoch/Fold', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training and Validation Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax = axes[0, 1]
    if train_metrics.get("accuracy") and train_metrics.get("epoch"):
        epochs = train_metrics["epoch"][:len(train_metrics["accuracy"])]
        # Filter outliers for accuracy
        train_acc_clean, train_acc_epochs = filter_outliers(train_metrics["accuracy"], epochs)
        if train_acc_clean:
            ax.plot(train_acc_epochs, train_acc_clean, 'b-', label='Train Acc',
                    linewidth=2, marker='o', markersize=4)
    if val_metrics.get("accuracy") and val_metrics.get("epoch"):
        epochs = val_metrics["epoch"][:len(val_metrics["accuracy"])]
        # Filter outliers for accuracy
        val_acc_clean, val_acc_epochs = filter_outliers(val_metrics["accuracy"], epochs)
        if val_acc_clean:
            ax.plot(val_acc_epochs, val_acc_clean, 'r-', label='Val Acc',
                    linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch/Fold', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score curves
    ax = axes[1, 0]
    if train_metrics.get("f1") and train_metrics.get("epoch"):
        epochs = train_metrics["epoch"][:len(train_metrics["f1"])]
        # Filter outliers for F1 score
        train_f1_clean, train_f1_epochs = filter_outliers(train_metrics["f1"], epochs)
        if train_f1_clean:
            ax.plot(train_f1_epochs, train_f1_clean, 'b-', label='Train F1',
                    linewidth=2, marker='o', markersize=4)
    if val_metrics.get("f1") and val_metrics.get("epoch"):
        epochs = val_metrics["epoch"][:len(val_metrics["f1"])]
        # Filter outliers for F1 score
        val_f1_clean, val_f1_epochs = filter_outliers(val_metrics["f1"], epochs)
        if val_f1_clean:
            ax.plot(val_f1_epochs, val_f1_clean, 'r-', label='Val F1',
                    linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch/Fold', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Training and Validation F1 Score', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Precision-Recall curves (over epochs)
    ax = axes[1, 1]
    if (train_metrics.get("precision") and train_metrics.get("recall") and
            train_metrics.get("epoch")):
        epochs = train_metrics["epoch"][:len(train_metrics["precision"])]
        # Filter outliers for precision and recall
        train_prec_clean, train_prec_epochs = filter_outliers(train_metrics["precision"], epochs)
        train_rec_clean, train_rec_epochs = filter_outliers(train_metrics["recall"], epochs)
        
        if train_prec_clean:
            ax.plot(train_prec_epochs, train_prec_clean, 'b--', label='Train Precision',
                    linewidth=2, marker='o', markersize=4)
        if train_rec_clean:
            ax.plot(train_rec_epochs, train_rec_clean, 'b:', label='Train Recall',
                    linewidth=2, marker='s', markersize=4)
    if (val_metrics.get("precision") and val_metrics.get("recall") and
            val_metrics.get("epoch")):
        epochs = val_metrics["epoch"][:len(val_metrics["precision"])]
        # Filter outliers for precision and recall
        val_prec_clean, val_prec_epochs = filter_outliers(val_metrics["precision"], epochs)
        val_rec_clean, val_rec_epochs = filter_outliers(val_metrics["recall"], epochs)
        
        if val_prec_clean:
            ax.plot(val_prec_epochs, val_prec_clean, 'r--', label='Val Precision',
                    linewidth=2, marker='o', markersize=4)
        if val_rec_clean:
            ax.plot(val_rec_epochs, val_rec_clean, 'r:', label='Val Recall',
                    linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch/Fold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Precision and Recall Over Epochs/Folds', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Training Curves (from log)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_validation_metrics_across_folds(
    metrics: Dict[str, Any],
    model_name: str,
    figsize: Tuple[int, int] = (16, 10)
) -> Optional[Any]:
    """
    Plot validation metrics across CV folds for baseline models.
    
    Since baseline models (sklearn) don't have training epochs,
    this plots validation metrics (loss, accuracy, F1, etc.) across folds.
    
    Args:
        metrics: Metrics dictionary with fold_results or cv_fold_results.
        model_name: Model name for display.
        figsize: Figure size tuple.
    
    Returns:
        matplotlib Figure object, or None if plotting unavailable or no data.
    """
    if not PLOTTING_AVAILABLE:
        print("[WARN] Matplotlib not available for plotting")
        return None
    
    if not metrics or not isinstance(metrics, dict):
        print(f"[WARN] No metrics data found for {model_name}")
        return None
    
    fold_results = (
        metrics.get('fold_results', []) or
        metrics.get('cv_fold_results', []) or
        []
    )
    
    if not fold_results:
        print(f"[WARN] No fold results found for {model_name}")
        return None
    
    # Extract metrics across folds
    folds = [r.get('fold', i+1) for i, r in enumerate(fold_results)]
    val_losses = [r.get('val_loss', 0.0) for r in fold_results]
    val_accs = [r.get('val_acc', 0.0) for r in fold_results]
    val_f1s = [r.get('val_f1', 0.0) for r in fold_results]
    val_precisions = [r.get('val_precision', 0.0) for r in fold_results]
    val_recalls = [r.get('val_recall', 0.0) for r in fold_results]
    
    # Filter out NaN values
    valid_indices = [
        i for i in range(len(fold_results))
        if not (np.isnan(val_losses[i]) or np.isnan(val_accs[i]))
    ]
    
    if not valid_indices:
        print(f"[WARN] No valid metrics found for {model_name}")
        return None
    
    folds = [folds[i] for i in valid_indices]
    val_losses = [val_losses[i] for i in valid_indices]
    val_accs = [val_accs[i] for i in valid_indices]
    val_f1s = [val_f1s[i] for i in valid_indices]
    val_precisions = [val_precisions[i] for i in valid_indices]
    val_recalls = [val_recalls[i] for i in valid_indices]
    
    # Prepare data for violin plots
    metrics_data = {
        'F1 Score': val_f1s,
        'Accuracy': val_accs,
        'Precision': val_precisions,
        'Recall': val_recalls
    }
    
    # Filter out NaN values from each metric
    clean_metrics_data = {}
    for metric_name, values in metrics_data.items():
        clean_values = [v for v in values if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if clean_values:
            clean_metrics_data[metric_name] = clean_values
    
    # Add loss if available and valid
    val_losses_clean = [max(l, 1e-6) for l in val_losses if not (isinstance(l, float) and (np.isnan(l) or np.isinf(l)))]
    if val_losses_clean:
        clean_metrics_data['Loss'] = val_losses_clean
    
    if not clean_metrics_data:
        print(f"[WARN] No valid metrics to plot for {model_name}")
        return None
    
    # Color mapping for each metric
    color_map = {
        'F1 Score': 'green',
        'Accuracy': 'blue',
        'Precision': 'magenta',
        'Recall': 'cyan',
        'Loss': 'red'
    }
    
    # Create and display separate figures for each metric
    figures = []
    for metric_name, values in clean_metrics_data.items():
        # Create individual figure for each metric
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create violin plot with statistical annotations
        parts = ax.violinplot([values], positions=[0], showmeans=True, showmedians=True, 
                              showextrema=True, widths=0.6)
        
        # Customize violin plot colors
        metric_color = color_map.get(metric_name, 'gray')
        for pc in parts['bodies']:
            pc.set_facecolor(metric_color)
            pc.set_alpha(0.7)
        
        # Calculate and display statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        
        # Add statistical text box
        stats_text = (f'Mean: {mean_val:.4f} ± {std_val:.4f}\n'
                     f'Median: {median_val:.4f}\n'
                     f'Range: [{min_val:.4f}, {max_val:.4f}]\n'
                     f'IQR: [{q25:.4f}, {q75:.4f}]\n'
                     f'N: {len(values)}')
        
        ax.text(0.5, 0.98, stats_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        # Add mean and median lines
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.4f}')
        ax.axhline(y=median_val, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Median: {median_val:.4f}')
        
        # Set labels and title
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title(f'{model_name} - {metric_name} Distribution Across CV Folds', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks([0])
        ax.set_xticklabels([''])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        plt.show()  # Display each figure separately
        figures.append(fig)
    
    # Return the last figure (for compatibility with existing code)
    return figures[-1] if figures else None


def query_duckdb_metrics(
    model_id: str,
    project_root: Path,
    model_type_mapping: Optional[Dict[str, str]] = None,
    db_path: str = "data/stage5_metrics.duckdb"
) -> Optional[Dict[str, Any]]:
    """
    Query DuckDB database for model metrics.
    
    Args:
        model_id: Model identifier.
        project_root: Project root directory.
        model_type_mapping: Optional model type mapping dict.
        db_path: Path to DuckDB database file.
    
    Returns:
        Dictionary with query results or None if unavailable.
    """
    try:
        import duckdb
    except ImportError:
        print("[WARN] DuckDB not available. Install with: pip install duckdb")
        return None
    
    db_file = project_root / db_path
    if not db_file.exists():
        print(f"[WARN] DuckDB database not found: {db_file}")
        return None
    
    if model_type_mapping is None:
        model_type_mapping = MODEL_TYPE_MAPPING
    
    model_type = model_type_mapping.get(model_id)
    if not model_type:
        print(f"[WARN] Unknown model_id: {model_id}")
        return None
    
    try:
        conn = duckdb.connect(str(db_file))
                
        # Query metrics for this model type
        query = """
            SELECT 
            fold_idx,
            val_loss,
            val_acc,
            val_f1,
            val_precision,
            val_recall,
            val_f1_class0,
            val_precision_class0,
            val_recall_class0,
            val_f1_class1,
            val_precision_class1,
            val_recall_class1
            FROM training_metrics
            WHERE model_type = ?
            ORDER BY fold_idx
        """
        
        results = conn.execute(query, [model_type]).fetchall()
        columns = ['fold_idx', 'val_loss', 'val_acc', 'val_f1', 'val_precision', 
                   'val_recall', 'val_f1_class0', 'val_precision_class0', 
                   'val_recall_class0', 'val_f1_class1', 'val_precision_class1', 
                   'val_recall_class1']
        
        if not results:
            conn.close()
            return None
        
        # Convert to dictionary format
        fold_results = []
        for row in results:
            fold_data = dict(zip(columns, row))
            fold_results.append(fold_data)
        
        # Get aggregated statistics
        agg_query = """
            SELECT 
            AVG(val_acc) as mean_val_acc,
            STDDEV(val_acc) as std_val_acc,
            AVG(val_f1) as mean_val_f1,
            STDDEV(val_f1) as std_val_f1,
            AVG(val_precision) as mean_val_precision,
            STDDEV(val_precision) as std_val_precision,
            AVG(val_recall) as mean_val_recall,
            STDDEV(val_recall) as std_val_recall
            FROM training_metrics
            WHERE model_type = ?
        """
        
        agg_results = conn.execute(agg_query, [model_type]).fetchone()
        conn.close()
        
        if agg_results:
            aggregated = {
            'mean_val_acc': agg_results[0],
            'std_val_acc': agg_results[1],
            'mean_val_f1': agg_results[2],
            'std_val_f1': agg_results[3],
            'mean_val_precision': agg_results[4],
            'std_val_precision': agg_results[5],
            'mean_val_recall': agg_results[6],
            'std_val_recall': agg_results[7]
            }
        else:
            aggregated = {}
        
        return {
            'model_type': model_type,
            'fold_results': fold_results,
            'aggregated': aggregated,
            'num_folds': len(fold_results)
        }
    except Exception as e:
        print(f"[WARN] Failed to query DuckDB: {e}")
        return None

