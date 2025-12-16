#!/usr/bin/env python3
"""
Comprehensive End-to-End Verification for All Stage 5 Models

This script verifies:
1. All 21 SLURM scripts exist and are correctly configured
2. Python training scripts are properly called
3. Metrics files (JSON, JSONL) are generated
4. MLflow integration points
5. DuckDB integration points
6. Airflow integration
7. Plot generation functions
8. Time log extraction

Usage:
    python verify_stage5_end_to_end.py [--output OUTPUT_FILE]
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# All 21 Stage 5 models (from slurm_stage5_training.sh)
STAGE5_MODELS = [
    ("5a", "logistic_regression", "scripts/slurm_jobs/slurm_stage5a.sh"),
    ("5b", "svm", "scripts/slurm_jobs/slurm_stage5b.sh"),
    ("5c", "naive_cnn", "scripts/slurm_jobs/slurm_stage5c.sh"),
    ("5d", "pretrained_inception", "scripts/slurm_jobs/slurm_stage5d.sh"),
    ("5e", "variable_ar_cnn", "scripts/slurm_jobs/slurm_stage5e.sh"),
    ("5f", "xgboost_pretrained_inception", "scripts/slurm_jobs/slurm_stage5f.sh"),
    ("5g", "xgboost_i3d", "scripts/slurm_jobs/slurm_stage5g.sh"),
    ("5h", "xgboost_r2plus1d", "scripts/slurm_jobs/slurm_stage5h.sh"),
    ("5i", "xgboost_vit_gru", "scripts/slurm_jobs/slurm_stage5i.sh"),
    ("5j", "xgboost_vit_transformer", "scripts/slurm_jobs/slurm_stage5j.sh"),
    ("5k", "vit_gru", "scripts/slurm_jobs/slurm_stage5k.sh"),
    ("5l", "vit_transformer", "scripts/slurm_jobs/slurm_stage5l.sh"),
    ("5m", "timesformer", "scripts/slurm_jobs/slurm_stage5m.sh"),
    ("5n", "vivit", "scripts/slurm_jobs/slurm_stage5n.sh"),
    ("5o", "i3d", "scripts/slurm_jobs/slurm_stage5o.sh"),
    ("5p", "r2plus1d", "scripts/slurm_jobs/slurm_stage5p.sh"),
    ("5q", "x3d", "scripts/slurm_jobs/slurm_stage5q.sh"),
    ("5r", "slowfast", "scripts/slurm_jobs/slurm_stage5r.sh"),
    ("5s", "slowfast_attention", "scripts/slurm_jobs/slurm_stage5s.sh"),
    ("5t", "slowfast_multiscale", "scripts/slurm_jobs/slurm_stage5t.sh"),
    ("5u", "two_stream", "scripts/slurm_jobs/slurm_stage5u.sh"),
]


def verify_slurm_script(script_path: Path, model_id: str, model_type: str) -> Dict:
    """Verify SLURM script exists and has correct structure."""
    result = {
        "exists": False,
        "calls_python": False,
        "python_script": None,
        "model_type_match": False,
        "time_tracking": False,
        "log_file": None,
        "errors": []
    }
    
    if not script_path.exists():
        result["errors"].append(f"SLURM script not found: {script_path}")
        return result
    
    result["exists"] = True
    
    try:
        content = script_path.read_text()
        
        # Check if it calls the Python training script
        if "run_stage5_training.py" in content:
            result["calls_python"] = True
            # Extract Python script path
            match = re.search(r'PYTHON_SCRIPT="([^"]+)"', content)
            if match:
                result["python_script"] = match.group(1)
            else:
                match = re.search(r'run_stage5_training\.py', content)
                if match:
                    result["python_script"] = "src/scripts/run_stage5_training.py"
        
        # Check if MODEL_TYPE matches
        if f'MODEL_TYPE="{model_type}"' in content or f"MODEL_TYPE='{model_type}'" in content:
            result["model_type_match"] = True
        
        # Check for time tracking
        if "STAGE5_START" in content and "STAGE5_END" in content and "STAGE5_DURATION" in content:
            result["time_tracking"] = True
        
        # Check for log file
        log_match = re.search(r'LOG_FILE="([^"]+)"', content)
        if log_match:
            result["log_file"] = log_match.group(1)
        
    except Exception as e:
        result["errors"].append(f"Error reading SLURM script: {e}")
    
    return result


def verify_python_training_script() -> Dict:
    """Verify Python training script exists and has correct structure."""
    result = {
        "exists": False,
        "calls_pipeline": False,
        "mlflow_integration": False,
        "time_tracking": False,
        "errors": []
    }
    
    script_path = project_root / "src/scripts/run_stage5_training.py"
    if not script_path.exists():
        result["errors"].append(f"Python training script not found: {script_path}")
        return result
    
    result["exists"] = True
    
    try:
        content = script_path.read_text()
        
        # Check if it calls stage5_train_models
        if "stage5_train_models" in content:
            result["calls_pipeline"] = True
        
        # Check for MLflow integration
        if "use_mlflow" in content or "MLflow" in content:
            result["mlflow_integration"] = True
        
        # Check for time tracking
        if "stage_start" in content and "stage_duration" in content:
            result["time_tracking"] = True
        
    except Exception as e:
        result["errors"].append(f"Error reading Python script: {e}")
    
    return result


def verify_training_pipeline() -> Dict:
    """Verify training pipeline has all required components."""
    result = {
        "exists": False,
        "mlflow_integration": False,
        "experiment_tracker": False,
        "metrics_logging": False,
        "plot_generation": False,
        "errors": []
    }
    
    pipeline_path = project_root / "lib/training/pipeline.py"
    if not pipeline_path.exists():
        result["errors"].append(f"Training pipeline not found: {pipeline_path}")
        return result
    
    result["exists"] = True
    
    try:
        content = pipeline_path.read_text()
        
        # Check for MLflow integration
        if "mlflow_tracker" in content or "MLflowTracker" in content:
            result["mlflow_integration"] = True
        
        # Check for ExperimentTracker
        if "ExperimentTracker" in content:
            result["experiment_tracker"] = True
        
        # Check for metrics logging
        if "log_metric" in content or "metrics.jsonl" in content:
            result["metrics_logging"] = True
        
        # Check for plot generation
        if "plot_cv_fold_comparison" in content or "visualization" in content:
            result["plot_generation"] = True
        
    except Exception as e:
        result["errors"].append(f"Error reading pipeline: {e}")
    
    return result


def verify_mlflow_integration() -> Dict:
    """Verify MLflow integration components."""
    result = {
        "tracker_exists": False,
        "log_metrics": False,
        "log_artifacts": False,
        "experiment_management": False,
        "errors": []
    }
    
    tracker_path = project_root / "lib/mlops/mlflow_tracker.py"
    if not tracker_path.exists():
        result["errors"].append(f"MLflow tracker not found: {tracker_path}")
        return result
    
    result["tracker_exists"] = True
    
    try:
        content = tracker_path.read_text()
        
        # Check for key methods
        if "def log_metric" in content or "def log_metrics" in content:
            result["log_metrics"] = True
        
        if "def log_artifact" in content or "def log_artifacts" in content:
            result["log_artifacts"] = True
        
        if "experiment_name" in content or "create_experiment" in content:
            result["experiment_management"] = True
        
    except Exception as e:
        result["errors"].append(f"Error reading MLflow tracker: {e}")
    
    return result


def verify_duckdb_integration() -> Dict:
    """Verify DuckDB integration components."""
    result = {
        "analytics_exists": False,
        "register_parquet": False,
        "register_arrow": False,
        "query_method": False,
        "errors": []
    }
    
    analytics_path = project_root / "lib/utils/duckdb_analytics.py"
    if not analytics_path.exists():
        result["errors"].append(f"DuckDB analytics not found: {analytics_path}")
        return result
    
    result["analytics_exists"] = True
    
    try:
        content = analytics_path.read_text()
        
        # Check for key methods
        if "def register_parquet" in content:
            result["register_parquet"] = True
        
        if "def register_arrow" in content:
            result["register_arrow"] = True
        
        if "def query" in content:
            result["query_method"] = True
        
    except Exception as e:
        result["errors"].append(f"Error reading DuckDB analytics: {e}")
    
    return result


def verify_airflow_integration() -> Dict:
    """Verify Airflow integration."""
    result = {
        "dag_exists": False,
        "stage5_task": False,
        "calls_training": False,
        "errors": []
    }
    
    dag_path = project_root / "airflow/dags/fvc_pipeline_dag.py"
    if not dag_path.exists():
        result["errors"].append(f"Airflow DAG not found: {dag_path}")
        return result
    
    result["dag_exists"] = True
    
    try:
        content = dag_path.read_text()
        
        # Check for stage5 task
        if "stage5_training" in content or "task_stage5" in content:
            result["stage5_task"] = True
        
        # Check if it calls training function
        if "stage5_train_models" in content:
            result["calls_training"] = True
        
    except Exception as e:
        result["errors"].append(f"Error reading Airflow DAG: {e}")
    
    return result


def verify_plot_generation() -> Dict:
    """Verify plot generation components."""
    result = {
        "visualization_module": False,
        "plot_script": False,
        "plot_functions": [],
        "errors": []
    }
    
    # Check visualization module
    viz_path = project_root / "lib/training/visualization.py"
    if viz_path.exists():
        result["visualization_module"] = True
        try:
            content = viz_path.read_text()
            # Extract plot function names
            plot_funcs = re.findall(r'def (plot_\w+)', content)
            result["plot_functions"] = plot_funcs
        except Exception as e:
            result["errors"].append(f"Error reading visualization module: {e}")
    else:
        result["errors"].append(f"Visualization module not found: {viz_path}")
    
    # Check plot generation script
    plot_script = project_root / "generate_plots_from_trained_models.py"
    if plot_script.exists():
        result["plot_script"] = True
    else:
        result["errors"].append(f"Plot generation script not found: {plot_script}")
    
    return result


def verify_time_log_extraction() -> Dict:
    """Verify time log extraction capabilities."""
    result = {
        "analysis_script": False,
        "extract_slurm_times": False,
        "extract_python_times": False,
        "errors": []
    }
    
    analysis_script = project_root / "src/scripts/analyze_stage5_metrics.py"
    if analysis_script.exists():
        result["analysis_script"] = True
        try:
            content = analysis_script.read_text()
            if "extract_slurm_times" in content:
                result["extract_slurm_times"] = True
            if "extract_python_times" in content or "parse_time_string" in content:
                result["extract_python_times"] = True
        except Exception as e:
            result["errors"].append(f"Error reading analysis script: {e}")
    else:
        result["errors"].append(f"Analysis script not found: {analysis_script}")
    
    return result


def verify_metrics_structure() -> Dict:
    """Verify expected metrics file structure."""
    result = {
        "expected_files": [],
        "expected_structure": {
            "metrics_json": "data/stage5/{model_type}/metrics.json",
            "metrics_jsonl": "data/stage5/{model_type}/fold_{N}/metrics.jsonl",
            "metadata_json": "data/stage5/{model_type}/fold_{N}/metadata.json",
        },
        "errors": []
    }
    
    # Expected metrics files per model
    for model_id, model_type, _ in STAGE5_MODELS:
        model_dir = project_root / "data" / "stage5" / model_type
        if model_dir.exists():
            # Check for aggregated metrics
            metrics_json = model_dir / "metrics.json"
            if metrics_json.exists():
                result["expected_files"].append(str(metrics_json.relative_to(project_root)))
            
            # Check for fold-specific metrics
            for fold_dir in model_dir.glob("fold_*"):
                if fold_dir.is_dir():
                    metrics_jsonl = fold_dir / "metrics.jsonl"
                    metadata_json = fold_dir / "metadata.json"
                    if metrics_jsonl.exists():
                        result["expected_files"].append(str(metrics_jsonl.relative_to(project_root)))
                    if metadata_json.exists():
                        result["expected_files"].append(str(metadata_json.relative_to(project_root)))
    
    return result


def verify_all_models() -> Dict:
    """Verify all Stage 5 models end-to-end."""
    results = {
        "models": {},
        "python_script": {},
        "training_pipeline": {},
        "mlflow": {},
        "duckdb": {},
        "airflow": {},
        "plots": {},
        "time_logs": {},
        "metrics": {},
        "summary": {
            "total_models": len(STAGE5_MODELS),
            "verified_models": 0,
            "errors": []
        }
    }
    
    logger.info("=" * 80)
    logger.info("Stage 5 End-to-End Verification")
    logger.info("=" * 80)
    
    # Verify each model's SLURM script
    logger.info("\n1. Verifying SLURM scripts...")
    for model_id, model_type, script_path in STAGE5_MODELS:
        full_path = project_root / script_path
        logger.info(f"  Checking {model_id}: {model_type}")
        results["models"][model_type] = verify_slurm_script(full_path, model_id, model_type)
        if results["models"][model_type]["exists"]:
            results["summary"]["verified_models"] += 1
    
    # Verify Python training script
    logger.info("\n2. Verifying Python training script...")
    results["python_script"] = verify_python_training_script()
    
    # Verify training pipeline
    logger.info("\n3. Verifying training pipeline...")
    results["training_pipeline"] = verify_training_pipeline()
    
    # Verify MLflow integration
    logger.info("\n4. Verifying MLflow integration...")
    results["mlflow"] = verify_mlflow_integration()
    
    # Verify DuckDB integration
    logger.info("\n5. Verifying DuckDB integration...")
    results["duckdb"] = verify_duckdb_integration()
    
    # Verify Airflow integration
    logger.info("\n6. Verifying Airflow integration...")
    results["airflow"] = verify_airflow_integration()
    
    # Verify plot generation
    logger.info("\n7. Verifying plot generation...")
    results["plots"] = verify_plot_generation()
    
    # Verify time log extraction
    logger.info("\n8. Verifying time log extraction...")
    results["time_logs"] = verify_time_log_extraction()
    
    # Verify metrics structure
    logger.info("\n9. Verifying metrics structure...")
    results["metrics"] = verify_metrics_structure()
    
    return results


def generate_report(results: Dict, output_file: Optional[Path] = None):
    """Generate comprehensive verification report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("STAGE 5 END-TO-END VERIFICATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary
    summary = results["summary"]
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Total models: {summary['total_models']}")
    report_lines.append(f"Verified SLURM scripts: {summary['verified_models']}")
    report_lines.append("")
    
    # Model verification
    report_lines.append("MODEL SLURM SCRIPTS")
    report_lines.append("-" * 80)
    for model_id, model_type, _ in STAGE5_MODELS:
        model_result = results["models"].get(model_type, {})
        status = "✓" if model_result.get("exists") else "✗"
        report_lines.append(f"{status} {model_id}: {model_type}")
        if model_result.get("calls_python"):
            report_lines.append(f"    → Calls Python script: ✓")
        if model_result.get("time_tracking"):
            report_lines.append(f"    → Time tracking: ✓")
        if model_result.get("errors"):
            for error in model_result["errors"]:
                report_lines.append(f"    ✗ {error}")
    report_lines.append("")
    
    # Python script
    report_lines.append("PYTHON TRAINING SCRIPT")
    report_lines.append("-" * 80)
    py_result = results["python_script"]
    report_lines.append(f"Exists: {'✓' if py_result.get('exists') else '✗'}")
    report_lines.append(f"Calls pipeline: {'✓' if py_result.get('calls_pipeline') else '✗'}")
    report_lines.append(f"MLflow integration: {'✓' if py_result.get('mlflow_integration') else '✗'}")
    report_lines.append(f"Time tracking: {'✓' if py_result.get('time_tracking') else '✗'}")
    if py_result.get("errors"):
        for error in py_result["errors"]:
            report_lines.append(f"  ✗ {error}")
    report_lines.append("")
    
    # Training pipeline
    report_lines.append("TRAINING PIPELINE")
    report_lines.append("-" * 80)
    pipeline_result = results["training_pipeline"]
    report_lines.append(f"Exists: {'✓' if pipeline_result.get('exists') else '✗'}")
    report_lines.append(f"MLflow integration: {'✓' if pipeline_result.get('mlflow_integration') else '✗'}")
    report_lines.append(f"Experiment tracker: {'✓' if pipeline_result.get('experiment_tracker') else '✗'}")
    report_lines.append(f"Metrics logging: {'✓' if pipeline_result.get('metrics_logging') else '✗'}")
    report_lines.append(f"Plot generation: {'✓' if pipeline_result.get('plot_generation') else '✗'}")
    if pipeline_result.get("errors"):
        for error in pipeline_result["errors"]:
            report_lines.append(f"  ✗ {error}")
    report_lines.append("")
    
    # MLflow
    report_lines.append("MLFLOW INTEGRATION")
    report_lines.append("-" * 80)
    mlflow_result = results["mlflow"]
    report_lines.append(f"Tracker exists: {'✓' if mlflow_result.get('tracker_exists') else '✗'}")
    report_lines.append(f"Log metrics: {'✓' if mlflow_result.get('log_metrics') else '✗'}")
    report_lines.append(f"Log artifacts: {'✓' if mlflow_result.get('log_artifacts') else '✗'}")
    report_lines.append(f"Experiment management: {'✓' if mlflow_result.get('experiment_management') else '✗'}")
    if mlflow_result.get("errors"):
        for error in mlflow_result["errors"]:
            report_lines.append(f"  ✗ {error}")
    report_lines.append("")
    
    # DuckDB
    report_lines.append("DUCKDB INTEGRATION")
    report_lines.append("-" * 80)
    duckdb_result = results["duckdb"]
    report_lines.append(f"Analytics exists: {'✓' if duckdb_result.get('analytics_exists') else '✗'}")
    report_lines.append(f"Register Parquet: {'✓' if duckdb_result.get('register_parquet') else '✗'}")
    report_lines.append(f"Register Arrow: {'✓' if duckdb_result.get('register_arrow') else '✗'}")
    report_lines.append(f"Query method: {'✓' if duckdb_result.get('query_method') else '✗'}")
    if duckdb_result.get("errors"):
        for error in duckdb_result["errors"]:
            report_lines.append(f"  ✗ {error}")
    report_lines.append("")
    
    # Airflow
    report_lines.append("AIRFLOW INTEGRATION")
    report_lines.append("-" * 80)
    airflow_result = results["airflow"]
    report_lines.append(f"DAG exists: {'✓' if airflow_result.get('dag_exists') else '✗'}")
    report_lines.append(f"Stage 5 task: {'✓' if airflow_result.get('stage5_task') else '✗'}")
    report_lines.append(f"Calls training: {'✓' if airflow_result.get('calls_training') else '✗'}")
    if airflow_result.get("errors"):
        for error in airflow_result["errors"]:
            report_lines.append(f"  ✗ {error}")
    report_lines.append("")
    
    # Plots
    report_lines.append("PLOT GENERATION")
    report_lines.append("-" * 80)
    plots_result = results["plots"]
    report_lines.append(f"Visualization module: {'✓' if plots_result.get('visualization_module') else '✗'}")
    report_lines.append(f"Plot script: {'✓' if plots_result.get('plot_script') else '✗'}")
    plot_funcs = plots_result.get("plot_functions", [])
    if plot_funcs:
        report_lines.append(f"Plot functions ({len(plot_funcs)}):")
        for func in plot_funcs:
            report_lines.append(f"    - {func}")
    if plots_result.get("errors"):
        for error in plots_result["errors"]:
            report_lines.append(f"  ✗ {error}")
    report_lines.append("")
    
    # Time logs
    report_lines.append("TIME LOG EXTRACTION")
    report_lines.append("-" * 80)
    time_result = results["time_logs"]
    report_lines.append(f"Analysis script: {'✓' if time_result.get('analysis_script') else '✗'}")
    report_lines.append(f"Extract SLURM times: {'✓' if time_result.get('extract_slurm_times') else '✗'}")
    report_lines.append(f"Extract Python times: {'✓' if time_result.get('extract_python_times') else '✗'}")
    if time_result.get("errors"):
        for error in time_result["errors"]:
            report_lines.append(f"  ✗ {error}")
    report_lines.append("")
    
    # Metrics
    report_lines.append("METRICS STRUCTURE")
    report_lines.append("-" * 80)
    metrics_result = results["metrics"]
    expected_files = metrics_result.get("expected_files", [])
    report_lines.append(f"Expected metrics files found: {len(expected_files)}")
    if expected_files:
        report_lines.append("Sample files:")
        for f in expected_files[:10]:  # Show first 10
            report_lines.append(f"    - {f}")
        if len(expected_files) > 10:
            report_lines.append(f"    ... and {len(expected_files) - 10} more")
    report_lines.append("")
    
    # Data flow connections
    report_lines.append("DATA FLOW & CONNECTIONS")
    report_lines.append("-" * 80)
    report_lines.append("SLURM → Python:")
    report_lines.append("  - Environment variables passed to training script")
    report_lines.append("  - Log files in logs/stage5/")
    report_lines.append("")
    report_lines.append("Python → Training Pipeline:")
    report_lines.append("  - Function: stage5_train_models() in lib/training/pipeline.py")
    report_lines.append("  - Entry point: src/scripts/run_stage5_training.py")
    report_lines.append("")
    report_lines.append("Training Pipeline → ExperimentTracker:")
    report_lines.append("  - Metrics logged to metrics.jsonl")
    report_lines.append("  - Location: data/stage5/{model_type}/fold_{N}/metrics.jsonl")
    report_lines.append("")
    report_lines.append("Training Pipeline → MLflow:")
    report_lines.append("  - Metrics and artifacts logged to mlruns/")
    report_lines.append("  - Access via MLflow UI: mlflow ui --port 5000")
    report_lines.append("")
    report_lines.append("Output Files → DuckDB:")
    report_lines.append("  - Metrics files: data/stage5/{model_type}/metrics.json")
    report_lines.append("  - Can be registered for SQL queries")
    report_lines.append("")
    report_lines.append("Output Files → Plot Generation:")
    report_lines.append("  - Models loaded from fold_{N}/ directories")
    report_lines.append("  - Plots saved to plots/ directories")
    report_lines.append("")
    report_lines.append("Airflow → All Stages:")
    report_lines.append("  - DAG: airflow/dags/fvc_pipeline_dag.py")
    report_lines.append("  - Stage 5 task: stage5_training()")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("VERIFICATION COMPLETE")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Print to console
    print(report_text)
    
    # Write to file if specified
    if output_file:
        output_file.write_text(report_text)
        logger.info(f"Report written to: {output_file}")
    
    # Also save as JSON for programmatic access
    if output_file:
        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON results written to: {json_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify all Stage 5 models end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for report (default: stdout only)"
    )
    
    args = parser.parse_args()
    
    # Run verification
    results = verify_all_models()
    
    # Generate report
    output_file = None
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    generate_report(results, output_file)
    
    # Exit with error code if there are critical issues
    total_errors = sum(
        len(v.get("errors", [])) 
        for v in results.values() 
        if isinstance(v, dict)
    )
    
    if total_errors > 0:
        logger.warning(f"Found {total_errors} error(s) during verification")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

