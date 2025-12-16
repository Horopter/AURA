#!/usr/bin/env python3
"""
Generate Comprehensive Report for All Stage 5 Models

This script generates:
1. Graphs and plots for all models
2. Time logs from SLURM and Python logs
3. Metrics comparison across models
4. Performance analysis
5. Connection verification (MLflow, DuckDB, Airflow)

Usage:
    python generate_stage5_comprehensive_report.py [--output-dir OUTPUT_DIR]
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

try:
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logging.warning("Plotting libraries not available. Install with: pip install matplotlib seaborn")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# All 21 Stage 5 models
STAGE5_MODELS = [
    ("5a", "logistic_regression"),
    ("5b", "svm"),
    ("5c", "naive_cnn"),
    ("5d", "pretrained_inception"),
    ("5e", "variable_ar_cnn"),
    ("5f", "xgboost_pretrained_inception"),
    ("5g", "xgboost_i3d"),
    ("5h", "xgboost_r2plus1d"),
    ("5i", "xgboost_vit_gru"),
    ("5j", "xgboost_vit_transformer"),
    ("5k", "vit_gru"),
    ("5l", "vit_transformer"),
    ("5m", "timesformer"),
    ("5n", "vivit"),
    ("5o", "i3d"),
    ("5p", "r2plus1d"),
    ("5q", "x3d"),
    ("5r", "slowfast"),
    ("5s", "slowfast_attention"),
    ("5t", "slowfast_multiscale"),
    ("5u", "two_stream"),
]


def parse_time_string(time_str: str) -> Optional[float]:
    """Parse time string to seconds."""
    if not time_str:
        return None
    
    # Extract seconds
    seconds_match = re.search(r'(\d+)s', time_str)
    if seconds_match:
        return float(seconds_match.group(1))
    
    # Extract minutes
    minutes_match = re.search(r'(\d+)\s*minutes?', time_str)
    if minutes_match:
        return float(minutes_match.group(1)) * 60
    
    # Extract hours
    hours_match = re.search(r'(\d+)\s*hours?', time_str)
    if hours_match:
        return float(hours_match.group(1)) * 3600
    
    return None


def extract_slurm_times(model_type: str) -> List[Dict]:
    """Extract execution times from SLURM log files."""
    times = []
    logs_dir = project_root / "logs" / "stage5"
    
    if not logs_dir.exists():
        return times
    
    # Pattern: stage5{letter}-{JOB_ID}.out
    pattern = re.compile(rf'stage5\w+-(\d+)\.out')
    
    for log_file in logs_dir.glob(f"stage5*.out"):
        if model_type not in log_file.name:
            continue
        
        try:
            content = log_file.read_text()
            
            # Look for execution time
            time_match = re.search(
                r'Execution time:\s*(\d+)s\s*\((\d+)\s*minutes?\)',
                content
            )
            if time_match:
                seconds = int(time_match.group(1))
                job_id = pattern.search(log_file.name)
                times.append({
                    "model_type": model_type,
                    "job_id": job_id.group(1) if job_id else None,
                    "duration_seconds": seconds,
                    "duration_minutes": seconds / 60,
                    "source": "slurm",
                    "log_file": str(log_file.relative_to(project_root))
                })
        except Exception as e:
            logger.warning(f"Error reading SLURM log {log_file}: {e}")
    
    return times


def extract_python_times(model_type: str) -> List[Dict]:
    """Extract execution times from Python training logs."""
    times = []
    logs_dir = project_root / "logs"
    
    if not logs_dir.exists():
        return times
    
    # Look for stage5_training_*.log files
    for log_file in logs_dir.glob("stage5_training_*.log"):
        try:
            content = log_file.read_text()
            
            # Check if this log is for our model type
            if model_type not in content:
                continue
            
            # Look for execution time
            time_match = re.search(
                r'Execution time:\s*([\d.]+)\s*seconds\s*\(([\d.]+)\s*minutes?\)',
                content
            )
            if time_match:
                seconds = float(time_match.group(1))
                times.append({
                    "model_type": model_type,
                    "duration_seconds": seconds,
                    "duration_minutes": seconds / 60,
                    "source": "python",
                    "log_file": str(log_file.relative_to(project_root))
                })
        except Exception as e:
            logger.warning(f"Error reading Python log {log_file}: {e}")
    
    return times


def load_model_metrics(model_type: str) -> Optional[Dict]:
    """Load aggregated metrics for a model."""
    metrics_file = project_root / "data" / "stage5" / model_type / "metrics.json"
    
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading metrics for {model_type}: {e}")
        return None


def collect_all_metrics() -> Dict:
    """Collect metrics for all models."""
    all_metrics = {}
    
    for model_id, model_type in STAGE5_MODELS:
        metrics = load_model_metrics(model_type)
        if metrics:
            all_metrics[model_type] = metrics
    
    return all_metrics


def collect_all_times() -> List[Dict]:
    """Collect time logs for all models."""
    all_times = []
    
    for model_id, model_type in STAGE5_MODELS:
        slurm_times = extract_slurm_times(model_type)
        python_times = extract_python_times(model_type)
        all_times.extend(slurm_times)
        all_times.extend(python_times)
    
    return all_times


def generate_metrics_comparison_plot(metrics: Dict, output_path: Path):
    """Generate metrics comparison plot across all models."""
    if not HAS_PLOTTING:
        logger.warning("Plotting not available, skipping metrics comparison plot")
        return
    
    # Prepare data
    model_names = []
    val_accs = []
    val_f1s = []
    val_precisions = []
    val_recalls = []
    
    for model_type, model_metrics in metrics.items():
        if "mean_val_acc" in model_metrics:
            model_names.append(model_type)
            val_accs.append(model_metrics.get("mean_val_acc", 0))
            val_f1s.append(model_metrics.get("mean_val_f1", 0))
            val_precisions.append(model_metrics.get("mean_val_precision", 0))
            val_recalls.append(model_metrics.get("mean_val_recall", 0))
    
    if not model_names:
        logger.warning("No metrics data available for plotting")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stage 5 Models: Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy
    ax = axes[0, 0]
    ax.barh(model_names, val_accs, color='steelblue')
    ax.set_xlabel('Validation Accuracy')
    ax.set_title('Mean Validation Accuracy')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    # F1 Score
    ax = axes[0, 1]
    ax.barh(model_names, val_f1s, color='forestgreen')
    ax.set_xlabel('Validation F1 Score')
    ax.set_title('Mean Validation F1 Score')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    # Precision
    ax = axes[1, 0]
    ax.barh(model_names, val_precisions, color='coral')
    ax.set_xlabel('Validation Precision')
    ax.set_title('Mean Validation Precision')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    # Recall
    ax = axes[1, 1]
    ax.barh(model_names, val_recalls, color='mediumpurple')
    ax.set_xlabel('Validation Recall')
    ax.set_title('Mean Validation Recall')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics comparison plot: {output_path}")


def generate_time_analysis_plot(times: List[Dict], output_path: Path):
    """Generate time analysis plot."""
    if not HAS_PLOTTING:
        logger.warning("Plotting not available, skipping time analysis plot")
        return
    
    if not times:
        logger.warning("No time data available for plotting")
        return
    
    # Prepare data
    df = pd.DataFrame(times)
    
    # Group by model type
    model_times = df.groupby('model_type')['duration_minutes'].agg(['mean', 'std', 'count'])
    model_times = model_times.sort_values('mean', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(model_times))
    ax.barh(y_pos, model_times['mean'], xerr=model_times['std'], 
            color='steelblue', alpha=0.7, capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_times.index)
    ax.set_xlabel('Execution Time (minutes)', fontsize=12)
    ax.set_title('Stage 5 Models: Execution Time Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, (idx, row) in enumerate(model_times.iterrows()):
        ax.text(row['mean'] + row['std'] + 5, i, f"n={int(row['count'])}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved time analysis plot: {output_path}")


def generate_connection_diagram(output_path: Path):
    """Generate a diagram showing all connections."""
    if not HAS_PLOTTING:
        logger.warning("Plotting not available, skipping connection diagram")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Create a simple text-based diagram
    diagram_text = """
    STAGE 5 END-TO-END DATA FLOW
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SLURM SCRIPTS (21 models)                    │
    │  slurm_stage5a.sh → slurm_stage5b.sh → ... → slurm_stage5u.sh  │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
                                │ Environment variables
                                │ Log files: logs/stage5/
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              Python Training Script                              │
    │         src/scripts/run_stage5_training.py                       │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
                                │ Function call
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              Training Pipeline                                   │
    │         lib/training/pipeline.py                                │
    │         stage5_train_models()                                    │
    └───────────┬───────────────────────────────┬─────────────────────┘
                │                                 │
                │                                 │
        ┌───────┴───────┐              ┌─────────┴─────────┐
        │               │              │                   │
        ▼               ▼              ▼                   ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐      ┌─────────────┐
    │Experiment│   │  MLflow │   │  DuckDB │      │   Plots    │
    │ Tracker  │   │ Tracker │   │Analytics │      │ Generation │
    │(JSONL)  │   │(mlruns/)│   │(SQL)     │      │(PNG files) │
    └─────────┘   └─────────┘   └─────────┘      └─────────────┘
        │               │              │                   │
        │               │              │                   │
        └───────┬───────┴──────────────┴───────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Output Files                                  │
    │  data/stage5/{model_type}/fold_{N}/metrics.jsonl                 │
    │  data/stage5/{model_type}/metrics.json                           │
    │  data/stage5/{model_type}/plots/*.png                            │
    └─────────────────────────────────────────────────────────────────┘
                │
                │
                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Airflow DAG                                   │
    │         airflow/dags/fvc_pipeline_dag.py                         │
    │         stage5_training() task                                   │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    ax.text(0.5, 0.5, diagram_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved connection diagram: {output_path}")


def generate_time_log_report(times: List[Dict], output_path: Path):
    """Generate time log report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("STAGE 5 MODELS: TIME LOGS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if not times:
        report_lines.append("No time logs found.")
        report_lines.append("")
    else:
        # Group by model type
        by_model = defaultdict(list)
        for time_entry in times:
            by_model[time_entry['model_type']].append(time_entry)
        
        for model_type in sorted(by_model.keys()):
            model_times = by_model[model_type]
            report_lines.append(f"Model: {model_type}")
            report_lines.append("-" * 80)
            
            for time_entry in model_times:
                source = time_entry.get('source', 'unknown')
                duration_min = time_entry.get('duration_minutes', 0)
                duration_sec = time_entry.get('duration_seconds', 0)
                log_file = time_entry.get('log_file', 'N/A')
                
                report_lines.append(f"  Source: {source}")
                report_lines.append(f"  Duration: {duration_min:.2f} minutes ({duration_sec:.0f} seconds)")
                report_lines.append(f"  Log file: {log_file}")
                report_lines.append("")
            
            # Summary
            avg_time = np.mean([t['duration_minutes'] for t in model_times])
            report_lines.append(f"  Average: {avg_time:.2f} minutes")
            report_lines.append("")
    
    report_text = "\n".join(report_lines)
    output_path.write_text(report_text)
    logger.info(f"Saved time log report: {output_path}")


def generate_metrics_report(metrics: Dict, output_path: Path):
    """Generate metrics report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("STAGE 5 MODELS: METRICS SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if not metrics:
        report_lines.append("No metrics found.")
        report_lines.append("")
    else:
        # Create table
        report_lines.append(f"{'Model':<30} {'Val Acc':<12} {'Val F1':<12} {'Val Prec':<12} {'Val Rec':<12}")
        report_lines.append("-" * 80)
        
        for model_type in sorted(metrics.keys()):
            m = metrics[model_type]
            val_acc = m.get('mean_val_acc', 0)
            val_f1 = m.get('mean_val_f1', 0)
            val_prec = m.get('mean_val_precision', 0)
            val_rec = m.get('mean_val_recall', 0)
            
            report_lines.append(
                f"{model_type:<30} {val_acc:<12.4f} {val_f1:<12.4f} {val_prec:<12.4f} {val_rec:<12.4f}"
            )
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    output_path.write_text(report_text)
    logger.info(f"Saved metrics report: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive report for all Stage 5 models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stage5_analysis",
        help="Output directory for reports and plots (default: data/stage5_analysis)"
    )
    
    args = parser.parse_args()
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Generating Comprehensive Stage 5 Report")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Collect data
    logger.info("1. Collecting metrics...")
    metrics = collect_all_metrics()
    logger.info(f"   Found metrics for {len(metrics)} models")
    
    logger.info("2. Collecting time logs...")
    times = collect_all_times()
    logger.info(f"   Found {len(times)} time log entries")
    
    # Generate reports
    logger.info("3. Generating reports...")
    generate_metrics_report(metrics, output_dir / "metrics_report.txt")
    generate_time_log_report(times, output_dir / "time_logs_report.txt")
    
    # Generate plots
    if HAS_PLOTTING:
        logger.info("4. Generating plots...")
        generate_metrics_comparison_plot(metrics, output_dir / "metrics_comparison.png")
        generate_time_analysis_plot(times, output_dir / "time_analysis.png")
        generate_connection_diagram(output_dir / "connection_diagram.png")
    else:
        logger.warning("4. Skipping plots (plotting libraries not available)")
    
    # Save data as JSON
    logger.info("5. Saving data as JSON...")
    with open(output_dir / "metrics_data.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    with open(output_dir / "times_data.json", 'w') as f:
        json.dump(times, f, indent=2, default=str)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Report generation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  - metrics_report.txt")
    logger.info(f"  - time_logs_report.txt")
    if HAS_PLOTTING:
        logger.info(f"  - metrics_comparison.png")
        logger.info(f"  - time_analysis.png")
        logger.info(f"  - connection_diagram.png")
    logger.info(f"  - metrics_data.json")
    logger.info(f"  - times_data.json")


if __name__ == "__main__":
    main()

