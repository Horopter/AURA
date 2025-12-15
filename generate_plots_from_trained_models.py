#!/usr/bin/env python3
"""
Generate comprehensive plots from already trained models.

This script leverages existing architecture and code to:
1. Load training history from metrics.jsonl files
2. Load trained models and generate predictions
3. Generate training/validation loss curves
4. Generate ROC/PR curves
5. Generate other diagnostic plots

Usage:
    python generate_plots_from_trained_models.py [--model-type MODEL_TYPE] [--output-dir OUTPUT_DIR]
    
    Or use the wrapper script to suppress macOS objc warnings:
    ./generate_plots.sh [--model-type MODEL_TYPE] [--output-dir OUTPUT_DIR]

Note: On macOS, you may see warnings about duplicate class implementations in libavdevice.
These are harmless and occur when both PyAV's bundled libraries and Homebrew's ffmpeg are present.
Use the wrapper script (generate_plots.sh) to suppress these warnings.
"""

import sys
import os
from pathlib import Path

# Suppress duplicate library warnings from PyAV/ffmpeg on macOS
# This occurs when both PyAV's bundled libraries and Homebrew's ffmpeg are present
# These warnings are harmless but annoying. We suppress them by filtering stderr.
if sys.platform == 'darwin':  # macOS
    import io
    
    class ObjCWarningFilter:
        """Filter to suppress objc duplicate class warnings on macOS."""
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
        
        def write(self, text):
            # Filter out objc duplicate class warnings
            if 'objc[' in text and 'is implemented in both' in text:
                return  # Suppress this warning
            self.original_stderr.write(text)
        
        def flush(self):
            self.original_stderr.flush()
        
        def __getattr__(self, name):
            return getattr(self.original_stderr, name)
    
    # Apply filter to suppress warnings
    sys.stderr = ObjCWarningFilter(sys.stderr)

import json
import logging
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, 
    average_precision_score, confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from lib.data import stratified_kfold
from lib.utils.paths import load_metadata_flexible
from lib.training.model_factory import create_model, get_model_config, is_pytorch_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Constants
N_SPLITS = 5
RANDOM_STATE = 42
STAGE5_DIR = project_root / "data" / "stage5"


def load_training_history(metrics_file: Path) -> Optional[Dict]:
    """Load training history from metrics.jsonl file."""
    if not metrics_file.exists():
        return None
    
    try:
        train_metrics = {"epoch": [], "loss": [], "accuracy": [], "f1": []}
        val_metrics = {"epoch": [], "loss": [], "accuracy": [], "f1": []}
        
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
                        if metric == "loss":
                            train_metrics["loss"].append(value)
                            train_metrics["epoch"].append(epoch)
                        elif metric == "accuracy":
                            train_metrics["accuracy"].append(value)
                        elif metric == "f1":
                            train_metrics["f1"].append(value)
                    elif phase == "val":
                        if metric == "loss":
                            val_metrics["loss"].append(value)
                            val_metrics["epoch"].append(epoch)
                        elif metric == "accuracy":
                            val_metrics["accuracy"].append(value)
                        elif metric == "f1":
                            val_metrics["f1"].append(value)
                except json.JSONDecodeError:
                    continue
        
        # Ensure epochs are aligned
        if train_metrics["epoch"]:
            train_metrics["epoch"] = sorted(set(train_metrics["epoch"]))
        if val_metrics["epoch"]:
            val_metrics["epoch"] = sorted(set(val_metrics["epoch"]))
        
        return {
            "train": train_metrics,
            "val": val_metrics
        }
    except Exception as e:
        logger.debug(f"Error loading training history from {metrics_file}: {e}")
        return None


def load_model_from_fold(model_type: str, fold_dir: Path, project_root_str: str):
    """Load model from fold directory."""
    pytorch_models = {
        "naive_cnn", "pretrained_inception", "variable_ar_cnn", "vit_gru", 
        "vit_transformer", "timesformer", "vivit", "i3d", "r2plus1d", 
        "x3d", "slowfast", "slowfast_attention", "slowfast_multiscale", "two_stream"
    }
    
    if model_type in pytorch_models:
        # PyTorch models require VideoDataset - skip for now
        return None, None
    
    try:
        xgb_file = fold_dir / "xgboost_model.json"
        joblib_file = fold_dir / "model.joblib"
        
        # XGBoost models
        if xgb_file.exists() or model_type.startswith("xgboost_"):
            model = create_model(model_type, get_model_config(model_type))
            model.load(str(fold_dir))
            return model, None
        
        # Sklearn models
        if joblib_file.exists() or model_type in {"logistic_regression", "svm", "sklearn_logreg"}:
            import joblib
            if joblib_file.exists():
                model = joblib.load(joblib_file)
            else:
                model = create_model(model_type, get_model_config(model_type))
                if hasattr(model, 'load'):
                    model.load(str(fold_dir))
            scaler_file = fold_dir / "scaler.joblib"
            scaler = joblib.load(scaler_file) if scaler_file.exists() else None
            return model, scaler
        
        return None, None
    except Exception as e:
        logger.debug(f"Error loading model from {fold_dir}: {e}")
        return None, None


def get_predictions_from_model(model, model_type: str, val_df: pl.DataFrame, project_root_str: str):
    """Get predictions from loaded model."""
    try:
        if is_pytorch_model(model_type):
            logger.debug(f"PyTorch model {model_type} requires VideoDataset, skipping")
            return None, None
        
        if hasattr(model, 'predict'):
            y_probs = model.predict(val_df, project_root=project_root_str)
            if y_probs.ndim == 2:
                y_probs = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs.flatten()
            else:
                y_probs = y_probs.flatten()
            
            labels = val_df["label"].to_list()
            label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
            y_true = np.array([label_map[label] for label in labels])
            
            y_probs = np.clip(y_probs, 0.0, 1.0)
            
            return y_true, y_probs
        else:
            logger.debug(f"Model {model_type} does not have predict method")
            return None, None
    except Exception as e:
        logger.debug(f"Error getting predictions: {e}")
        return None, None


def plot_training_curves(history: Dict, output_path: Path, model_name: str):
    """Plot training and validation loss/accuracy curves."""
    train_metrics = history.get("train", {})
    val_metrics = history.get("val", {})
    
    if not train_metrics.get("epoch") and not val_metrics.get("epoch"):
        logger.warning(f"No training history available for {model_name}")
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training Loss
    if train_metrics.get("loss") and train_metrics.get("epoch"):
        epochs = train_metrics["epoch"][:len(train_metrics["loss"])]
        axes[0, 0].plot(epochs, train_metrics["loss"], label='Train Loss', linewidth=2, marker='o', markersize=4)
    if val_metrics.get("loss") and val_metrics.get("epoch"):
        epochs = val_metrics["epoch"][:len(val_metrics["loss"])]
        axes[0, 0].plot(epochs, val_metrics["loss"], label='Val Loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training Accuracy
    if train_metrics.get("accuracy"):
        epochs = train_metrics["epoch"][:len(train_metrics["accuracy"])] if train_metrics.get("epoch") else range(len(train_metrics["accuracy"]))
        axes[0, 1].plot(epochs, train_metrics["accuracy"], label='Train Acc', linewidth=2, marker='o', markersize=4)
    if val_metrics.get("accuracy"):
        epochs = val_metrics["epoch"][:len(val_metrics["accuracy"])] if val_metrics.get("epoch") else range(len(val_metrics["accuracy"]))
        axes[0, 1].plot(epochs, val_metrics["accuracy"], label='Val Acc', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Training F1
    if train_metrics.get("f1"):
        epochs = train_metrics["epoch"][:len(train_metrics["f1"])] if train_metrics.get("epoch") else range(len(train_metrics["f1"]))
        axes[1, 0].plot(epochs, train_metrics["f1"], label='Train F1', linewidth=2, marker='o', markersize=4)
    if val_metrics.get("f1"):
        epochs = val_metrics["epoch"][:len(val_metrics["f1"])] if val_metrics.get("epoch") else range(len(val_metrics["f1"]))
        axes[1, 0].plot(epochs, val_metrics["f1"], label='Val F1', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score', fontweight='bold')
    axes[1, 0].set_title('Training and Validation F1 Score', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # Loss comparison (zoomed)
    if train_metrics.get("loss") and val_metrics.get("loss"):
        train_epochs = train_metrics["epoch"][:len(train_metrics["loss"])] if train_metrics.get("epoch") else range(len(train_metrics["loss"]))
        val_epochs = val_metrics["epoch"][:len(val_metrics["loss"])] if val_metrics.get("epoch") else range(len(val_metrics["loss"]))
        axes[1, 1].plot(train_epochs, train_metrics["loss"], label='Train Loss', linewidth=2, alpha=0.7)
        axes[1, 1].plot(val_epochs, val_metrics["loss"], label='Val Loss', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('Loss', fontweight='bold')
        axes[1, 1].set_title('Loss Comparison (Overlay)', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name.replace("_", " ").title()} - Training Curves', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated training curves: {output_path}")
    return True


def plot_roc_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, output_path: Path, model_name: str):
    """Generate ROC and PR curves."""
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})', linewidth=2, color='#1f77b4')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curve', fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    
    # PR Curve
    ax2.plot(recall, precision, label=f'PR (AP={ap:.4f})', linewidth=2, color='#ff7f0e')
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    
    plt.suptitle(f'{model_name.replace("_", " ").title()} - ROC and PR Curves', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated ROC/PR curves: {output_path} (AUC={auc:.4f}, AP={ap:.4f})")
    return auc, ap


def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, model_name: str):
    """Generate confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'{model_name.replace("_", " ").title()} - Confusion Matrix', 
                 fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated confusion matrix: {output_path}")


def plot_prediction_distribution(y_true: np.ndarray, y_probs: np.ndarray, output_path: Path, model_name: str):
    """Plot distribution of predictions for each class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of predictions by class
    real_probs = y_probs[y_true == 0]
    fake_probs = y_probs[y_true == 1]
    
    axes[0].hist(real_probs, bins=50, alpha=0.7, label='Real', color='blue', edgecolor='black')
    axes[0].hist(fake_probs, bins=50, alpha=0.7, label='Fake', color='red', edgecolor='black')
    axes[0].set_xlabel('Predicted Probability', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Prediction Distribution by Class', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    
    # Box plot
    data_to_plot = [real_probs, fake_probs]
    bp = axes[1].boxplot(data_to_plot, labels=['Real', 'Fake'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    axes[1].set_ylabel('Predicted Probability', fontweight='bold')
    axes[1].set_title('Prediction Distribution (Box Plot)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])
    
    plt.suptitle(f'{model_name.replace("_", " ").title()} - Prediction Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated prediction distribution: {output_path}")


def process_model(model_type: str, output_dir: Path):
    """Process a single model and generate all plots."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing model: {model_type}")
    logger.info(f"{'='*60}")
    
    model_dir = STAGE5_DIR / model_type
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return False
    
    # Create output directory for this model
    model_output_dir = output_dir / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata for CV splits
    metadata_paths = [
        project_root / "data" / "features_stage4" / "features_scaled_metadata.arrow",
        project_root / "data" / "features_stage4" / "features_scaled_metadata.parquet",
        project_root / "data" / "scaled_videos" / "scaled_metadata.arrow",
        project_root / "data" / "scaled_videos" / "scaled_metadata.parquet",
        project_root / "data" / "features_stage2" / "features_metadata.arrow",
    ]
    
    metadata_df = None
    for path in metadata_paths:
        if path.exists():
            metadata_df = load_metadata_flexible(str(path))
            if metadata_df is not None:
                logger.info(f"Loaded metadata from: {path.name}")
                break
    
    if metadata_df is None:
        logger.warning(f"Could not load metadata for {model_type}")
        return False
    
    # Get CV folds
    cv_folds = stratified_kfold(metadata_df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    
    # Find fold directories
    fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        logger.warning(f"No fold directories found for {model_type}")
        return False
    
    logger.info(f"Found {len(fold_dirs)} fold directories")
    
    # Collect training histories
    all_train_histories = []
    all_y_true = []
    all_y_probs = []
    
    project_root_str = str(project_root)
    
    for fold_idx, (train_df, val_df) in enumerate(cv_folds):
        if fold_idx >= len(fold_dirs):
            break
        
        fold_dir = fold_dirs[fold_idx]
        
        # Load training history
        metrics_file = fold_dir / "metrics.jsonl"
        history = load_training_history(metrics_file)
        if history:
            all_train_histories.append(history)
        
        # Load model and get predictions
        model, scaler = load_model_from_fold(model_type, fold_dir, project_root_str)
        if model is not None:
            y_true, y_probs = get_predictions_from_model(model, model_type, val_df, project_root_str)
            if y_true is not None and y_probs is not None:
                all_y_true.append(y_true)
                all_y_probs.append(y_probs)
                logger.info(f"  Fold {fold_idx + 1}: Collected {len(y_true)} predictions")
    
    # Generate plots
    
    # 1. Training curves (if available)
    if all_train_histories:
        # Aggregate histories across folds
        aggregated_history = {
            "train": {"epoch": [], "loss": [], "accuracy": [], "f1": []},
            "val": {"epoch": [], "loss": [], "accuracy": [], "f1": []}
        }
        
        # For simplicity, use the first fold's history (or average if needed)
        if all_train_histories:
            aggregated_history = all_train_histories[0]
        
        plot_training_curves(aggregated_history, 
                            model_output_dir / "training_curves.png", 
                            model_type)
    
    # 2. ROC/PR curves (if predictions available)
    if all_y_true and all_y_probs:
        all_y_true_flat = np.concatenate(all_y_true)
        all_y_probs_flat = np.concatenate(all_y_probs)
        
        auc, ap = plot_roc_pr_curves(all_y_true_flat, all_y_probs_flat,
                                     model_output_dir / "roc_pr_curves.png",
                                     model_type)
        
        # 3. Confusion matrix
        y_pred = (all_y_probs_flat > 0.5).astype(int)
        plot_confusion_matrix_heatmap(all_y_true_flat, y_pred,
                                     model_output_dir / "confusion_matrix.png",
                                     model_type)
        
        # 4. Prediction distribution
        plot_prediction_distribution(all_y_true_flat, all_y_probs_flat,
                                    model_output_dir / "prediction_distribution.png",
                                    model_type)
        
        # 5. Save metrics summary
        metrics_summary = {
            "model_type": model_type,
            "n_samples": len(all_y_true_flat),
            "auc": float(auc),
            "ap": float(ap),
            "accuracy": float((y_pred == all_y_true_flat).mean()),
            "mean_prob_real": float(all_y_probs_flat[all_y_true_flat == 0].mean()),
            "mean_prob_fake": float(all_y_probs_flat[all_y_true_flat == 1].mean()),
        }
        
        with open(model_output_dir / "metrics_summary.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"✓ Generated all plots for {model_type}")
        logger.info(f"  AUC: {auc:.4f}, AP: {ap:.4f}")
        return True
    else:
        logger.warning(f"Could not generate prediction-based plots for {model_type}")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots from trained models")
    parser.add_argument("--model-type", type=str, default=None,
                       help="Specific model type to process (default: all)")
    parser.add_argument("--output-dir", type=str, default="data/stage5/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Generate Plots from Trained Models")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    
    # Find all model directories
    if args.model_type:
        model_types = [args.model_type]
    else:
        model_types = [d.name for d in STAGE5_DIR.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
    
    logger.info(f"Found {len(model_types)} model types: {', '.join(model_types)}")
    
    success_count = 0
    for model_type in model_types:
        try:
            if process_model(model_type, output_dir):
                success_count += 1
        except Exception as e:
            logger.error(f"Error processing {model_type}: {e}", exc_info=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Complete! Processed {success_count}/{len(model_types)} models")
    logger.info(f"Plots saved to: {output_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

