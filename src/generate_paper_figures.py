"""
Generate Publication-Ready Figures for IEEE Paper

This script generates high-quality figures suitable for academic paper submission.
All figures are exported in multiple formats (PNG, PDF, SVG) with proper styling.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
})


def load_training_results(results_dir: Path) -> Dict:
    """Load all training results."""
    results = {}
    
    if not results_dir.exists():
        return results
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_type = model_dir.name
        fold_results_path = model_dir / "fold_results.csv"
        
        if fold_results_path.exists():
            try:
                df = pl.read_csv(fold_results_path)
                results[model_type] = {
                    "fold_results": df,
                    "has_fold_results": True
                }
            except Exception:
                pass
    
    return results


def plot_model_comparison_paper(results: Dict, output_dir: Path):
    """Generate publication-ready model comparison figure."""
    model_names = []
    accuracies = []
    stds = []
    conf_intervals = []
    
    for model_type, model_data in results.items():
        if not model_data.get("has_fold_results", False):
            continue
        
        df = model_data["fold_results"]
        acc_cols = [col for col in df.columns if "acc" in col.lower() and col != "fold"]
        
        if acc_cols:
            values = df[acc_cols[0]].to_list()
            acc_mean = float(np.mean(values))
            acc_std = float(np.std(values))
            n = len(values)
            
            ci = stats.t.interval(0.95, n-1, loc=acc_mean, scale=stats.sem(values))
            conf_intervals.append([acc_mean - ci[0], ci[1] - acc_mean])
            
            model_names.append(model_type.replace("_", " ").title())
            accuracies.append(acc_mean)
            stds.append(acc_std)
    
    if not model_names:
        print("No model results found for comparison plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(model_names))
    colors = sns.color_palette("husl", len(model_names))
    
    bars = ax.bar(x_pos, accuracies, yerr=[ci[1] for ci in conf_intervals],
                  capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, acc, std) in enumerate(zip(bars, accuracies, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + conf_intervals[i][1] + 0.01,
                f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontweight='bold')
    ax.set_title('Model Performance Comparison with 95% Confidence Intervals', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax.legend()
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'figure_model_comparison.{fmt}', format=fmt, dpi=300)
    
    plt.close()
    print(f"✓ Generated model comparison figure")


def plot_kfold_distribution_paper(results: Dict, output_dir: Path):
    """Generate K-fold distribution box plot."""
    all_data = []
    
    for model_type, model_data in results.items():
        if not model_data.get("has_fold_results", False):
            continue
        
        df = model_data["fold_results"]
        acc_cols = [col for col in df.columns if "acc" in col.lower() and col != "fold"]
        
        if acc_cols:
            values = df[acc_cols[0]].to_list()
            model_name = model_type.replace("_", " ").title()
            
            for val in values:
                all_data.append({
                    "Model": model_name,
                    "Accuracy": float(val)
                })
    
    if not all_data:
        print("No data for K-fold distribution plot.")
        return
    
    df_plot = pd.DataFrame(all_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df_plot["Model"].unique()
    positions = np.arange(len(models))
    
    data_to_plot = [df_plot[df_plot["Model"] == m]["Accuracy"].values for m in models]
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(models))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('K-Fold Cross-Validation Distribution', fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'figure_kfold_distribution.{fmt}', format=fmt, dpi=300)
    
    plt.close()
    print(f"✓ Generated K-fold distribution figure")


def plot_metrics_comparison_paper(results: Dict, output_dir: Path):
    """Generate multi-metric comparison figure."""
    model_names = []
    metrics_data = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    
    for model_type, model_data in results.items():
        if not model_data.get("has_fold_results", False):
            continue
        
        df = model_data["fold_results"]
        model_name = model_type.replace("_", " ").title()
        model_names.append(model_name)
        
        for metric in ["accuracy", "precision", "recall", "f1"]:
            metric_cols = [col for col in df.columns if metric in col.lower() and col != "fold"]
            if metric_cols:
                metrics_data[metric].append(float(df[metric_cols[0]].mean()))
            else:
                metrics_data[metric].append(0.0)
    
    if not model_names:
        print("No data for metrics comparison plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        if any(v > 0 for v in values):
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=metric.title(), 
                  color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Comparison: Multiple Metrics', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'figure_metrics_comparison.{fmt}', format=fmt, dpi=300)
    
    plt.close()
    print(f"✓ Generated metrics comparison figure")


def plot_statistical_significance_paper(results: Dict, output_dir: Path):
    """Generate statistical significance heatmap."""
    model_data = []
    
    for model_type, model_info in results.items():
        if not model_info.get("has_fold_results", False):
            continue
        
        df = model_info["fold_results"]
        acc_cols = [col for col in df.columns if "acc" in col.lower() and col != "fold"]
        
        if acc_cols:
            values = df[acc_cols[0]].to_list()
            model_data.append({
                "model": model_type,
                "values": values
            })
    
    if len(model_data) < 2:
        print("Need at least 2 models for statistical significance plot.")
        return
    
    # Compute p-value matrix
    n_models = len(model_data)
    p_matrix = np.ones((n_models, n_models))
    model_names = [m["model"].replace("_", " ").title() for m in model_data]
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                t_stat, p_value = stats.ttest_ind(model_data[i]["values"], model_data[j]["values"])
                p_matrix[i, j] = p_value
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('Statistical Significance Matrix (P-values)', fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P-value', fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'figure_statistical_significance.{fmt}', format=fmt, dpi=300)
    
    plt.close()
    print(f"✓ Generated statistical significance figure")


def generate_summary_table(results: Dict, output_dir: Path):
    """Generate LaTeX table of results."""
    summary_data = []
    
    for model_type, model_data in results.items():
        if not model_data.get("has_fold_results", False):
            continue
        
        df = model_data["fold_results"]
        acc_cols = [col for col in df.columns if "acc" in col.lower() and col != "fold"]
        
        if acc_cols:
            acc_col = acc_cols[0]
            values = df[acc_col].to_list()
            n = len(values)
            ci = stats.t.interval(0.95, n-1, loc=np.mean(values), scale=stats.sem(values))
            
            summary_data.append({
                "Model": model_type.replace("_", " ").title(),
                "Mean": f"{np.mean(values):.4f}",
                "Std": f"{np.std(values):.4f}",
                "CI Lower": f"{ci[0]:.4f}",
                "CI Upper": f"{ci[1]:.4f}",
                "Min": f"{np.min(values):.4f}",
                "Max": f"{np.max(values):.4f}",
                "Folds": n
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Save as CSV
        df_summary.to_csv(output_dir / 'table_results_summary.csv', index=False)
        
        # Generate LaTeX table
        latex_table = df_summary.to_latex(index=False, float_format="%.4f",
                                         caption="Model Performance Summary",
                                         label="tab:results")
        
        with open(output_dir / 'table_results_summary.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"✓ Generated summary table (CSV and LaTeX)")


def main():
    """Generate all publication figures."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate publication-ready figures")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).parent.parent),
                       help="Project root directory")
    parser.add_argument("--output-dir", type=str, default="paper_figures",
                       help="Output directory for figures")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    results_dir = project_root / "data" / "training_results"
    output_dir = project_root / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Publication-Ready Figures for IEEE Paper")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load results
    results = load_training_results(results_dir)
    
    if not results:
        print("ERROR: No training results found!")
        print(f"Please ensure results exist in: {results_dir}")
        return
    
    models_with_results = [m for m, d in results.items() if d.get("has_fold_results", False)]
    print(f"Found {len(models_with_results)} models with results: {', '.join(models_with_results)}")
    print()
    
    # Generate figures
    plot_model_comparison_paper(results, output_dir)
    plot_kfold_distribution_paper(results, output_dir)
    plot_metrics_comparison_paper(results, output_dir)
    plot_statistical_significance_paper(results, output_dir)
    generate_summary_table(results, output_dir)
    
    print()
    print("=" * 80)
    print("✓ All figures generated successfully!")
    print(f"Figures saved to: {output_dir}")
    print("=" * 80)
    print("\nFormats generated:")
    print("  - PNG (300 DPI) - for presentations and web")
    print("  - PDF - for LaTeX documents")
    print("  - SVG - for vector graphics")
    print("\nTables generated:")
    print("  - CSV - for data analysis")
    print("  - LaTeX - for paper inclusion")


if __name__ == "__main__":
    main()

