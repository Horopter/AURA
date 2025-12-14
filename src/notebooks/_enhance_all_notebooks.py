#!/usr/bin/env python3
"""
Comprehensive enhancement script for all model notebooks.
Adds architecture details, hyperparameters, MLOps integration, and technical deep-dives.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# Model type to notebook filename mapping
MODEL_MAPPING = {
    "5a_logistic_regression": "logistic_regression",
    "5alpha_sklearn_logreg": "sklearn_logreg",
    "5b_svm": "svm",
    "5beta_gradient_boosting": "gradient_boosting",
    "5c_naive_cnn": "naive_cnn",
    "5d_pretrained_inception": "pretrained_inception",
    "5e_variable_ar_cnn": "variable_ar_cnn",
    "5f_xgboost_pretrained_inception": "xgboost_pretrained_inception",
    "5g_xgboost_i3d": "xgboost_i3d",
    "5h_xgboost_r2plus1d": "xgboost_r2plus1d",
    "5i_xgboost_vit_gru": "xgboost_vit_gru",
    "5j_xgboost_vit_transformer": "xgboost_vit_transformer",
    "5k_vit_gru": "vit_gru",
    "5l_vit_transformer": "vit_transformer",
    "5m_timesformer": "timesformer",
    "5n_vivit": "vivit",
    "5o_i3d": "i3d",
    "5p_r2plus1d": "r2plus1d",
    "5q_x3d": "x3d",
    "5r_slowfast": "slowfast",
    "5s_slowfast_attention": "slowfast_attention",
    "5t_slowfast_multiscale": "slowfast_multiscale",
    "5u_two_stream": "two_stream",
}

# Hyperparameter configurations from grid_search.py
HYPERPARAMS = {
    "logistic_regression": {"C": [0.1, 1.0, 10.0], "max_iter": [1000, 2000]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
    "naive_cnn": {"learning_rate": [5e-4], "weight_decay": [1e-4], "batch_size": [1], "num_epochs": [25]},
    "pretrained_inception": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "variable_ar_cnn": {"learning_rate": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "vit_gru": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "vit_transformer": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "slowfast": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4]},
    "x3d": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "i3d": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "r2plus1d": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "timesformer": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "vivit": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "two_stream": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "slowfast_attention": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "slowfast_multiscale": {"learning_rate": [1e-4], "backbone_lr": [5e-6], "head_lr": [5e-4], "weight_decay": [1e-4], "batch_size": [2]},
    "xgboost_pretrained_inception": {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8]},
    "xgboost_i3d": {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8]},
    "xgboost_r2plus1d": {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8]},
    "xgboost_vit_gru": {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8]},
    "xgboost_vit_transformer": {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8]},
}

def create_architecture_section(model_type: str) -> List[str]:
    """Create architecture deep-dive section."""
    sections = {
        "logistic_regression": [
            "## Architecture Deep-Dive\n\n",
            "**Logistic Regression** is a linear classifier that models the probability of a binary outcome using the logistic function.\n\n",
            "### Mathematical Foundation\n\n",
            "The model predicts:\n\n",
            "$$P(y=1|x) = \\frac{1}{1 + e^{-(w^T x + b)}}$$\n\n",
            "where:\n",
            "- $w$: weight vector (learned parameters)\n",
            "- $b$: bias term\n",
            "- $x$: feature vector (handcrafted features from Stage 2)\n\n",
            "### Implementation\n\n",
            "**Location**: `lib/training/_linear.py`\n\n",
            "```python\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "\n",
            "model = LogisticRegression(\n",
            "    C=1.0,  # Inverse regularization strength\n",
            "    max_iter=2000,  # Maximum iterations\n",
            "    solver='lbfgs',  # Optimizer\n",
            "    class_weight='balanced'  # Handle class imbalance\n",
            ")\n",
            "```\n\n",
            "### Feature Engineering\n\n",
            "Uses **handcrafted features** from Stage 2:\n",
            "- Noise residual energy (3 features)\n",
            "- DCT statistics (5 features)\n",
            "- Blur/sharpness metrics (3 features)\n",
            "- Block boundary inconsistency (1 feature)\n",
            "- Codec cues (3 features)\n",
            "- **Total**: ~15 features per video\n\n",
            "### Regularization\n\n",
            "- **L2 Regularization**: Controlled by `C` parameter (inverse of regularization strength)\n",
            "- **Class Balancing**: `class_weight='balanced'` handles imbalanced datasets\n",
            "- **Feature Scaling**: StandardScaler applied before training\n"
        ],
        "naive_cnn": [
            "## Architecture Deep-Dive\n\n",
            "**Naive CNN** processes video frames independently using 2D convolutions, then aggregates frame-level predictions.\n\n",
            "### Architecture Details\n\n",
            "**Input**: (N, C, T, H, W) or (N, T, C, H, W) video tensors\n",
            "- N: batch size\n",
            "- C: channels (3 for RGB)\n",
            "- T: temporal frames (up to 1000)\n",
            "- H, W: spatial dimensions (256x256 after scaling)\n\n",
            "**Processing Pipeline**:\n",
            "1. **Frame Reshaping**: (N, T, C, H, W) → (N×T, C, H, W)\n",
            "2. **Chunked Processing**: Process 10 frames at a time to avoid OOM\n",
            "3. **2D CNN Layers**:\n",
            "   - Conv2d(3→32) + BatchNorm + ReLU + MaxPool(2)\n",
            "   - Conv2d(32→64) + BatchNorm + ReLU + MaxPool(2)\n",
            "   - Conv2d(64→128) + BatchNorm + ReLU + AdaptiveAvgPool(1,1)\n",
            "4. **Classification Head**:\n",
            "   - Linear(128→64) + ReLU + Dropout(0.5)\n",
            "   - Linear(64→2) for binary classification\n",
            "5. **Temporal Aggregation**: Average frame predictions for video-level output\n\n",
            "### Implementation Code\n\n",
            "**Location**: `lib/training/_cnn.py`\n\n",
            "```python\n",
            "class NaiveCNNBaseline(nn.Module):\n",
            "    def __init__(self, num_frames: int = 1000, num_classes: int = 2):\n",
            "        super().__init__()\n",
            "        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)\n",
            "        self.bn1 = nn.BatchNorm2d(32)\n",
            "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)\n",
            "        self.bn2 = nn.BatchNorm2d(64)\n",
            "        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)\n",
            "        self.bn3 = nn.BatchNorm2d(128)\n",
            "        self.pool = nn.AdaptiveAvgPool2d((1, 1))\n",
            "        self.fc1 = nn.Linear(128, 64)\n",
            "        self.dropout = nn.Dropout(0.5)\n",
            "        self.fc2 = nn.Linear(64, num_classes)\n",
            "        \n",
            "    def forward(self, x):\n",
            "        # Process frames in chunks, average predictions\n",
            "        ...\n",
            "```\n\n",
            "### Memory Optimization\n\n",
            "- **Chunked Processing**: 10 frames per chunk (prevents OOM)\n",
            "- **Batch Size**: 1 (processes 1000 frames per video)\n",
            "- **Gradient Accumulation**: 16 steps (effective batch size = 16)\n",
            "- **Initialization**: He initialization for ReLU activations\n"
        ],
        # Add more architecture sections as needed
    }
    
    return sections.get(model_type, [
        "## Architecture Deep-Dive\n\n",
        f"**{model_type}** architecture details.\n\n",
        "See model implementation in `lib/training/` for specific architecture code.\n"
    ])

def create_hyperparameter_section(model_type: str) -> List[str]:
    """Create hyperparameter configuration section."""
    if model_type not in HYPERPARAMS:
        return [
            "## Hyperparameter Configuration\n\n",
            "Hyperparameters configured in `lib/training/grid_search.py`.\n"
        ]
    
    params = HYPERPARAMS[model_type]
    lines = [
        "## Hyperparameter Configuration\n\n",
        "**Training Hyperparameters** (from `lib/training/grid_search.py`):\n\n"
    ]
    
    for key, value in params.items():
        if isinstance(value, list):
            if len(value) == 1:
                lines.append(f"- **{key}**: {value[0]}\n")
            else:
                lines.append(f"- **{key}**: {value} (grid search)\n")
        else:
            lines.append(f"- **{key}**: {value}\n")
    
    # Add rationale
    if "batch_size" in params:
        batch_size = params["batch_size"][0] if isinstance(params["batch_size"], list) else params["batch_size"]
        lines.extend([
            "\n**Rationale**:\n",
            f"- **Batch Size {batch_size}**: Memory-constrained (processes up to 1000 frames per video)\n",
            "- **Single Hyperparameter Combination**: Reduced from 5+ combinations for training efficiency\n",
            "- **Gradient Accumulation**: Maintains effective batch size despite small batch_size\n"
        ])
    else:
        lines.extend([
            "\n**Rationale**:\n",
            "- **Single Hyperparameter Combination**: Reduced from multiple combinations for efficiency\n",
            "- **Grid Search**: Performed on 20% sample, best params used for full training\n"
        ])
    
    return lines

def create_mlops_section() -> List[str]:
    """Create MLOps integration section."""
    return [
        "## MLOps Integration\n\n",
        "### Experiment Tracking with MLflow\n\n",
        "This model integrates with MLflow for comprehensive experiment tracking:\n\n",
        "```python\n",
        "from lib.mlops.mlflow_tracker import create_mlflow_tracker\n",
        "\n",
        "# MLflow automatically tracks:\n",
        "# - Hyperparameters (learning_rate, batch_size, etc.)\n",
        "# - Metrics (train_loss, val_acc, test_f1, etc.)\n",
        "# - Model artifacts (checkpoints, configs)\n",
        "# - Run metadata (tags, timestamps, fold numbers)\n",
        "```\n\n",
        "**Access MLflow UI**:\n",
        "```bash\n",
        "mlflow ui --port 5000\n",
        "# Open http://localhost:5000\n",
        "```\n\n",
        "### DuckDB Analytics\n\n",
        "Query training results with SQL for fast analytics:\n\n",
        "```python\n",
        "from lib.utils.duckdb_analytics import DuckDBAnalytics\n",
        "\n",
        "analytics = DuckDBAnalytics()\n",
        "analytics.register_parquet('results', 'data/stage5/{model_type}/metrics.json')\n",
        "result = analytics.query(\"\"\"\n",
        "    SELECT \n",
        "        fold,\n",
        "        AVG(test_f1) as avg_f1,\n",
        "        STDDEV(test_f1) as std_f1\n",
        "    FROM results\n",
        "    GROUP BY fold\n",
        "\"\"\")\n",
        "```\n\n",
        "### Airflow Orchestration\n\n",
        "Pipeline orchestrated via Apache Airflow DAG (`airflow/dags/fvc_pipeline_dag.py`):\n",
        "- **Dependency Management**: Automatic task ordering\n",
        "- **Retry Logic**: Automatic retries on failure\n",
        "- **Monitoring**: Web UI for pipeline status\n",
        "- **Scheduling**: Cron-based scheduling support\n"
    ]

def create_training_methodology_section() -> List[str]:
    """Create training methodology section."""
    return [
        "## Training Methodology\n\n",
        "### 5-Fold Stratified Cross-Validation\n\n",
        "- **Purpose**: Robust performance estimates, prevents overfitting\n",
        "- **Stratification**: Ensures class balance in each fold\n",
        "- **Evaluation**: Metrics averaged across folds with standard deviation\n",
        "- **Rationale**: More reliable than single train/test split\n\n",
        "### Regularization Strategy\n\n",
        "- **Weight Decay (L2)**: 1e-4 (PyTorch models)\n",
        "- **Dropout**: 0.5 in classification heads (PyTorch models)\n",
        "- **Early Stopping**: Patience=5 epochs (prevents overfitting)\n",
        "- **Gradient Clipping**: max_norm=1.0 (prevents exploding gradients)\n",
        "- **Class Weights**: Balanced sampling for imbalanced datasets\n\n",
        "### Optimization\n\n",
        "- **Optimizer**: AdamW with betas=(0.9, 0.999)\n",
        "- **Mixed Precision**: AMP (Automatic Mixed Precision) for memory efficiency\n",
        "- **Gradient Accumulation**: Dynamic based on batch size (maintains effective batch size)\n",
        "- **Learning Rate Schedule**: Cosine annealing with warmup (2 epochs)\n",
        "- **Differential Learning Rates**: Lower LR for pretrained backbones (5e-6) vs heads (5e-4)\n\n",
        "### Data Pipeline\n\n",
        "- **Video Loading**: Frame-by-frame decoding (50x memory reduction)\n",
        "- **Augmentation**: Pre-generated augmentations (reproducible, fast)\n",
        "- **Scaling**: Fixed 256x256 max dimension with letterboxing\n",
        "- **Frame Sampling**: Uniform sampling across video duration\n"
    ]

def create_design_rationale_section(model_type: str) -> List[str]:
    """Create design rationale section."""
    rationales = {
        "logistic_regression": [
            "## Design Rationale\n\n",
            "### Why Logistic Regression?\n\n",
            "- **Baseline Model**: Establishes performance floor for comparison\n",
            "- **Interpretability**: Linear model, easy to understand feature importance\n",
            "- **Speed**: Fast training and inference (no GPU required)\n",
            "- **Feature-Based**: Works with handcrafted features (no video processing needed)\n\n",
            "### Limitations\n\n",
            "- **Linear Decision Boundary**: May not capture complex non-linear patterns\n",
            "- **Feature Dependency**: Performance depends on quality of handcrafted features\n",
            "- **No Temporal Modeling**: Treats features as independent (no sequence modeling)\n"
        ],
        "naive_cnn": [
            "## Design Rationale\n\n",
            "### Why \"Naive\" CNN?\n\n",
            "- **Baseline Purpose**: Simple 2D CNN establishes baseline for video models\n",
            "- **Frame-Independent**: Processes each frame independently (no temporal modeling)\n",
            "- **Memory Efficient**: Chunked processing handles long videos (1000 frames)\n",
            "- **Comparison Point**: Demonstrates benefit of temporal models (3D CNNs, Transformers)\n\n",
            "### Trade-offs\n\n",
            "- **No Temporal Modeling**: Loses temporal relationships between frames\n",
            "- **Simple Architecture**: May underfit complex patterns\n",
            "- **Chunked Processing**: Adds complexity but necessary for memory constraints\n"
        ],
    }
    
    return rationales.get(model_type, [
        "## Design Rationale\n\n",
        f"See master pipeline notebook (`00_MASTER_PIPELINE_JOURNEY.ipynb`) for comprehensive design rationale.\n"
    ])

def enhance_notebook(notebook_path: Path, model_type: str) -> None:
    """Enhance a single notebook with comprehensive sections."""
    import re
    # Read and fix control characters
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Remove invalid control characters (except newlines, tabs, carriage returns)
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
    nb = json.loads(content)
    
    # Find insertion point (after "Model Overview" section)
    insert_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and 'Model Overview' in ''.join(cell['source']):
            # Find the end of Model Overview section (before Training Instructions)
            for j in range(i + 1, len(nb['cells'])):
                if 'Training Instructions' in ''.join(nb['cells'][j].get('source', [])):
                    insert_idx = j
                    break
            if insert_idx is None:
                insert_idx = i + 1
            break
    
    if insert_idx is None:
        print(f"⚠ Warning: Could not find insertion point in {notebook_path.name}")
        return
    
    # Create new cells
    new_cells = []
    
    # Architecture section
    arch_source = create_architecture_section(model_type)
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": arch_source
    })
    
    # Hyperparameter section
    hyper_source = create_hyperparameter_section(model_type)
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": hyper_source
    })
    
    # MLOps section
    mlops_source = create_mlops_section()
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": mlops_source
    })
    
    # Training methodology section
    training_source = create_training_methodology_section()
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": training_source
    })
    
    # Design rationale section
    rationale_source = create_design_rationale_section(model_type)
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": rationale_source
    })
    
    # Insert new cells
    for i, cell in enumerate(new_cells):
        nb['cells'].insert(insert_idx + i, cell)
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"✓ Enhanced {notebook_path.name}")

def main():
    """Enhance all model notebooks."""
    notebooks_dir = Path(__file__).parent
    
    enhanced = 0
    skipped = 0
    
    for notebook_file in sorted(notebooks_dir.glob("5*.ipynb")):
        if notebook_file.name.startswith("00_"):  # Skip master notebook
            continue
        
        # Extract model type from filename
        model_type = None
        for key, value in MODEL_MAPPING.items():
            if key in notebook_file.name:
                model_type = value
                break
        
        if model_type:
            try:
                enhance_notebook(notebook_file, model_type)
                enhanced += 1
            except Exception as e:
                print(f"✗ Error enhancing {notebook_file.name}: {e}")
                skipped += 1
        else:
            print(f"⚠ Skipping {notebook_file.name} (no model type mapping)")
            skipped += 1
    
    print(f"\n✓ Enhanced {enhanced} notebooks")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} notebooks")

if __name__ == "__main__":
    main()
