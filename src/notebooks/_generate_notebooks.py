#!/usr/bin/env python3
"""
Generate notebooks for models 5d-5u.
This script creates Jupyter notebooks for all remaining models.
"""

MODEL_CONFIGS = {
    "5d": {
        "name": "pretrained_inception",
        "title": "Pretrained Inception Video Model",
        "description": "Uses pretrained R3D-18 backbone with Inception blocks for variable aspect ratio support.",
        "architecture": "Pretrained R3D-18 (Kinetics-400) backbone + Inception3D blocks + adaptive pooling. Supports variable aspect ratios through convolutional layers and global average pooling.",
        "type": "pytorch",
        "script": "slurm_stage5d.sh"
    },
    "5e": {
        "name": "variable_ar_cnn",
        "title": "Variable Aspect Ratio CNN",
        "description": "Custom 3D CNN with Inception blocks that supports variable aspect ratios via adaptive pooling.",
        "architecture": "Custom 3D CNN with Inception-style blocks. Uses 3D convolutions with adaptive average pooling to handle videos of different aspect ratios and lengths.",
        "type": "pytorch",
        "script": "slurm_stage5e.sh"
    },
    "5f": {
        "name": "xgboost_pretrained_inception",
        "title": "XGBoost with Pretrained Inception Features",
        "description": "XGBoost classifier using features extracted from pretrained Inception video model.",
        "architecture": "Two-stage: (1) Pretrained Inception video model extracts deep features from videos, (2) XGBoost gradient boosting classifier makes final prediction. Combines deep learning feature extraction with tree-based classification.",
        "type": "xgboost",
        "script": "slurm_stage5f.sh"
    },
    "5g": {
        "name": "xgboost_i3d",
        "title": "XGBoost with I3D Features",
        "description": "XGBoost classifier using features extracted from I3D (Inflated 3D ConvNet) model.",
        "architecture": "Two-stage: (1) I3D (Inflated 3D ConvNet) extracts spatiotemporal features, (2) XGBoost makes final prediction. I3D inflates 2D ImageNet pretrained weights into 3D for video understanding.",
        "type": "xgboost",
        "script": "slurm_stage5g.sh"
    },
    "5h": {
        "name": "xgboost_r2plus1d",
        "title": "XGBoost with R(2+1)D Features",
        "description": "XGBoost classifier using features extracted from R(2+1)D (Factorized 3D Convolutions) model.",
        "architecture": "Two-stage: (1) R(2+1)D extracts features using factorized 3D convolutions (2D spatial + 1D temporal), (2) XGBoost makes final prediction. More efficient than full 3D convolutions.",
        "type": "xgboost",
        "script": "slurm_stage5h.sh"
    },
    "5i": {
        "name": "xgboost_vit_gru",
        "title": "XGBoost with ViT-GRU Features",
        "description": "XGBoost classifier using features extracted from Vision Transformer with GRU temporal head.",
        "architecture": "Two-stage: (1) ViT extracts frame-level features, GRU models temporal relationships, (2) XGBoost makes final prediction. Combines transformer spatial understanding with RNN temporal modeling.",
        "type": "xgboost",
        "script": "slurm_stage5i.sh"
    },
    "5j": {
        "name": "xgboost_vit_transformer",
        "title": "XGBoost with ViT-Transformer Features",
        "description": "XGBoost classifier using features extracted from Vision Transformer with Transformer encoder temporal head.",
        "architecture": "Two-stage: (1) ViT extracts frame-level features, Transformer encoder models temporal relationships, (2) XGBoost makes final prediction. Full transformer architecture for both spatial and temporal understanding.",
        "type": "xgboost",
        "script": "slurm_stage5j.sh"
    },
    "5k": {
        "name": "vit_gru",
        "title": "Vision Transformer with GRU",
        "description": "ViT backbone extracts frame features, GRU models temporal relationships.",
        "architecture": "ViT-B/16 extracts features from each frame independently. GRU processes frame features sequentially to model temporal dynamics. Final classification from GRU hidden state.",
        "type": "pytorch",
        "script": "slurm_stage5k.sh"
    },
    "5l": {
        "name": "vit_transformer",
        "title": "Vision Transformer with Transformer Encoder",
        "description": "ViT backbone extracts frame features, Transformer encoder models temporal relationships.",
        "architecture": "ViT-B/16 extracts features from each frame. Transformer encoder with self-attention models temporal relationships across frames. Mean pooling over time for final classification.",
        "type": "pytorch",
        "script": "slurm_stage5l.sh"
    },
    "5m": {
        "name": "timesformer",
        "title": "TimeSformer",
        "description": "Space-time attention transformer for video understanding using divided space-time attention.",
        "architecture": "Divided space-time attention: applies spatial attention within each frame first, then temporal attention across frames. More efficient than joint space-time attention. Uses ViT patch embedding with learnable temporal embeddings.",
        "type": "pytorch",
        "script": "slurm_stage5m.sh"
    },
    "5n": {
        "name": "vivit",
        "title": "ViViT (Video Vision Transformer)",
        "description": "Video Vision Transformer using tubelet embedding for spatiotemporal tokenization.",
        "architecture": "Tubelet embedding: 3D convolutions extract spatiotemporal patches (tubelets) from video. Standard transformer encoder processes tubelet tokens. CLS token for classification. Captures both spatial and temporal information simultaneously.",
        "type": "pytorch",
        "script": "slurm_stage5n.sh"
    },
    "5o": {
        "name": "i3d",
        "title": "I3D (Inflated 3D ConvNet)",
        "description": "Inflated 3D ConvNet that inflates 2D ImageNet pretrained models into 3D for video.",
        "architecture": "Inflates 2D convolutions and pooling to 3D by replicating weights. Uses I3D R50 backbone pretrained on Kinetics-400. Processes full video clips with 3D convolutions for spatiotemporal feature learning.",
        "type": "pytorch",
        "script": "slurm_stage5o.sh"
    },
    "5p": {
        "name": "r2plus1d",
        "title": "R(2+1)D",
        "description": "Factorized 3D convolutions that factorize 3D conv into 2D spatial + 1D temporal.",
        "architecture": "Factorizes 3D convolutions into (2+1)D: 2D spatial convolution followed by 1D temporal convolution. More efficient than full 3D while maintaining representational power. R(2+1)D-18 architecture pretrained on Kinetics-400.",
        "type": "pytorch",
        "script": "slurm_stage5p.sh"
    },
    "5q": {
        "name": "x3d",
        "title": "X3D",
        "description": "Efficient video model that expands 2D networks along multiple dimensions (time, space, width, depth).",
        "architecture": "Systematically expands 2D networks along time, space, width, and depth dimensions. X3D-M variant optimized for efficiency. Uses 3D convolutions with careful dimension scaling for optimal performance/efficiency tradeoff.",
        "type": "pytorch",
        "script": "slurm_stage5q.sh"
    },
    "5r": {
        "name": "slowfast",
        "title": "SlowFast",
        "description": "Dual-pathway architecture: slow pathway for spatial semantics, fast pathway for motion.",
        "architecture": "Two pathways: (1) Slow pathway processes fewer frames at high spatial resolution for semantic understanding, (2) Fast pathway processes more frames at lower resolution for motion. Pathways fused via lateral connections. Pretrained on Kinetics-400.",
        "type": "pytorch",
        "script": "slurm_stage5r.sh"
    },
    "5s": {
        "name": "slowfast_attention",
        "title": "SlowFast with Attention",
        "description": "SlowFast architecture enhanced with self-attention and cross-attention mechanisms.",
        "architecture": "SlowFast dual-pathway architecture with added attention: self-attention within each pathway, cross-attention between slow and fast pathways. Attention mechanisms help pathways focus on relevant spatiotemporal regions and improve information exchange.",
        "type": "pytorch",
        "script": "slurm_stage5s.sh"
    },
    "5t": {
        "name": "slowfast_multiscale",
        "title": "Multi-Scale SlowFast",
        "description": "Multiple temporal sampling pathways at different scales for comprehensive temporal modeling.",
        "architecture": "Extends SlowFast to multiple pathways with different temporal sampling rates (e.g., 1x, 2x, 4x, 8x). Each pathway processes video at different temporal resolutions. Features from all pathways fused for final prediction. Captures both short-term and long-term temporal patterns.",
        "type": "pytorch",
        "script": "slurm_stage5t.sh"
    },
    "5u": {
        "name": "two_stream",
        "title": "Two-Stream Network",
        "description": "Dual streams: RGB stream for appearance, optical flow stream for motion, fused for final prediction.",
        "architecture": "Two parallel streams: (1) RGB stream processes raw video frames for appearance features, (2) Optical flow stream processes computed optical flow for motion features. Streams use ResNet or ViT backbones. Features fused via concatenation, weighted combination, or attention for final classification.",
        "type": "pytorch",
        "script": "slurm_stage5u.sh"
    }
}

def generate_notebook(model_id, config):
    """Generate a Jupyter notebook for a model."""
    
    model_name = config["name"]
    title = config["title"]
    description = config["description"]
    architecture = config.get("architecture", description)
    model_type = config["type"]
    script = config["script"]
    
    # Determine model file extension and load code
    if model_type == "xgboost":
        model_ext = "model.joblib"
        load_code = """    import joblib
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {type(model).__name__}")"""
    else:  # pytorch
        model_ext = "model.pt"
        # Determine num_frames based on model
        num_frames = 500 if model_name in ["naive_cnn", "variable_ar_cnn", "pretrained_inception"] else 1000
        load_code = f"""    # Create model instance
    config = RunConfig(
        run_id="demo",
        experiment_name="demo",
        model_type=MODEL_TYPE,
        num_frames={num_frames}
    )
    model = create_model(MODEL_TYPE, config)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"✓ Model architecture: {{type(model).__name__}}")"""
    
    notebook_content = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# Model {model_id}: {title}\\n",
    "\\n",
    "This notebook demonstrates the {title} for deepfake video detection.\\n",
    "\\n",
    "## Model Overview\\n",
    "\\n",
    "{description}\\n",
    "\\n",
    "## Training Instructions\\n",
    "\\n",
    "To train this model, run:\\n",
    "\\n",
    "```bash\\n",
    "sbatch src/scripts/{script}\\n",
    "```\\n",
    "\\n",
    "Or use Python:\\n",
    "\\n",
    "```python\\n",
    "from lib.training.pipeline import stage5_train_models\\n",
    "\\n",
    "results = stage5_train_models(\\n",
    "    project_root=\\".\\",\\n",
    "    scaled_metadata_path=\\"data/stage3/scaled_metadata.parquet\\",\\n",
    "    features_stage2_path=\\"data/stage2/features_metadata.parquet\\",\\n",
    "    features_stage4_path=None,\\n",
    "    model_types=[\\"{model_name}\\"],\\n",
    "    n_splits=5,\\n",
    "    num_frames=1000,\\n",
    "    output_dir=\\"data/stage5\\",\\n",
    "    use_tracking=True,\\n",
    "    use_mlflow=True\\n",
    ")\\n",
    "```"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "import sys\\n",
    "from pathlib import Path\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import polars as pl\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from IPython.display import Video, display, HTML\\n",
    "import json\\n",
    "import torch\\n",
    "import torch.nn as nn\\n",
    "\\n",
    "# Add project root to path\\n",
    "project_root = Path().absolute().parent.parent\\n",
    "sys.path.insert(0, str(project_root))\\n",
    "\\n",
    "from lib.training.model_factory import create_model\\n",
    "from lib.mlops.config import RunConfig\\n",
    "from lib.utils.paths import load_metadata_flexible\\n",
    "from lib.training.metrics_utils import compute_classification_metrics\\n",
    "\\n",
    "# Configuration\\n",
    "MODEL_TYPE = \\"{model_name}\\"\\n",
    "MODEL_DIR = project_root / \\"data\\" / \\"stage5\\" / MODEL_TYPE\\n",
    "SCALED_METADATA_PATH = project_root / \\"data\\" / \\"stage3\\" / \\"scaled_metadata.parquet\\"\\n",
    "\\n",
    "print(f\\"Project root: {{project_root}}\\")\\n",
    "print(f\\"Model directory: {{MODEL_DIR}}\\")\\n",
    "print(f\\"Model directory exists: {{MODEL_DIR.exists()}}\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Check for Saved Models"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "def check_saved_models(model_dir: Path):\\n",
    "    \\"\\"\\"Check for saved model files.\\"\\"\\"\\n",
    "    if not model_dir.exists():\\n",
    "        print(f\\"❌ Model directory does not exist: {{model_dir}}\\")\\n",
    "        return False, []\\n",
    "    \\n",
    "    fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith(\\"fold_\\")])\\n",
    "    \\n",
    "    if not fold_dirs:\\n",
    "        print(f\\"❌ No fold directories found in {{model_dir}}\\")\\n",
    "        return False, []\\n",
    "    \\n",
    "    print(f\\"✓ Found {{len(fold_dirs)}} fold(s)\\")\\n",
    "    \\n",
    "    models_found = []\\n",
    "    for fold_dir in fold_dirs:\\n",
    "        model_file = fold_dir / \\"{model_ext}\\"\\n",
    "        if model_file.exists():\\n",
    "            models_found.append((fold_dir.name, model_file))\\n",
    "            print(f\\"  ✓ {{fold_dir.name}}: Found {model_ext}\\")\\n",
    "        else:\\n",
    "            print(f\\"  ❌ {{fold_dir.name}}: No {model_ext} found\\")\\n",
    "    \\n",
    "    return len(models_found) > 0, models_found\\n",
    "\\n",
    "models_available, model_files = check_saved_models(MODEL_DIR)\\n",
    "\\n",
    "if not models_available:\\n",
    "    print(\\"\\\\n⚠️  No trained models found. Please train the model first using the instructions above.\\")\\n",
    "    print(f\\"Expected location: {{MODEL_DIR}}\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Load Model"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "if models_available:\\n",
    "    fold_name, model_path = model_files[0]\\n",
    "    print(f\\"Loading model from: {{model_path}}\\")\\n",
    "    \\n{load_code}
    "    \\n",
    "    print(f\\"✓ Model loaded successfully from {{fold_name}}\\")\\n",
    "    print(f\\"Model type: {{type(model)}}\\")\\n",
    "    \\n",
    "    # Load metadata\\n",
    "    scaled_df = load_metadata_flexible(str(SCALED_METADATA_PATH))\\n",
    "    \\n",
    "    if scaled_df is not None:\\n",
    "        print(f\\"\\\\n✓ Loaded {{scaled_df.height}} videos from scaled metadata\\")\\n",
    "        sample_videos = scaled_df.head(5).to_pandas()\\n",
    "        print(f\\"\\\\nSample videos for demonstration:\\")\\n",
    "        print(sample_videos[[\\"video_path\\", \\"label\\"]].to_string())\\n",
    "    else:\\n",
    "        print(\\"⚠️  Could not load metadata files\\")\\n",
    "else:\\n",
    "    print(\\"⚠️  Skipping model loading - no trained models found\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Display Sample Videos"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "if models_available and 'sample_videos' in locals():\\n",
    "    fig, axes = plt.subplots(1, min(3, len(sample_videos)), figsize=(15, 5))\\n",
    "    if len(sample_videos) == 1:\\n",
    "        axes = [axes]\\n",
    "    \\n",
    "    for idx, (ax, row) in enumerate(zip(axes, sample_videos.iterrows())):\\n",
    "        video_path = project_root / row[1][\\"video_path\\"]\\n",
    "        label = row[1][\\"label\\"]\\n",
    "        \\n",
    "        try:\\n",
    "            import cv2\\n",
    "            cap = cv2.VideoCapture(str(video_path))\\n",
    "            if cap.isOpened():\\n",
    "                ret, frame = cap.read()\\n",
    "                if ret:\\n",
    "                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\\n",
    "                    ax.imshow(frame_rgb)\\n",
    "                    ax.set_title(f\\"{{Path(video_path).name}}\\\\nLabel: {{label}}\\", fontsize=10)\\n",
    "                cap.release()\\n",
    "        except Exception as e:\\n",
    "            ax.text(0.5, 0.5, f\\"Video: {{Path(video_path).name}}\\\\nLabel: {{label}}\\", \\n",
    "                    ha='center', va='center', fontsize=12, transform=ax.transAxes)\\n",
    "        ax.axis('off')\\n",
    "    \\n",
    "    plt.tight_layout()\\n",
    "    plt.show()\\n",
    "    \\n",
    "    print(\\"\\\\nNote: To play videos in the notebook, use:\\")\\n",
    "    print(\\"display(Video('path/to/video.mp4', embed=True, width=640, height=480))\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Model Performance Summary"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "if models_available:\\n",
    "    fold_dir = model_files[0][0]\\n",
    "    metrics_file = MODEL_DIR / fold_dir / \\"metrics.json\\"\\n",
    "    \\n",
    "    if metrics_file.exists():\\n",
    "        with open(metrics_file, 'r') as f:\\n",
    "            metrics = json.load(f)\\n",
    "        \\n",
    "        print(\\"Model Performance Metrics:\\")\\n",
    "        print(\\"=\\" * 50)\\n",
    "        for key, value in metrics.items():\\n",
    "            if isinstance(value, (int, float)):\\n",
    "                print(f\\"{{key}}: {{value:.4f}}\\")\\n",
    "            else:\\n",
    "                print(f\\"{{key}}: {{value}}\\")\\n",
    "        \\n",
    "        if 'accuracy' in metrics or 'f1_score' in metrics:\\n",
    "            fig, ax = plt.subplots(figsize=(8, 6))\\n",
    "            metric_names = ['accuracy', 'precision', 'recall', 'f1_score']\\n",
    "            metric_values = [metrics.get(m, 0) for m in metric_names]\\n",
    "            \\n",
    "            bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])\\n",
    "            ax.set_ylabel('Score')\\n",
    "            ax.set_title('{title} Model Performance')\\n",
    "            ax.set_ylim(0, 1)\\n",
    "            \\n",
    "            for bar, val in zip(bars, metric_values):\\n",
    "                height = bar.get_height()\\n",
    "                ax.text(bar.get_x() + bar.get_width()/2., height,\\n",
    "                       f'{{val:.3f}}', ha='center', va='bottom')\\n",
    "            \\n",
    "            plt.tight_layout()\\n",
    "            plt.show()\\n",
    "    else:\\n",
    "        print(\\"⚠️  Metrics file not found.\\")\\n",
    "        print(f\\"Expected: {{metrics_file}}\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Model Architecture Summary\\n",
    "\\n",
    "**{title}**\\n",
    "\\n",
    "{architecture}\\n",
    "\\n",
    "**Implementation:**\\n",
    "- Model code: `lib/training/{model_name}.py`\\n",
    "\\n",
    "**Advantages:**\\n",
    "- See model-specific advantages in the implementation\\n",
    "\\n",
    "**Use Cases:**\\n",
    "- Deepfake video detection\\n",
    "- Video authenticity verification"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''
    
    return notebook_content

if __name__ == "__main__":
    from pathlib import Path
    
    output_dir = Path(__file__).parent
    
    for model_id, config in MODEL_CONFIGS.items():
        notebook_content = generate_notebook(model_id, config)
        output_path = output_dir / f"{model_id}_{config['name']}.ipynb"
        
        with open(output_path, 'w') as f:
            f.write(notebook_content)
        
        print(f"Generated: {output_path}")
