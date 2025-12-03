# Ensemble Model Training

This document describes the ensemble training functionality that combines predictions from multiple trained models.

## Overview

After training individual models, you can train an ensemble model that learns to combine their predictions. This typically improves performance by leveraging the strengths of different model architectures.

## How It Works

1. **Load Trained Models**: For each fold, load all trained base models
2. **Collect Predictions**: Get predictions from each model on train/val sets
3. **Train Meta-Learner**: Train a small MLP to learn optimal combination weights
4. **Save Ensemble**: Save the ensemble model for each fold

## Usage

### Command Line

```bash
# Train individual models first
python src/scripts/run_stage5_training.py \
    --model-types x3d slowfast vit_transformer i3d r2plus1d \
    --n-splits 5

# Then train ensemble (optional)
python src/scripts/run_stage5_training.py \
    --model-types x3d slowfast vit_transformer i3d r2plus1d \
    --n-splits 5 \
    --train-ensemble \
    --ensemble-method meta_learner
```

### In Pipeline

```bash
python src/run_new_pipeline.py \
    --only-stage 5 \
    --model-types x3d slowfast vit_transformer \
    --train-ensemble \
    --ensemble-method meta_learner
```

### Python API

```python
from lib.training.ensemble import train_ensemble_model

results = train_ensemble_model(
    project_root=".",
    scaled_metadata_path="data/scaled_videos/scaled_metadata.arrow",
    base_model_types=["x3d", "slowfast", "vit_transformer", "i3d", "r2plus1d"],
    base_models_dir="data/training_results",
    n_splits=5,
    num_frames=8,
    output_dir="data/training_results",
    ensemble_method="meta_learner"  # or "weighted_average"
)
```

## Ensemble Methods

### 1. Meta-Learner (Recommended)

Trains a small MLP (Multi-Layer Perceptron) to learn optimal combination weights.

**Architecture**:
- Input: Concatenated probabilities from all base models (num_models × 2)
- Hidden layers: 2 layers with ReLU activation
- Output: Binary classification logit

**Advantages**:
- Learns non-linear combinations
- Can adapt to different model strengths
- Typically performs better than simple averaging

**Disadvantages**:
- Requires additional training time
- Slightly more complex

### 2. Weighted Average

Simple average of probabilities from all base models.

**Advantages**:
- Fast (no training needed)
- Simple and interpretable
- No risk of overfitting

**Disadvantages**:
- Assumes all models contribute equally
- May not be optimal

## Requirements

1. **Trained Base Models**: Individual models must be trained first
2. **Same K-Fold Splits**: Ensemble uses the same k-fold splits as base models
3. **Model Checkpoints**: Models must be saved (automatically done after training)

## Output Structure

```
data/training_results/
├── ensemble/
│   ├── fold_1/
│   │   ├── meta_learner.pt      # Trained meta-learner weights
│   │   └── base_models.txt       # List of base models used
│   ├── fold_2/
│   │   └── ...
│   ├── ...
│   └── fold_results.csv          # Ensemble results per fold
├── x3d/
│   └── fold_1/
│       └── model.pt               # Base model checkpoint
├── slowfast/
│   └── ...
└── ...
```

## Results

The ensemble results are saved in `data/training_results/ensemble/fold_results.csv` with columns:
- `fold`: Fold number
- `val_loss`: Validation loss (if meta-learner)
- `val_acc`: Validation accuracy
- `num_base_models`: Number of base models used
- `base_models`: Comma-separated list of base model names

## Best Practices

1. **Diverse Base Models**: Use models with different architectures (e.g., 3D CNN, Transformer, Baseline)
2. **Quality Over Quantity**: 3-5 well-trained models often outperform 10+ mediocre models
3. **Validation**: Ensemble should improve over best individual model
4. **Paper Reporting**: Report both individual and ensemble results

## Troubleshooting

### "No checkpoint found"

**Problem**: Ensemble can't find trained model checkpoints.

**Solution**: Ensure models were trained and saved. Check that `model.pt` exists in each fold directory.

### "No predictions collected"

**Problem**: No base models were successfully loaded.

**Solution**: 
- Verify all base models completed training
- Check that model directories exist
- Review logs for loading errors

### Memory Issues

**Problem**: Running out of memory during ensemble training.

**Solution**:
- Reduce number of base models
- Use `weighted_average` instead of `meta_learner`
- Process one fold at a time

## Performance Expectations

Typical improvements:
- **Meta-Learner**: 1-3% accuracy improvement over best individual model
- **Weighted Average**: 0.5-1.5% accuracy improvement

The improvement depends on:
- Diversity of base models
- Individual model performance
- Dataset characteristics

## For Your Paper

When reporting ensemble results:

1. **Mention Base Models**: List all models used in ensemble
2. **Report Improvement**: Show ensemble vs. best individual model
3. **Explain Method**: Briefly describe meta-learner architecture
4. **Ablation**: Consider reporting weighted average as baseline

Example:
> "We train an ensemble model combining predictions from X3D, SlowFast, ViT-Transformer, I3D, and R(2+1)D. A two-layer MLP meta-learner (64 hidden units) learns to optimally combine model predictions. The ensemble achieves 92.3% accuracy, a 1.8% improvement over the best individual model (X3D: 90.5%)."

