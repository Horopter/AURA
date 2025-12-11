# ML Methodology and Best Practices

This document outlines critical statistical and ML methodology issues that were identified and fixed, along with comprehensive ML/MLOps best practices implemented to ensure valid results and optimal training performance.

## Critical Methodology Fixes

### 1. **Data Leakage via Duplicate Groups** (CRITICAL - FIXED)

**Problem**: The `stratified_kfold` function did NOT handle `dup_group`, meaning videos from the same duplicate group could appear in both training and validation sets. This causes **severe data leakage** and invalidates all results.

**Impact**: 
- Model performance would be artificially inflated
- Results would not generalize to new data
- Paper would be rejected for methodological flaws

**Fix**: Modified `stratified_kfold` to:
- Group by `dup_group` first (if present)
- Assign entire groups to folds, not individual videos
- Ensures all videos in a duplicate group stay in the same fold
- Added validation check to detect and prevent leakage

**Location**: `lib/data/loading.py::stratified_kfold()`

### 2. **Function Signature Mismatch** (CRITICAL - FIXED)

**Problem**: `pipeline.py` called `stratified_kfold()` with parameters `fold_idx` and `random_seed` that don't exist. The function returns a list of all folds, not a single fold.

**Impact**: Code would crash at runtime.

**Fix**: 
- Updated `pipeline.py` to get all folds at once
- Extract specific fold from the list
- Added data leakage validation check per fold

**Location**: `lib/training/pipeline.py::stage5_train_models()`

### 3. **No Test Set for Single Runs** (IMPORTANT - RECOMMENDED)

**Problem**: For a single run (not multiple runs for statistical validation), you need a proper train/val/test split, not just k-fold CV. K-fold CV is for model selection, but you need a held-out test set for final evaluation.

**Recommendation**: 
- Use `train_val_test_split()` for single runs
- Reserve test set (10%) for final evaluation only
- Use validation set (20%) for hyperparameter tuning
- Use training set (70%) for training

**Status**: Function exists but not used in pipeline. Consider adding option.

### 4. **Feature Collinearity Removal Timing** (MINOR - ACCEPTABLE)

**Current**: Collinearity removal happens per-model during training (in `fit()` method).

**Status**: This is acceptable because:
- Features are extracted per-video (no global statistics)
- Collinearity removal is done on training set only
- Each fold gets its own collinearity analysis

**Note**: This is actually correct - collinearity should be removed per-fold to avoid leakage.

## Validation Checks Added

### Data Leakage Detection

After each fold split, the code now:
1. Checks if `dup_group` exists
2. Verifies no duplicate groups appear in both train and val
3. Logs warning and raises error if leakage detected
4. Logs confirmation if no leakage found

### Class Balance Verification

The code checks:
- All classes present in both train and val sets
- Class ratios are balanced (within 20% difference)
- Logs warnings for imbalanced splits

## ML/MLOps Best Practices

### 1. Gradient Clipping

**Problem**: Gradient explosion can cause training instability, especially in deep networks.

**Solution**: Implemented gradient clipping with configurable `max_grad_norm`.

```python
# In OptimConfig
max_grad_norm: float = 1.0  # Clip gradients to this norm (0 = disabled)

# During training
if max_grad_norm > 0:
    if scaler is not None:
        scaler.unscale_(optimizer)  # Required for AMP
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm=max_grad_norm
    )
```

**Benefits**:
- Prevents gradient explosion
- Stabilizes training
- Works with mixed precision (AMP)

### 2. Learning Rate Scheduling

**Problem**: Fixed learning rate or simple step decay can lead to suboptimal convergence.

**Solution**: Implemented warmup + cosine annealing scheduler.

```python
# In TrainConfig
scheduler_type: str = "cosine"  # "cosine", "step", or "none"
warmup_epochs: int = 2  # Number of warmup epochs
warmup_factor: float = 0.1  # Initial LR = base_lr * warmup_factor
```

**Benefits**:
- **Warmup**: Gradually increases LR from small value, preventing early instability
- **Cosine Annealing**: Smoothly decreases LR, allowing fine-tuning near convergence
- Better convergence than StepLR

### 3. Differential Learning Rates

**Problem**: Pretrained models need different learning rates for backbone (frozen/pretrained) vs head (newly initialized).

**Solution**: Automatic detection and separate parameter groups.

```python
# In OptimConfig
backbone_lr: Optional[float] = None  # If None, uses lr
head_lr: Optional[float] = None  # If None, uses lr * 10 (common practice)

# Automatically applied for pretrained models:
# - I3D, R(2+1)D, SlowFast, X3D
# - ViT-GRU, ViT-Transformer
```

**Benefits**:
- Backbone learns slowly (preserves pretrained features)
- Head learns quickly (new task-specific features)
- Better transfer learning performance

### 4. Batch Normalization

**Problem**: BatchNorm behaves differently in train vs eval mode.

**Solution**: Explicit mode management in training loop.

```python
# Training
model.train()  # BatchNorm uses batch statistics, Dropout active
train_loss = train_one_epoch(...)

# Evaluation
model.eval()  # BatchNorm uses running statistics, Dropout disabled
val_loss, val_acc = evaluate(...)
```

**Benefits**:
- Correct behavior during training (batch statistics)
- Correct behavior during evaluation (running statistics)
- Consistent results

### 5. Dropout

**Problem**: Dropout must be disabled during evaluation.

**Solution**: Automatic mode switching.

```python
# Training: Dropout active
model.train()
output = model(input)

# Evaluation: Dropout disabled
model.eval()
with torch.no_grad():
    output = model(input)
```

**Benefits**:
- Correct training behavior (regularization)
- Correct evaluation behavior (no dropout)
- Consistent results

### 6. Weight Initialization

**Problem**: Poor initialization can lead to vanishing/exploding gradients.

**Solution**: Proper initialization for custom models.

```python
def _initialize_weights(self):
    """Initialize model weights using He initialization for ReLU activations."""
    for m in self.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
```

**Benefits**:
- Prevents vanishing/exploding gradients
- Faster convergence
- Better training stability

### 7. Gradient Monitoring

**Problem**: Hard to debug gradient issues without visibility.

**Solution**: Logging gradient norms for debugging.

```python
# In TrainConfig
log_grad_norm: bool = False  # Enable to log gradient norms

# During training
if log_grad_norm:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    logger.info(f"Gradient norm: {grad_norm:.4f}")
```

**Benefits**:
- Debug gradient issues
- Monitor training stability
- Detect vanishing/exploding gradients

## Recommendations for Single Run

Since you can only run once or twice, follow these guidelines:

### Option 1: Single Train/Val/Test Split (Recommended)

```python
from lib.data import train_val_test_split, SplitConfig

# Create single split
cfg = SplitConfig(val_size=0.2, test_size=0.1, random_state=42)
splits = train_val_test_split(scaled_df, cfg)

train_df = splits["train"]
val_df = splits["val"]
test_df = splits["test"]  # Use ONLY for final evaluation

# Train on train_df, tune on val_df, evaluate ONCE on test_df
```

**Advantages**:
- Clear separation of concerns
- Test set used only once (no overfitting)
- Standard practice for single runs

**Disadvantages**:
- Less robust than k-fold CV
- Single evaluation may be noisy

### Option 2: K-Fold CV (Current Implementation)

**Current**: Uses 5-fold CV, which is acceptable but:
- Each fold's validation set is used for evaluation
- No separate test set
- Results are average across folds

**For Paper**:
- Report mean ± std across folds
- Mention that this is cross-validation, not a held-out test set
- Consider adding a final test set if possible

## What Was Checked

1. ✅ **Data Leakage**: Fixed duplicate group handling
2. ✅ **Function Signatures**: Fixed mismatched parameters
3. ✅ **Stratification**: Verified stratified splits by label and platform
4. ✅ **Class Balance**: Added checks and warnings
5. ✅ **Feature Extraction**: Verified per-video (no global leakage)
6. ✅ **Collinearity Removal**: Verified per-fold (correct)
7. ✅ **Gradient Clipping**: Implemented
8. ✅ **Learning Rate Scheduling**: Warmup + cosine annealing
9. ✅ **Differential Learning Rates**: For pretrained models
10. ✅ **Batch Normalization**: Proper train/eval mode
11. ✅ **Dropout**: Properly disabled during evaluation
12. ✅ **Weight Initialization**: Proper initialization
13. ✅ **Gradient Monitoring**: Optional logging

## What to Watch For

1. **Test Set Usage**: If you add a test set, use it ONLY ONCE for final evaluation
2. **Random Seeds**: Use fixed random seeds (42) for reproducibility
3. **Data Leakage**: Always verify no duplicate groups across splits
4. **Class Imbalance**: Monitor class ratios in each split
5. **Feature Scaling**: Ensure scaler is fit only on training data
6. **Gradient Norms**: Monitor for vanishing/exploding gradients
7. **Learning Rate**: Monitor LR schedule and adjust if needed
8. **Batch Normalization**: Ensure proper train/eval mode switching

## For Your Paper

When writing the methodology section:

1. **Data Splitting**: 
   - "We use stratified k-fold cross-validation (k=5) with stratification by label and platform"
   - "Duplicate video groups are kept together to prevent data leakage"

2. **Validation**:
   - "We verify no duplicate groups appear in both training and validation sets"
   - "Class balance is maintained across folds (±20%)"

3. **Feature Preprocessing**:
   - "Collinear features (correlation > 0.95) are removed per-fold using training data only"
   - "Feature scaling is fit on training data and applied to validation data"

4. **Training**:
   - "We use gradient clipping (max_norm=1.0) to prevent gradient explosion"
   - "Learning rate scheduling with warmup (2 epochs) and cosine annealing"
   - "Differential learning rates for pretrained models (backbone: 1e-5, head: 1e-4)"
   - "Batch normalization and dropout properly managed in train/eval modes"

5. **Evaluation**:
   - "Results are reported as mean ± standard deviation across 5 folds"
   - "No hyperparameter tuning was performed on the validation set" (if true)

## Critical Reminders

1. **NEVER** use test set for model selection or hyperparameter tuning
2. **ALWAYS** verify no data leakage (duplicate groups)
3. **ALWAYS** use fixed random seeds for reproducibility
4. **ALWAYS** report mean ± std, not just mean
5. **ALWAYS** mention limitations (single run, no separate test set if applicable)
6. **ALWAYS** ensure proper train/eval mode switching
7. **ALWAYS** monitor gradient norms during training
8. **ALWAYS** validate feature scaling is fit only on training data

## Verification Checklist

Before running your final experiment:

- [ ] Verify `dup_group` is present in metadata
- [ ] Run data leakage check (automatic, but verify logs)
- [ ] Check class balance in each fold
- [ ] Use fixed random seed (42)
- [ ] Verify gradient clipping is enabled
- [ ] Verify learning rate scheduling is configured
- [ ] Verify differential learning rates for pretrained models
- [ ] Verify proper train/eval mode switching
- [ ] Document exact methodology in paper
- [ ] Report appropriate statistics (mean ± std)
- [ ] Mention limitations honestly

## Feature Collinearity Removal

### Overview

Feature collinearity (high correlation between features) can cause several problems:
- **Multicollinearity**: Makes it difficult to interpret feature importance
- **Numerical instability**: Can cause convergence issues in linear models
- **Overfitting**: Redundant features can lead to overfitting
- **Increased computation**: Unnecessary features increase training time

The pipeline automatically removes collinear features before training to mitigate these issues.

### Methods

The collinearity removal supports two methods:

1. **Correlation-based** (default): Removes features with correlation ≥ 0.95
   - Computes pairwise correlation matrix
   - For highly correlated pairs, removes the feature with lower variance
   - Fast and effective for most cases

2. **VIF-based** (optional): Uses Variance Inflation Factor (VIF)
   - Calculates VIF for each feature
   - Removes features with VIF > 10.0
   - More computationally expensive but more rigorous

3. **Both**: Applies correlation filtering first, then VIF filtering

### When It's Applied

Collinearity removal is automatically applied:
1. **Baseline Models** (Logistic Regression, SVM): Applied during `fit()` after feature extraction
2. **Feature Loading**: When loading features from Stage 2 and Stage 4 metadata files

### Configuration

```python
from lib.training.feature_preprocessing import remove_collinear_features

features_filtered, kept_indices, kept_names = remove_collinear_features(
    features,
    feature_names=feature_names,
    correlation_threshold=0.95,  # Maximum correlation allowed
    vif_threshold=10.0,          # Maximum VIF allowed
    method="correlation"          # "correlation", "vif", or "both"
)
```

### Default Settings

- **Correlation threshold**: 0.95 (features with |correlation| ≥ 0.95 are considered collinear)
- **VIF threshold**: 10.0 (features with VIF > 10.0 are considered problematic)
- **Method**: "correlation" (fastest and most commonly used)

## Ensemble Training

### Overview

After training individual models, you can train an ensemble model that learns to combine their predictions. This typically improves performance by leveraging the strengths of different model architectures.

### How It Works

1. **Load Trained Models**: For each fold, load all trained base models
2. **Collect Predictions**: Get predictions from each model on train/val sets
3. **Train Meta-Learner**: Train a small MLP to learn optimal combination weights
4. **Save Ensemble**: Save the ensemble model for each fold

### Usage

```bash
# Train individual models first
python src/scripts/run_stage5_training.py \
    --model-types x3d slowfast vit_transformer i3d r2plus1d \
    --n-splits 5 \
    --train-ensemble \
    --ensemble-method meta_learner
```

### Ensemble Methods

1. **Meta-Learner** (Recommended): Trains a small MLP to learn optimal combination weights
   - Input: Concatenated probabilities from all base models
   - Hidden layers: 2 layers with ReLU activation
   - Output: Binary classification logit
   - Advantages: Learns non-linear combinations, typically performs better
   - Disadvantages: Requires additional training time

2. **Weighted Average**: Simple average of probabilities from all base models
   - Advantages: Fast, simple, interpretable, no risk of overfitting
   - Disadvantages: Assumes all models contribute equally

### Requirements

1. **Trained Base Models**: Individual models must be trained first
2. **Same K-Fold Splits**: Ensemble uses the same k-fold splits as base models
3. **Model Checkpoints**: Models must be saved (automatically done after training)

### Best Practices

1. **Diverse Base Models**: Use models with different architectures (e.g., 3D CNN, Transformer, Baseline)
2. **Quality Over Quantity**: 3-5 well-trained models often outperform 10+ mediocre models
3. **Validation**: Ensemble should improve over best individual model
4. **Paper Reporting**: Report both individual and ensemble results

### Performance Expectations

Typical improvements:
- **Meta-Learner**: 1-3% accuracy improvement over best individual model
- **Weighted Average**: 0.5-1.5% accuracy improvement

---

**Last Updated**: After critical methodology review and ML best practices implementation
**Status**: All critical issues fixed and validated, best practices implemented
