# Feature Collinearity Removal

This document describes the automatic collinearity removal feature that is applied before training to improve model performance and stability.

## Overview

Feature collinearity (high correlation between features) can cause several problems in machine learning:
- **Multicollinearity**: Makes it difficult to interpret feature importance
- **Numerical instability**: Can cause convergence issues in linear models
- **Overfitting**: Redundant features can lead to overfitting
- **Increased computation**: Unnecessary features increase training time

The pipeline automatically removes collinear features before training to mitigate these issues.

## Implementation

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

1. **Baseline Models** (Logistic Regression, SVM):
   - Applied during `fit()` after feature extraction
   - Features are filtered before scaling and training
   - Same filtering is applied during `predict()` for consistency

2. **Feature Loading**:
   - When loading features from Stage 2 and Stage 4 metadata files
   - Before combining features from multiple stages

### Configuration

The collinearity removal can be configured with:

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

## How It Works

### Step 1: Zero Variance Removal

First, features with zero or near-zero variance are removed:
- These features provide no information
- Can cause numerical issues in some models

### Step 2: Correlation Analysis

For correlation-based method:
1. Compute correlation matrix for all features
2. Identify pairs with |correlation| ≥ threshold
3. For each pair, remove the feature with lower variance
4. Continue until no highly correlated pairs remain

### Step 3: VIF Analysis (if enabled)

For VIF-based method:
1. Calculate VIF for each feature
2. Remove features with VIF > threshold
3. Recalculate VIF for remaining features
4. Repeat until all features have acceptable VIF

## Example Output

```
Removed 2 features with zero variance
Removed 5 collinear features (correlation >= 0.95)
Final feature count: 43/50 (86.0% retained)
Using 43 features after collinearity removal
```

## Benefits

1. **Improved Model Stability**: Reduces numerical instability in linear models
2. **Faster Training**: Fewer features mean faster training
3. **Better Generalization**: Reduces overfitting from redundant features
4. **Clearer Interpretability**: Remaining features are more independent

## Dependencies

The collinearity removal requires:
- `scikit-learn` (for VarianceThreshold)
- `statsmodels` (for VIF calculation, optional)

Install with:
```bash
pip install scikit-learn statsmodels
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Logging

The collinearity removal process is logged at INFO level:
- Number of features removed at each step
- Final feature count and retention percentage
- Feature names that were removed (at DEBUG level)

## Integration

The collinearity removal is automatically integrated into:

1. **`LogisticRegressionBaseline.fit()`**: Applied before scaling
2. **`SVMBaseline.fit()`**: Applied before scaling
3. **`load_and_combine_features()`**: Applied when loading features from metadata files

No additional configuration is needed - it works automatically!

## Customization

To customize collinearity removal for your use case:

```python
# In your training script
from lib.training.feature_preprocessing import remove_collinear_features

# More aggressive removal (lower correlation threshold)
features_filtered, indices, names = remove_collinear_features(
    features,
    correlation_threshold=0.90,  # More aggressive
    method="correlation"
)

# Use VIF for more rigorous filtering
features_filtered, indices, names = remove_collinear_features(
    features,
    vif_threshold=5.0,  # Stricter VIF threshold
    method="vif"
)

# Use both methods
features_filtered, indices, names = remove_collinear_features(
    features,
    correlation_threshold=0.95,
    vif_threshold=10.0,
    method="both"
)
```

## References

- [Multicollinearity in Regression](https://en.wikipedia.org/wiki/Multicollinearity)
- [Variance Inflation Factor](https://en.wikipedia.org/wiki/Variance_inflation_factor)
- [Feature Selection in scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)

