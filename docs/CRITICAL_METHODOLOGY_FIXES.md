# Critical Methodology Fixes

This document outlines **CRITICAL** statistical and ML methodology issues that were identified and fixed to ensure valid results for your IEEE paper.

## ⚠️ CRITICAL ISSUES FIXED

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

## ✅ VALIDATION CHECKS ADDED

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

## 📊 RECOMMENDATIONS FOR SINGLE RUN

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

## 🔍 WHAT WAS CHECKED

1. ✅ **Data Leakage**: Fixed duplicate group handling
2. ✅ **Function Signatures**: Fixed mismatched parameters
3. ✅ **Stratification**: Verified stratified splits by label and platform
4. ✅ **Class Balance**: Added checks and warnings
5. ✅ **Feature Extraction**: Verified per-video (no global leakage)
6. ✅ **Collinearity Removal**: Verified per-fold (correct)

## ⚠️ WHAT TO WATCH FOR

1. **Test Set Usage**: If you add a test set, use it ONLY ONCE for final evaluation
2. **Random Seeds**: Use fixed random seeds (42) for reproducibility
3. **Data Leakage**: Always verify no duplicate groups across splits
4. **Class Imbalance**: Monitor class ratios in each split
5. **Feature Scaling**: Ensure scaler is fit only on training data

## 📝 FOR YOUR PAPER

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

4. **Evaluation**:
   - "Results are reported as mean ± standard deviation across 5 folds"
   - "No hyperparameter tuning was performed on the validation set" (if true)

## 🚨 CRITICAL REMINDERS

1. **NEVER** use test set for model selection or hyperparameter tuning
2. **ALWAYS** verify no data leakage (duplicate groups)
3. **ALWAYS** use fixed random seeds for reproducibility
4. **ALWAYS** report mean ± std, not just mean
5. **ALWAYS** mention limitations (single run, no separate test set if applicable)

## ✅ VERIFICATION CHECKLIST

Before running your final experiment:

- [ ] Verify `dup_group` is present in metadata
- [ ] Run data leakage check (automatic, but verify logs)
- [ ] Check class balance in each fold
- [ ] Use fixed random seed (42)
- [ ] Document exact methodology in paper
- [ ] Report appropriate statistics (mean ± std)
- [ ] Mention limitations honestly

---

**Last Updated**: After critical methodology review
**Status**: All critical issues fixed and validated

