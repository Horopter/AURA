# Comprehensive Optimization Plan

## Critical Issues Identified

### 1. VariableARVideoModel - Unused Model
- **Status**: Defined but never used in training
- **Action**: Add to model_factory OR remove completely
- **Decision**: Add it as "variable_ar_cnn" model type

### 2. Duplicate Frame Counting (CRITICAL)
- **Issue**: Every stage counts frames by decoding entire video
- **Impact**: Massive waste - videos decoded 3-5x per stage
- **Locations**:
  - `augment_video()` - counts frames
  - `load_frames()` - counts frames again
  - `scale_video()` - counts frames again
  - `extract_features_from_video()` - uses metadata but inefficient
- **Fix**: Create cached video metadata system

### 3. File Naming - Not Professional
- **Current**: `naive_cnn.py`, `vit_gru.py`, `logistic_regression.py`
- **Target**: sklearn-style names like `_base.py`, `_cnn.py`, `_transformer.py`

### 4. GPU Optimization Missing
- Missing: `pin_memory`, `non_blocking`, proper device placement
- Need: GPU memory pooling, async transfers

### 5. OOM Issues Throughout
- Stage 1: Some protection
- Stage 2-5: Inconsistent
- Need: Unified OOM handling

### 6. Empty Folders
- `logs/`, `models/`, `runs/runs/` - empty or redundant

