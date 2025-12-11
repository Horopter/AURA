# Pipeline Optimizations

This document describes all optimizations implemented across the FVC pipeline, including memory optimizations, MLOps improvements, and system enhancements.

## Memory Optimizations

### 1. Reduced Resource Usage

**Batch Size Reduction**:
- **Before**: `batch_size=32` (16 real + 16 fake per batch)
- **After**: Ultra-conservative batch sizes per model (1-8 depending on model)
- **Impact**: ~4-32x reduction in memory per batch
- **Compensation**: Increased `gradient_accumulation_steps` (8-16) to maintain effective batch size

**Number of Workers**:
- **Before**: `num_workers=4`
- **After**: `num_workers=0` (CPU-only or test mode to avoid multiprocessing memory overhead)
- **Impact**: Eliminates memory overhead from parallel data loading workers

**Frame Count**:
- **Before**: `num_frames=16`
- **After**: `num_frames=6` (ultra-conservative)
- **Impact**: ~2.7x reduction in memory per video sample

**Resolution**:
- **Before**: `fixed_size=224` (224×224 pixels)
- **After**: `fixed_size=112` (112×112 pixels, configurable via `FVC_FIXED_SIZE`)
- **Impact**: ~4x reduction in memory per frame (224² → 112²)

### 2. Frame-by-Frame Video Decoding (CRITICAL)

**Problem**: Loading entire videos into memory causes massive memory spikes
- Large video (1920×1080, 30fps, 10s = 300 frames): ~1.87 GB per video
- With base memory (~31GB), loading one large video can spike past 80GB → OOM

**Solution**: Decode only the frames we need using PyAV
- **Before**: Load entire video → extract 6 frames → delete video
- **After**: Seek to specific frames → decode only those 6 frames → never load full video
- **Memory per video**: ~37 MB (6 frames) instead of ~1.87 GB (300 frames)
- **Memory reduction**: ~50x reduction per video

**Benefits**:
- Prevents OOM during augmentation generation
- Allows processing large videos without memory spikes
- Stable memory usage (~1-2 GB instead of 30+ GB)

### 3. Incremental CSV Writing for Metadata

**Problem**: Accumulating all augmented clip metadata in memory
- 298 videos × 1 augmentation = 298 metadata rows
- Each row: video_path, label, original_video, augmentation_idx
- Memory accumulation as list grows

**Solution**: Write metadata directly to CSV incrementally
- **Before**: `augmented_rows.append({...})` → `pl.DataFrame(augmented_rows)` at end
- **After**: Write each row immediately to CSV file
- **Memory**: Constant (no accumulation)

**Benefits**:
- Eliminates unbounded memory growth from metadata list
- Memory stays constant regardless of dataset size

### 4. One Video at a Time Processing

**Implementation**:
- Process videos one at a time (`batch_size=1`) during augmentation generation
- Aggressive GC after each video
- Clear video tensors and clips from memory immediately after processing
- Delete frames after stacking into clip
- Delete clip after saving to disk

**Benefits**:
- Prevents memory accumulation during augmentation generation
- Allows processing large datasets without OOM
- Minimal peak memory usage

### 5. Shared Augmentations Across K-Fold

**Before**: Each fold generated its own augmentations
- Memory: 5x augmentation generation (one per fold)
- Disk: 5x storage space
- Time: 5x generation time

**After**: Single shared augmentation generation
- Generate augmentations once for all unique videos
- Filter augmentations per fold based on training videos
- Memory: 1x generation
- Disk: 1x storage (shared across folds)
- Time: 1x generation time

### 6. Aggressive Garbage Collection

**Enhanced GC Strategy**:
- 3 passes of `gc.collect()` instead of 1
- CUDA cache clearing after every batch
- CUDA synchronization to ensure cleanup
- GC after every pipeline stage
- GC after every epoch
- GC after every k-fold fold

## Memory Usage Estimates

### Before All Optimizations
- Batch size 32: ~2-3 GB per batch
- 16 frames: ~1.5 GB per batch
- 4 workers: ~500 MB overhead
- Full video loading: ~1.87 GB per video during augmentation
- **Total**: ~4-5 GB per batch + model + overhead + video loading = **~80 GB** (OOM)

### After Latest Optimizations (Current)
- Batch size 1-8 (model-dependent): ~0.1-0.75 GB per batch
- 6 frames: ~0.3 GB per batch
- 0 workers: 0 MB overhead
- **Frame-by-frame decoding**: ~37 MB per video (only 6 frames loaded)
- **Incremental CSV writing**: Constant memory (no accumulation)
- **One video at a time**: Minimal peak memory
- **Total**: ~1-2 GB per batch + model + overhead = **~5-10 GB** (well within limits)

**Key Breakthrough**: Frame-by-frame decoding eliminates the memory spike from loading entire videos, reducing per-video memory from ~1.87 GB to ~37 MB (50x reduction).

## MLOps Optimizations

### 1. Experiment Tracking & Versioning

**Before**:
- No experiment tracking
- No configuration versioning
- Metrics scattered in logs

**After**:
- **RunConfig**: Centralized configuration with deterministic hashing
- **ExperimentTracker**: Structured metrics logging (JSONL format)
- **Unique Run IDs**: Each experiment gets a unique identifier
- **Configuration Hashing**: Reproducible experiments via config hashes

**Benefits**:
- Reproducibility: Can recreate any experiment from config
- Comparison: Easy to compare different runs
- Audit Trail: Complete history of all experiments

### 2. Pipeline Orchestration

**Before**:
- Linear notebook execution
- No dependency management
- Manual error recovery

**After**:
- **PipelineStage**: Modular stages with dependencies
- **MLOpsPipeline**: Automatic dependency resolution
- **Stage Validation**: Built-in validation for each stage
- **Error Recovery**: Clear error messages with stage context

**Benefits**:
- Modularity: Easy to add/remove stages
- Reliability: Dependencies ensure correct execution order
- Debugging: Clear failure points

### 3. Enhanced Checkpointing

**Before**:
- Only model weights saved
- No resume capability
- No optimizer/scheduler state

**After**:
- **Full State Checkpoints**: Model + optimizer + scheduler + epoch + metrics
- **Resume Capability**: Automatically resume from latest checkpoint
- **Best Model Tracking**: Separate best model checkpoint
- **CheckpointManager**: Centralized checkpoint management

**Benefits**:
- Fault Tolerance: Can resume after crashes
- Time Savings: Don't restart from scratch
- Better Models: Always have best model saved

### 4. Structured Metrics Logging

**Before**:
- Metrics in unstructured logs
- Hard to query/analyze

**After**:
- **JSONL Format**: One metric per line
- **Structured Fields**: step, epoch, phase, metric, value
- **Polars Integration**: Easy DataFrame analysis
- **Best Metric Tracking**: Automatic best value tracking

**Benefits**:
- Analysis: Easy to plot/analyze metrics
- Comparison: Compare metrics across runs
- Monitoring: Track training progress

### 5. Configuration Management

**Before**:
- Config scattered across notebook
- No versioning
- Hard to reproduce

**After**:
- **RunConfig Dataclass**: Single source of truth
- **JSON Serialization**: Human-readable config files
- **Config Hashing**: Deterministic hashing for versioning
- **Metadata Logging**: System info, versions, etc.

**Benefits**:
- Reproducibility: Exact config saved with each run
- Version Control: Can track config changes
- Sharing: Easy to share configs with team

## Pipeline Improvements

### 1. Pandera (Stage 2) - Schema Validation

**Location**: `lib/utils/schemas.py`

**Purpose**: Validate Stage 1 output schema before processing in Stage 2.

**Features**:
- Schema definitions for all pipeline stages (1-5)
- Automatic validation of data integrity between stages
- Graceful fallback if Pandera is not available

**Benefits**:
- Early detection of data corruption or schema mismatches
- Ensures data integrity across pipeline stages
- Prevents downstream errors from invalid data

### 2. MLflow (Stage 5) - Enhanced Experiment Tracking

**Location**: `lib/mlops/mlflow_tracker.py`

**Purpose**: Enhanced experiment tracking with MLflow UI, model registry, and artifact management.

**Features**:
- MLflow integration alongside existing ExperimentTracker
- Automatic logging of metrics, parameters, and models
- Model registry support
- Artifact tracking

**Benefits**:
- Better experiment management with MLflow UI
- Model versioning and registry
- Easy comparison of experiments
- Artifact management

### 3. WebDataset (Stage 5) - Improved Data Loading

**Location**: `lib/models/webdataset_loader.py`

**Purpose**: Efficient data loading for large-scale training using tar archives.

**Features**:
- WebDataset DataLoader creation
- Tar archive creation from video files
- Streaming data access
- Memory-efficient loading

**Benefits**:
- Faster data loading for large datasets
- Reduced I/O overhead
- Better scalability for distributed training
- Efficient storage format

### 4. DuckDB (Stages 2-5) - Analytics Queries

**Location**: `lib/utils/duckdb_analytics.py`

**Purpose**: Fast SQL queries on pipeline data for analytics.

**Features**:
- SQL queries on Polars DataFrames
- Direct querying of Arrow/Parquet files
- Pre-built analytics functions
- Fast aggregations and joins

**Benefits**:
- Fast analytics queries on large datasets
- SQL interface for complex queries
- Direct querying of Arrow/Parquet files
- Pre-built analytics functions

## Summary

These optimizations reduce memory usage by approximately **8-16x** while maintaining:
- Training effectiveness (via increased gradient accumulation)
- Augmentation quality (same augmentations, just fewer frames and lower resolution)
- Reproducibility (deterministic seeds)
- Efficiency (shared augmentations across folds)

The pipeline should now run comfortably within 80 GB memory limits on the SLURM cluster, with typical usage around 5-10 GB during augmentation generation and 10-20 GB during training.
