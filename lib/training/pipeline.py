"""
Model training pipeline.

Trains models using scaled videos and extracted features.
Supports multiple model types and k-fold cross-validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional
import polars as pl
import torch
from torch.utils.data import DataLoader

from lib.data import stratified_kfold, load_metadata
from lib.models import VideoConfig, VideoDataset
from lib.mlops.config import ExperimentTracker, CheckpointManager
from lib.training.trainer import OptimConfig, TrainConfig, fit
from lib.training.model_factory import create_model, is_pytorch_model, get_model_config
from lib.training.feature_preprocessing import remove_collinear_features, load_and_combine_features
from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


def stage5_train_models(
    project_root: str,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: str,
    model_types: List[str],
    n_splits: int = 5,
    num_frames: int = 8,
    output_dir: str = "data/training_results",
    use_tracking: bool = True,
    train_ensemble: bool = False,
    ensemble_method: str = "meta_learner"
) -> Dict:
    """
    Stage 5: Train models using scaled videos and features.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled metadata (from Stage 3)
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        model_types: List of model types to train
        n_splits: Number of k-fold splits
        num_frames: Number of frames per video
        output_dir: Directory to save training results
        use_tracking: Whether to use experiment tracking
        train_ensemble: Whether to train ensemble model after individual models (default: False)
        ensemble_method: Ensemble method - "meta_learner" or "weighted_average" (default: "meta_learner")
    
    Returns:
        Dictionary of training results
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata (support both CSV and Arrow/Parquet)
    logger.info("Stage 5: Loading metadata...")
    
    from lib.utils.paths import load_metadata_flexible
    
    scaled_df = load_metadata_flexible(scaled_metadata_path)
    if scaled_df is None:
        raise FileNotFoundError(f"Scaled metadata not found: {scaled_metadata_path}")
    
    features2_df = load_metadata_flexible(features_stage2_path)
    features4_df = load_metadata_flexible(features_stage4_path)
    
    logger.info(f"Stage 5: Found {scaled_df.height} scaled videos")
    
    # Create video config
    video_config = VideoConfig(
        num_frames=num_frames,
        fixed_size=224,
    )
    
    results = {}
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 5: Training model: {model_type}")
        logger.info(f"{'='*80}")
        
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model config
        model_config = get_model_config(model_type)
        
        # K-fold cross-validation
        fold_results = []
        
        # CRITICAL FIX: Get all folds at once (stratified_kfold returns list of all folds)
        all_folds = stratified_kfold(
            scaled_df,
            n_splits=n_splits,
            random_state=42
        )
        
        if len(all_folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds, got {len(all_folds)}")
        
        for fold_idx in range(n_splits):
            logger.info(f"\nTraining {model_type} - Fold {fold_idx + 1}/{n_splits}")
            
            # Get the specific fold
            train_df, val_df = all_folds[fold_idx]
            
            # Validate no data leakage (check dup_group if present)
            if "dup_group" in scaled_df.columns:
                train_groups = set(train_df["dup_group"].unique().to_list())
                val_groups = set(val_df["dup_group"].unique().to_list())
                overlap = train_groups & val_groups
                if overlap:
                    logger.error(
                        f"CRITICAL: Data leakage detected in fold {fold_idx + 1}! "
                        f"{len(overlap)} duplicate groups appear in both train and val: {list(overlap)[:5]}"
                    )
                    raise ValueError(f"Data leakage: duplicate groups in both train and val sets")
                logger.info(f"Fold {fold_idx + 1}: No data leakage (checked {len(train_groups)} train groups, {len(val_groups)} val groups)")
            
            # Create datasets
            train_dataset = VideoDataset(
                train_df,
                project_root=str(project_root),
                config=video_config,
            )
            val_dataset = VideoDataset(
                val_df,
                project_root=str(project_root),
                config=video_config,
            )
            
            # Create data loaders
            # GPU-optimized DataLoader settings
            use_cuda = torch.cuda.is_available()
            num_workers = model_config.get("num_workers", 0)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_config.get("batch_size", 8),
                shuffle=True,
                num_workers=num_workers,
                pin_memory=use_cuda,  # Faster GPU transfer
                persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=model_config.get("batch_size", 8),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_cuda,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            
            # Train model
            if is_pytorch_model(model_type):
                # PyTorch model training
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = create_model(model_type, model_config)
                model = model.to(device)
                
                # Create optimizer and scheduler with ML best practices
                optim_cfg = OptimConfig(
                    lr=model_config.get("learning_rate", 1e-4),
                    weight_decay=model_config.get("weight_decay", 1e-4),
                    max_grad_norm=model_config.get("max_grad_norm", 1.0),  # Gradient clipping
                    # Use differential LR for pretrained models
                    backbone_lr=model_config.get("backbone_lr", None),
                    head_lr=model_config.get("head_lr", None),
                )
                train_cfg = TrainConfig(
                    num_epochs=model_config.get("num_epochs", 20),
                    device=str(device),
                    use_amp=model_config.get("use_amp", True),
                    gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 1),
                    early_stopping_patience=model_config.get("early_stopping_patience", 5),
                    scheduler_type=model_config.get("scheduler_type", "cosine"),  # Better than StepLR
                    warmup_epochs=model_config.get("warmup_epochs", 2),  # LR warmup
                    warmup_factor=model_config.get("warmup_factor", 0.1),
                    log_grad_norm=model_config.get("log_grad_norm", False),  # Debug gradient norms
                )
                
                # Determine if we should use differential LR (for pretrained models)
                use_differential_lr = model_type in [
                    "i3d", "r2plus1d", "slowfast", "x3d", "pretrained_inception",
                    "vit_gru", "vit_transformer"
                ]
                
                # Create tracker and checkpoint manager
                fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)
                
                if use_tracking:
                    tracker = ExperimentTracker(str(fold_output_dir))
                    ckpt_manager = CheckpointManager(str(fold_output_dir))
                else:
                    tracker = None
                    ckpt_manager = None
                
                logger.info(f"Training PyTorch model {model_type} on fold {fold_idx + 1}...")
                
                # Train
                try:
                    model = fit(
                        model,
                        train_loader,
                        val_loader,
                        optim_cfg,
                        train_cfg,
                        use_differential_lr=use_differential_lr,  # Use differential LR for pretrained models
                    )
                    
                    # Evaluate final model
                    from lib.training.trainer import evaluate
                    val_loss, val_acc = evaluate(model, val_loader, device=str(device))
                    
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    
                    logger.info(f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Save model for ensemble training
                    model.eval()
                    model_path = fold_output_dir / "model.pt"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved model to {model_path}")
                    
                except Exception as e:
                    logger.error(f"Error training fold {fold_idx + 1}: {e}", exc_info=True)
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": float('nan'),
                        "val_acc": float('nan'),
                    })
                
                # Clear model and aggressively free memory
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                aggressive_gc(clear_cuda=False)
            else:
                # Baseline model training (sklearn)
                logger.warning(f"Baseline model training not yet implemented in stage5")
                fold_results.append({
                    "fold": fold_idx + 1,
                    "val_loss": float('nan'),
                    "val_acc": float('nan'),
                })
        
        # Aggregate results
        if fold_results:
            avg_val_loss = sum(r["val_loss"] for r in fold_results if not (isinstance(r["val_loss"], float) and (r["val_loss"] != r["val_loss"]))) / len([r for r in fold_results if not (isinstance(r["val_loss"], float) and (r["val_loss"] != r["val_loss"]))])
            avg_val_acc = sum(r["val_acc"] for r in fold_results if not (isinstance(r["val_acc"], float) and (r["val_acc"] != r["val_acc"]))) / len([r for r in fold_results if not (isinstance(r["val_acc"], float) and (r["val_acc"] != r["val_acc"]))])
            
            results[model_type] = {
                "fold_results": fold_results,
                "avg_val_loss": avg_val_loss,
                "avg_val_acc": avg_val_acc,
            }
            
            logger.info(f"\n{model_type} - Avg Val Loss: {avg_val_loss:.4f}, Avg Val Acc: {avg_val_acc:.4f}")
        
        # Aggressive GC after all folds for this model type
        aggressive_gc(clear_cuda=False)
    
    # Train ensemble if requested
    if train_ensemble:
        logger.info("\n" + "="*80)
        logger.info("Training Ensemble Model")
        logger.info("="*80)
        
        try:
            from .ensemble import train_ensemble_model
            
            ensemble_results = train_ensemble_model(
                project_root=str(project_root),
                scaled_metadata_path=scaled_metadata_path,
                base_model_types=model_types,
                base_models_dir=str(output_dir),
                n_splits=n_splits,
                num_frames=num_frames,
                output_dir=str(output_dir),
                ensemble_method=ensemble_method
            )
            
            results["ensemble"] = ensemble_results
            logger.info("✓ Ensemble training completed")
        except Exception as e:
            logger.error(f"Error training ensemble: {e}", exc_info=True)
            logger.warning("Continuing without ensemble results")
    
    return results

