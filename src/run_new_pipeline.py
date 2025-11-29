#!/usr/bin/env python3
"""
New 5-Stage Pipeline Runner

Stage 1: Augment videos (10 augmentations per video) → 11N videos
Stage 2: Extract handcrafted features from all 11N videos → M features
Stage 3: Downscale videos → 11N downscaled videos
Stage 4: Extract additional features from downscaled videos → P features
Stage 5: Train models using downscaled videos + M + P features
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.augmentation import stage1_augment_videos
from lib.features import stage2_extract_features, stage4_extract_downscaled_features
from lib.downscaling import stage3_downscale_videos
from lib.training import stage5_train_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run 5-stage pipeline")
    parser.add_argument("--project-root", type=str, default=os.getcwd(), help="Project root directory")
    parser.add_argument("--num-augmentations", type=int, default=10, help="Number of augmentations per video")
    parser.add_argument("--skip-stage", type=int, nargs="+", default=[], help="Stages to skip (1-5)")
    parser.add_argument("--only-stage", type=int, nargs="+", default=[], help="Only run these stages (1-5)")
    parser.add_argument("--model-types", type=str, nargs="+", default=["logistic_regression", "svm"], help="Models to train in Stage 5")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of k-fold splits")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    
    logger.info("="*80)
    logger.info("NEW 5-STAGE PIPELINE")
    logger.info("="*80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Number of augmentations: {args.num_augmentations}")
    logger.info(f"Skip stages: {args.skip_stage}")
    logger.info(f"Only stages: {args.only_stage}")
    logger.info("="*80)
    
    # Determine which stages to run
    stages_to_run = []
    if args.only_stage:
        stages_to_run = args.only_stage
    else:
        stages_to_run = [1, 2, 3, 4, 5]
        stages_to_run = [s for s in stages_to_run if s not in args.skip_stage]
    
    # Stage 1: Augmentation
    if 1 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: VIDEO AUGMENTATION")
        logger.info("="*80)
        stage1_df = stage1_augment_videos(
            project_root=str(project_root),
            num_augmentations=args.num_augmentations,
            output_dir="data/augmented_videos"
        )
        stage1_metadata_path = project_root / "data" / "augmented_videos" / "augmented_metadata.csv"
    else:
        logger.info("Skipping Stage 1")
        stage1_metadata_path = project_root / "data" / "augmented_videos" / "augmented_metadata.csv"
        if not stage1_metadata_path.exists():
            logger.error(f"Stage 1 metadata not found: {stage1_metadata_path}")
            return
    
    # Stage 2: Extract features from original videos
    if 2 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: EXTRACT HANDCRAFTED FEATURES (M features)")
        logger.info("="*80)
        stage2_df = stage2_extract_features(
            project_root=str(project_root),
            augmented_metadata_path=str(stage1_metadata_path),
            num_frames=8,
            output_dir="data/features_stage2"
        )
        stage2_metadata_path = project_root / "data" / "features_stage2" / "features_metadata.csv"
    else:
        logger.info("Skipping Stage 2")
        stage2_metadata_path = project_root / "data" / "features_stage2" / "features_metadata.csv"
        if not stage2_metadata_path.exists():
            logger.error(f"Stage 2 metadata not found: {stage2_metadata_path}")
            return
    
    # Stage 3: Downscale videos
    if 3 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: DOWNSCALE VIDEOS")
        logger.info("="*80)
        stage3_df = stage3_downscale_videos(
            project_root=str(project_root),
            augmented_metadata_path=str(stage1_metadata_path),
            output_dir="data/downscaled_videos",
            method="resolution",  # or "autoencoder"
            target_size=(224, 224)
        )
        stage3_metadata_path = project_root / "data" / "downscaled_videos" / "downscaled_metadata.csv"
    else:
        logger.info("Skipping Stage 3")
        stage3_metadata_path = project_root / "data" / "downscaled_videos" / "downscaled_metadata.csv"
        if not stage3_metadata_path.exists():
            logger.error(f"Stage 3 metadata not found: {stage3_metadata_path}")
            return
    
    # Stage 4: Extract features from downscaled videos
    if 4 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: EXTRACT FEATURES FROM DOWNSCALED VIDEOS (P features)")
        logger.info("="*80)
        stage4_df = stage4_extract_downscaled_features(
            project_root=str(project_root),
            downscaled_metadata_path=str(stage3_metadata_path),
            num_frames=8,
            output_dir="data/features_stage4"
        )
        stage4_metadata_path = project_root / "data" / "features_stage4" / "features_downscaled_metadata.csv"
    else:
        logger.info("Skipping Stage 4")
        stage4_metadata_path = project_root / "data" / "features_stage4" / "features_downscaled_metadata.csv"
        if not stage4_metadata_path.exists():
            logger.error(f"Stage 4 metadata not found: {stage4_metadata_path}")
            return
    
    # Stage 5: Training
    if 5 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: TRAIN MODELS")
        logger.info("="*80)
        results = stage5_train_models(
            project_root=str(project_root),
            downscaled_metadata_path=str(stage3_metadata_path),
            features_stage2_path=str(stage2_metadata_path),
            features_stage4_path=str(stage4_metadata_path),
            model_types=args.model_types,
            n_splits=args.n_splits,
            num_frames=8,
            output_dir="data/training_results"
        )
        logger.info("✓ Stage 5 complete: Training finished")
    else:
        logger.info("Skipping Stage 5")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

