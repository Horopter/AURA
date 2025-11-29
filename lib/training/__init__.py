"""
Model training module.

Provides:
- Baseline models (Logistic Regression, SVM, Naive CNN)
- Frame-temporal models (ViT+GRU, ViT+Transformer)
- Spatiotemporal models (SlowFast, X3D)
- Training utilities and loops
- Model factory
- Stage 5: Training pipeline
"""

from .baseline_models import (
    LogisticRegressionBaseline,
    SVMBaseline,
    NaiveCNNBaseline,
)
from .frame_temporal_models import (
    ViTGRUModel,
    ViTTransformerModel,
)
from .spatiotemporal_models import (
    SlowFastModel,
    X3DModel,
)
from .model_factory import (
    create_model,
    get_model_config,
    is_pytorch_model,
    list_available_models,
    download_pretrained_models,
    MODEL_MEMORY_CONFIGS,
)
from .video_training import (
    OptimConfig,
    TrainConfig,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    evaluate,
    fit,
)
from .pipeline_stage5_training import stage5_train_models

__all__ = [
    "LogisticRegressionBaseline",
    "SVMBaseline",
    "NaiveCNNBaseline",
    "ViTGRUModel",
    "ViTTransformerModel",
    "SlowFastModel",
    "X3DModel",
    "create_model",
    "get_model_config",
    "is_pytorch_model",
    "list_available_models",
    "download_pretrained_models",
    "MODEL_MEMORY_CONFIGS",
    "OptimConfig",
    "TrainConfig",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "fit",
    "stage5_train_models",
]

