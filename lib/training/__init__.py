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

# Core training utilities
from .trainer import (
    OptimConfig,
    TrainConfig,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    evaluate,
    fit,
)

# Models - feature-based models (always available)
from ._linear import LogisticRegressionBaseline
from ._svm import SVMBaseline
from ._cnn import NaiveCNNBaseline
from ._xgboost_pretrained import XGBoostPretrainedBaseline

# Video-based models - lazy imports to avoid loading heavy dependencies when not needed
# These are imported on-demand via model_factory.create_model()
# Importing them here would cause issues when only using sklearn models
def _lazy_import_video_models():
    """Lazy import for video models - only called when needed."""
    from ._transformer_gru import ViTGRUModel
    from ._transformer import ViTTransformerModel
    from .slowfast import SlowFastModel
    from .x3d import X3DModel
    from .i3d import I3DModel
    from .r2plus1d import R2Plus1DModel
    from .timesformer import TimeSformerModel
    from .vivit import ViViTModel
    from .two_stream import TwoStreamModel
    from .slowfast_advanced import SlowFastAttentionModel, MultiScaleSlowFastModel
    return {
        "ViTGRUModel": ViTGRUModel,
        "ViTTransformerModel": ViTTransformerModel,
        "SlowFastModel": SlowFastModel,
        "X3DModel": X3DModel,
        "I3DModel": I3DModel,
        "R2Plus1DModel": R2Plus1DModel,
        "TimeSformerModel": TimeSformerModel,
        "ViViTModel": ViViTModel,
        "TwoStreamModel": TwoStreamModel,
        "SlowFastAttentionModel": SlowFastAttentionModel,
        "MultiScaleSlowFastModel": MultiScaleSlowFastModel,
    }

# For backward compatibility, provide lazy accessors
def __getattr__(name):
    """Lazy import for video models."""
    video_models = {
        "ViTGRUModel", "ViTTransformerModel", "SlowFastModel", "X3DModel",
        "I3DModel", "R2Plus1DModel", "TimeSformerModel", "ViViTModel",
        "TwoStreamModel", "SlowFastAttentionModel", "MultiScaleSlowFastModel"
    }
    if name in video_models:
        models = _lazy_import_video_models()
        return models[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Model factory
from .model_factory import (
    create_model,
    get_model_config,
    is_pytorch_model,
    list_available_models,
    download_pretrained_models,
    MODEL_MEMORY_CONFIGS,
)

# Training pipeline
from .pipeline import stage5_train_models

# Feature preprocessing
from .feature_preprocessing import remove_collinear_features, load_and_combine_features

# Additional utilities
from .trainer import (
    EarlyStopping,
    freeze_all,
    unfreeze_all,
    freeze_backbone_unfreeze_head,
    trainable_params,
    compute_class_counts,
    make_class_weights,
    make_weighted_sampler,
)

__all__ = [
    # Core
    "OptimConfig",
    "TrainConfig",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "fit",
    # Models
    "LogisticRegressionBaseline",
    "SVMBaseline",
    "NaiveCNNBaseline",
    "ViTGRUModel",
    "ViTTransformerModel",
    "SlowFastModel",
    "X3DModel",
    "I3DModel",
    "R2Plus1DModel",
    "XGBoostPretrainedBaseline",
    # Future models
    "TimeSformerModel",
    "ViViTModel",
    "TwoStreamModel",
    "SlowFastAttentionModel",
    "MultiScaleSlowFastModel",
    # Factory
    "create_model",
    "get_model_config",
    "is_pytorch_model",
    "is_xgboost_model",
    "list_available_models",
    "download_pretrained_models",
    "MODEL_MEMORY_CONFIGS",
    # Stage 5
    "stage5_train_models",
    # Feature preprocessing
    "remove_collinear_features",
    "load_and_combine_features",
    # Utilities
    "EarlyStopping",
    "freeze_all",
    "unfreeze_all",
    "freeze_backbone_unfreeze_head",
    "trainable_params",
    "compute_class_counts",
    "make_class_weights",
    "make_weighted_sampler",
]

