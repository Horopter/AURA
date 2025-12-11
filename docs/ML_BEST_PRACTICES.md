# ML/MLOps Best Practices Implementation

This document describes the comprehensive ML/MLOps improvements implemented to address common training issues including gradient vanishing/exploding, batch normalization, dropout, optimizer settings, and other best practices.

## Overview

The following improvements have been implemented across the training pipeline:

1. **Gradient Clipping** - Prevents gradient explosion
2. **Learning Rate Scheduling** - Warmup + Cosine Annealing
3. **Differential Learning Rates** - Different LRs for pretrained backbone vs head
4. **Batch Normalization** - Proper train/eval mode handling
5. **Dropout** - Properly disabled during evaluation
6. **Weight Initialization** - Proper initialization for custom models
7. **Gradient Monitoring** - Logging gradient norms for debugging

## 1. Gradient Clipping

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

## 2. Learning Rate Scheduling

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

## 3. Differential Learning Rates

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

## 4. Batch Normalization

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

## 5. Dropout

**Problem**: Dropout must be disabled during evaluation.

**Solution**: Automatic via `model.eval()` mode.

```python
# All models with dropout:
# - NaiveCNNBaseline: dropout=0.5
# - ViTGRUModel: dropout=0.5
# - ViTTransformerModel: dropout=0.5
# - EnsembleMetaLearner: dropout=0.3

# During evaluation
model.eval()  # Automatically disables all dropout layers
```

**Benefits**:
- No dropout during inference (correct predictions)
- Dropout active during training (regularization)

## 6. Weight Initialization

**Problem**: Random initialization can lead to poor convergence.

**Solution**: Proper initialization for custom models.

```python
# In NaiveCNNBaseline
def _initialize_weights(self):
    """Initialize model weights using He initialization for ReLU activations."""
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
```

**Benefits**:
- He initialization for ReLU activations (prevents vanishing gradients)
- Proper BatchNorm initialization
- Faster convergence

## 7. Gradient Monitoring

**Problem**: Hard to debug gradient issues without visibility.

**Solution**: Optional gradient norm logging.

```python
# In TrainConfig
log_grad_norm: bool = False  # Enable for debugging

# During training (if enabled)
if log_grad_norm and (batch_idx + 1) % log_interval == 0:
    logger.info(f"Gradient norm: {grad_norm:.4f} (clipped to {max_grad_norm})")
```

**Benefits**:
- Debug gradient issues
- Monitor training stability
- Verify gradient clipping is working

## 8. Optimizer Improvements

**Changed from Adam to AdamW**:

```python
# Old: Adam
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# New: AdamW (decoupled weight decay)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

**Benefits**:
- **AdamW**: Decouples weight decay from gradient-based updates
- Better generalization
- More stable training

## 9. Early Stopping

**Already implemented**, but now properly integrated:

```python
# In TrainConfig
early_stopping_patience: int = 5

# Automatically stops training if validation accuracy doesn't improve
# Restores best model state at the end
```

## Configuration Example

```python
from lib.training.trainer import OptimConfig, TrainConfig

# Optimizer config with gradient clipping
optim_cfg = OptimConfig(
    lr=1e-4,
    weight_decay=1e-4,
    max_grad_norm=1.0,  # Gradient clipping
    backbone_lr=1e-5,   # Lower LR for pretrained backbone
    head_lr=1e-3,       # Higher LR for new head
)

# Training config with warmup + cosine annealing
train_cfg = TrainConfig(
    num_epochs=20,
    scheduler_type="cosine",  # Better than StepLR
    warmup_epochs=2,          # LR warmup
    warmup_factor=0.1,         # Start at 10% of base LR
    early_stopping_patience=5,
    log_grad_norm=False,       # Enable for debugging
)
```

## Model-Specific Notes

### Pretrained Models (I3D, R2+1D, SlowFast, X3D)
- **Differential LR**: Automatically applied
- **BatchNorm**: Handled by torchvision backbones
- **Dropout**: Not used (rely on pretrained features)

### Custom Models (NaiveCNN, ViT-GRU, ViT-Transformer)
- **Weight Initialization**: Properly initialized
- **BatchNorm**: Explicitly added where needed
- **Dropout**: Used for regularization

### Baseline Models (Logistic Regression, SVM)
- **No gradients**: Not applicable
- **Feature preprocessing**: Collinearity removal applied

## Best Practices Checklist

✅ **Gradient Clipping**: Prevents explosion  
✅ **LR Warmup**: Prevents early instability  
✅ **Cosine Annealing**: Better convergence  
✅ **Differential LR**: Better transfer learning  
✅ **BatchNorm**: Proper train/eval mode  
✅ **Dropout**: Disabled during eval  
✅ **Weight Init**: Proper initialization  
✅ **Gradient Monitoring**: Debugging support  
✅ **AdamW**: Better weight decay  
✅ **Early Stopping**: Prevents overfitting  

## References

- [Gradient Clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [AdamW Optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [Cosine Annealing LR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
- [Batch Normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [He Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_)

