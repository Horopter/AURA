"""
SlowFast model: Dual-pathway network for video recognition.
"""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _verify_slowfast_model(backbone: nn.Module) -> bool:
    """
    Verify that a model is actually a SlowFast model.
    
    Args:
        backbone: Model to verify
    
    Returns:
        True if model appears to be SlowFast
    """
    model_name = str(type(backbone).__name__).lower()
    model_str = str(backbone).lower()
    
    # Check model structure to verify it's actually SlowFast
    # SlowFast models have dual pathways (pathway0/pathway1 or slow_pathway/fast_pathway)
    is_slowfast = (
        'slowfast' in model_name or 
        'slowfast' in model_str or
        hasattr(backbone, 'pathway0') or 
        hasattr(backbone, 'pathway1') or
        hasattr(backbone, 'slow_pathway') or
        hasattr(backbone, 'fast_pathway') or
        ('slow' in model_str and 'fast' in model_str and 'pathway' in model_str)
    )
    
    if not is_slowfast:
        raise RuntimeError(
            f"CRITICAL: Loaded model does not appear to be SlowFast. "
            f"Model type: {type(backbone).__name__}. "
            f"Model structure: {list(backbone.named_children())[:5] if hasattr(backbone, 'named_children') else 'N/A'}. "
            f"This may indicate a fallback or incorrect model was loaded."
        )
    
    return True


def _replace_classification_head(backbone: nn.Module, strategy: str = "auto") -> bool:
    """
    Replace classification head for binary classification.
    
    Args:
        backbone: Model backbone
        strategy: Strategy to use ("auto", "blocks_proj", "fc", "head", "classifier", "find_last")
    
    Returns:
        True if head was replaced successfully
    """
    head_replaced = False
    
    if strategy == "auto" or strategy == "blocks_proj":
        # Strategy 1: blocks[-1].proj (common PyTorchVideo structure)
        if hasattr(backbone, 'blocks') and len(backbone.blocks) > 0:
            last_block = backbone.blocks[-1]
            if hasattr(last_block, 'proj') and isinstance(last_block.proj, nn.Linear):
                in_features = last_block.proj.in_features
                last_block.proj = nn.Linear(in_features, 1)
                head_replaced = True
                logger.debug("Replaced SlowFast classification head at blocks[-1].proj")
                if strategy != "auto":
                    return head_replaced
    
    if (strategy == "auto" or strategy == "fc") and not head_replaced:
        # Strategy 2: Direct fc attribute (torchvision-style)
        if hasattr(backbone, 'fc'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, 1)
            head_replaced = True
            logger.debug("Replaced SlowFast classification head at fc")
            if strategy != "auto":
                return head_replaced
    
    if (strategy == "auto" or strategy == "head") and not head_replaced:
        # Strategy: head attribute
        if hasattr(backbone, 'head'):
            in_features = backbone.head.in_features
            backbone.head = nn.Linear(in_features, 1)
            head_replaced = True
            logger.debug("Replaced SlowFast classification head at head")
            if strategy != "auto":
                return head_replaced
    
    if (strategy == "auto" or strategy == "classifier") and not head_replaced:
        # Strategy: classifier attribute
        if hasattr(backbone, 'classifier'):
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Linear(in_features, 1)
            head_replaced = True
            logger.debug("Replaced SlowFast classification head at classifier")
            if strategy != "auto":
                return head_replaced
    
    if (strategy == "auto" or strategy == "find_last") and not head_replaced:
        # Strategy 3: Find last Linear layer (fallback)
        last_linear_name = None
        last_linear_in_features = None
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Linear):
                last_linear_name = name
                last_linear_in_features = module.in_features
        
        if last_linear_name is not None:
            parts = last_linear_name.split('.')
            if len(parts) > 1:
                parent = backbone
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], nn.Linear(last_linear_in_features, 1))
            else:
                setattr(backbone, last_linear_name, nn.Linear(last_linear_in_features, 1))
            head_replaced = True
            logger.debug(f"Replaced SlowFast classification head at: {last_linear_name} (fallback)")
            if strategy != "auto":
                return head_replaced
    
    if not head_replaced:
        # Last resort: add new fc layer (may not work correctly)
        logger.warning("Could not find SlowFast classification head, adding new fc layer (may not work correctly)")
        backbone.fc = nn.Linear(2048, 1)
        head_replaced = True
    
    return head_replaced


def _load_torchvision_slowfast(pretrained: bool) -> Optional[nn.Module]:
    """
    Load SlowFast from torchvision.
    
    Args:
        pretrained: Whether to load pretrained weights
    
    Returns:
        SlowFast model or None if not available
    """
    try:
        from torchvision.models.video import slowfast_r50, SlowFast_R50_Weights
        if pretrained:
            try:
                weights = SlowFast_R50_Weights.KINETICS400_V1
                backbone = slowfast_r50(weights=weights)
            except (AttributeError, ValueError):
                backbone = slowfast_r50(pretrained=True)
        else:
            backbone = slowfast_r50(pretrained=False)
        
        # Replace classification head
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        
        # Verify model structure
        _verify_slowfast_model(backbone)
        
        logger.info(f"✓ Verified SlowFast model structure: {type(backbone).__name__}")
        
        # Log model structure for debugging
        if hasattr(backbone, 'pathway0') or hasattr(backbone, 'pathway1'):
            logger.debug("SlowFast model has pathway0/pathway1 (PyTorchVideo structure)")
        if hasattr(backbone, 'slow_pathway') or hasattr(backbone, 'fast_pathway'):
            logger.debug("SlowFast model has slow_pathway/fast_pathway")
        
        return backbone
    except (ImportError, AttributeError):
        return None


def _load_pytorchhub_slowfast(pretrained: bool) -> Optional[nn.Module]:
    """
    Load SlowFast from PyTorch Hub.
    
    Args:
        pretrained: Whether to load pretrained weights
    
    Returns:
        SlowFast model or None if not available
    """
    try:
        import torch.hub
        if pretrained:
            logger.info("Loading SlowFast R50 from PyTorch Hub (facebookresearch/pytorchvideo)...")
            backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        else:
            backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
        
        # Replace classification head
        _replace_classification_head(backbone)
        
        # Verify model structure
        _verify_slowfast_model(backbone)
        
        logger.info(f"✓ Verified SlowFast model structure: {type(backbone).__name__}")
        
        # Log model structure for debugging
        if hasattr(backbone, 'pathway0') or hasattr(backbone, 'pathway1'):
            logger.debug("SlowFast model has pathway0/pathway1 (PyTorch Hub structure)")
        if hasattr(backbone, 'blocks'):
            logger.debug(f"SlowFast model has {len(backbone.blocks)} blocks")
        
        logger.info("✓ Loaded SlowFast from PyTorch Hub (pytorchvideo)")
        return backbone
    except Exception as hub_error:
        logger.debug(f"Failed to load from PyTorch Hub: {hub_error}")
        return None


def _load_pytorchvideo_slowfast(pretrained: bool) -> Optional[nn.Module]:
    """
    Load SlowFast from pytorchvideo library.
    
    Args:
        pretrained: Whether to load pretrained weights
    
    Returns:
        SlowFast model or None if not available
    """
    try:
        import pytorchvideo.models.hub as hub
        from pytorchvideo.models import slowfast
        
        logger.info("Trying to load SlowFast from pytorchvideo...")
        backbone = slowfast.create_slowfast(
            model_num_class=400,  # Kinetics-400 pretrained
            slowfast_fusion_conv_channel_ratio=1.0/8,
            slowfast_conv_channel_fusion_ratio=2,
        )
        
        # Load pretrained weights if available
        if pretrained:
            try:
                checkpoint = hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth"
                )
                backbone.load_state_dict(checkpoint['model_state'], strict=False)
                logger.info("✓ Loaded SlowFast pretrained weights from pytorchvideo")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights: {e}, using random initialization")
        
        # Replace classification head
        _replace_classification_head(backbone)
        
        # Verify model structure
        _verify_slowfast_model(backbone)
        
        logger.info(f"✓ Verified SlowFast model structure: {type(backbone).__name__}")
        
        # Log model structure for debugging
        if hasattr(backbone, 'pathway0') or hasattr(backbone, 'pathway1'):
            logger.debug("SlowFast model has pathway0/pathway1 (create_slowfast structure)")
        if hasattr(backbone, 'blocks'):
            logger.debug(f"SlowFast model has {len(backbone.blocks)} blocks")
        
        logger.info("✓ Loaded SlowFast from pytorchvideo")
        return backbone
    except ImportError:
        logger.debug("pytorchvideo not available")
        return None
    except Exception as pv_error:
        logger.debug(f"Failed to load from pytorchvideo: {pv_error}")
        return None


def _load_huggingface_slowfast(pretrained: bool) -> Optional[nn.Module]:
    """
    Load SlowFast from HuggingFace.
    
    Args:
        pretrained: Whether to load pretrained weights
    
    Returns:
        SlowFast model or None if not available
    """
    try:
        from transformers import AutoModelForVideoClassification, AutoModel
        
        # Try multiple possible SlowFast model names
        model_names = [
            "facebookresearch/slowfast",
            "MCG-NJU/videomae-base-finetuned-kinetics",
        ]
        
        for model_name in model_names:
            try:
                logger.info(f"Trying to load SlowFast from: {model_name}")
                # Try AutoModelForVideoClassification first
                try:
                    model = AutoModelForVideoClassification.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                except (OSError, ValueError, RuntimeError) as auto_model_error:
                    # Fallback to AutoModel
                    logger.debug(f"AutoModelForVideoClassification failed, trying AutoModel: {auto_model_error}")
                    model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                
                # Extract the actual SlowFast backbone
                if hasattr(model, 'slowfast'):
                    backbone = model.slowfast
                elif hasattr(model, 'model'):
                    backbone = model.model
                elif hasattr(model, 'backbone'):
                    backbone = model.backbone
                else:
                    backbone = model
                
                # Replace classification head
                _replace_classification_head(backbone)
                
                # Verify model structure
                _verify_slowfast_model(backbone)
                
                logger.info(f"✓ Verified SlowFast model structure: {type(backbone).__name__}")
                
                # Log model structure for debugging
                if hasattr(backbone, 'pathway0') or hasattr(backbone, 'pathway1'):
                    logger.debug("SlowFast model has pathway0/pathway1 (HuggingFace structure)")
                
                logger.info(f"✓ Loaded SlowFast from {model_name}")
                return backbone
            except Exception as hf_error:
                logger.debug(f"Failed to load from {model_name}: {hf_error}")
                continue
        
        return None
    except ImportError:
        logger.warning("transformers library not available for HuggingFace SlowFast")
        return None
    except Exception as e:
        logger.warning(f"Failed to load SlowFast from HuggingFace: {e}")
        return None


class SlowFastModel(nn.Module):
    """
    SlowFast network for video recognition.
    
    Implements a simplified SlowFast architecture:
    - Slow pathway: processes frames at low temporal rate (2 fps)
    - Fast pathway: processes frames at high temporal rate (8 fps)
    - Fusion: combines features from both pathways
    """
    
    def __init__(
        self,
        slow_frames: int = 16,
        fast_frames: int = 64,
        alpha: int = 8,  # Temporal ratio between fast and slow
        beta: float = 1.0 / 8,  # Channel ratio between fast and slow
        pretrained: bool = True
    ):
        """
        Initialize SlowFast model.
        
        Args:
            slow_frames: Number of frames for slow pathway
            fast_frames: Number of frames for fast pathway
            alpha: Temporal ratio (fast_fps / slow_fps)
            beta: Channel ratio (fast_channels / slow_channels)
            pretrained: Use pretrained weights if available
        """
        super().__init__()
        
        # Initialize flags
        self.use_torchvision = False
        self.use_pytorchvideo = False
        self.use_r3d_fallback = False
        
        # Try loading from different sources in order of preference
        backbone = None
        
        # 1. Try torchvision first (most reliable)
        backbone = _load_torchvision_slowfast(pretrained)
        if backbone is not None:
            self.backbone = backbone
            self.use_torchvision = True
            self.use_pytorchvideo = False
            self.use_r3d_fallback = False
        else:
            # 2. Try PyTorch Hub (recommended for pytorchvideo)
            logger.info("torchvision SlowFast not available. Trying PyTorch Hub (pytorchvideo)...")
            backbone = _load_pytorchhub_slowfast(pretrained)
            if backbone is not None:
                self.backbone = backbone
                self.use_torchvision = False
                self.use_pytorchvideo = True
                self.use_r3d_fallback = False
            else:
                # 3. Try pytorchvideo library directly
                logger.info("Trying pytorchvideo library directly...")
                backbone = _load_pytorchvideo_slowfast(pretrained)
                if backbone is not None:
                    self.backbone = backbone
                    self.use_torchvision = False
                    self.use_pytorchvideo = True
                    self.use_r3d_fallback = False
                else:
                    # 4. Try HuggingFace as last resort
                    logger.info("Trying HuggingFace for SlowFast...")
                    backbone = _load_huggingface_slowfast(pretrained)
                    if backbone is not None:
                        self.backbone = backbone
                        self.use_torchvision = False
                        self.use_pytorchvideo = True
                        self.use_r3d_fallback = False
                    else:
                        # CRITICAL: Do NOT use r3d_18 or simplified SlowFast as fallback
                        raise RuntimeError(
                            f"CRITICAL: Failed to load SlowFast model. "
                            f"Tried torchvision, PyTorch Hub (pytorchvideo), pytorchvideo library, and HuggingFace. "
                            f"SlowFast model is required - please install pytorchvideo: pip install pytorchvideo "
                            f"or ensure torchvision has SlowFast support. "
                            f"Fallback to r3d_18 or simplified SlowFast is disabled to ensure proper model implementation."
                        )
        
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.alpha = alpha
    
    # REMOVED: _build_simplified_slowfast and _make_res_block methods
    # These were fallback implementations that are no longer used.
    # We now require actual SlowFast models from PyTorchVideo, torchvision, or HuggingFace.
    # This ensures proper model implementation and prevents silent fallbacks to approximations.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # CRITICAL: Handle small spatial dimensions like X3D
        # SlowFast also requires minimum spatial dimensions for proper processing
        # NOTE: We UPSCALE small inputs (e.g., 5x8 -> 32x51), NOT downscale large inputs
        # This INCREASES memory usage per sample, so batch size should remain conservative
        if x.dim() == 5:
            N, C, T, H, W = x.shape
            min_spatial_size = 32  # Minimum required for SlowFast (similar to X3D)
            
            # Resize if spatial dimensions are too small (UPSCALE to meet minimum)
            if H < min_spatial_size or W < min_spatial_size:
                # Calculate target size maintaining aspect ratio
                if H <= 0 or W <= 0:
                    new_h = min_spatial_size
                    new_w = min_spatial_size
                else:
                    if H < W:
                        scale_factor = min_spatial_size / max(H, 1.0)
                        new_h = min_spatial_size
                        new_w = max(min_spatial_size, int(W * scale_factor))
                    else:
                        scale_factor = min_spatial_size / max(W, 1.0)
                        new_w = min_spatial_size
                        new_h = max(min_spatial_size, int(H * scale_factor))
                
                new_h = max(new_h, min_spatial_size)
                new_w = max(new_w, min_spatial_size)
                
                # Resize: (N, C, T, H, W) -> (N*T, C, H, W) -> resize -> (N, C, T, H', W')
                x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
                x_reshaped = x_reshaped.view(N * T, C, H, W)  # (N*T, C, H, W)
                x_resized = F.interpolate(
                    x_reshaped, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )  # (N*T, C, H', W')
                x_resized = x_resized.view(N, T, C, new_h, new_w)  # (N, T, C, H', W')
                x = x_resized.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H', W')
                
                if H < 16 or W < 16:
                    logger.debug(
                        f"SlowFast: Resized input from {H}x{W} to {new_h}x{new_w} "
                        f"(temporal: {T} frames) to meet minimum spatial dimension requirements"
                    )
        
        # CRITICAL: Ensure we're not using fallback implementations
        if self.use_r3d_fallback:
            raise RuntimeError(
                "CRITICAL: SlowFast is using r3d_18 fallback. This should not happen - "
                "proper SlowFast model should be loaded. Check model initialization."
            )
        
        if self.use_torchvision:
            # Use torchvision's SlowFast
            return self.backbone(x)
        
        # PyTorchVideo SlowFast expects list of tensors [slow_pathway, fast_pathway]
        # CRITICAL: PyTorchVideo SlowFast has specific temporal dimension requirements
        # The model's internal architecture performs temporal downsampling, and the fusion
        # layer requires compatible temporal dimensions after processing
        # 
        # The error "Expected size 63 but got size 125" indicates that after processing,
        # the slow and fast pathways have incompatible temporal dimensions for fusion.
        # This happens when the input temporal dimension T doesn't result in compatible
        # dimensions after the model's internal temporal downsampling.
        #
        # Solution: PyTorchVideo SlowFast typically expects input temporal dimensions
        # that are multiples of specific values. We need to ensure T results in
        # compatible dimensions after processing. Common compatible values are
        # multiples of (alpha * temporal_downsample_factor), where temporal_downsample_factor
        # is typically 2, 4, or 8 depending on the model architecture.
        if self.use_pytorchvideo:
            N, C, T, H, W = x.shape
            
            # PyTorchVideo SlowFast R50 typically expects temporal dimensions that result
            # in compatible dimensions after processing. The model has temporal stride 2
            # in multiple layers, so we need T such that after downsampling, both pathways align.
            # 
            # For alpha=8, common compatible T values are multiples of 32 or 64
            # (e.g., 32, 64, 96, 128, etc.) to ensure proper alignment after downsampling.
            #
            # However, since the user has provided proper scaled videos, we should work
            # with the given T. The issue might be that we need to ensure the slow pathway
            # sampling exactly matches what PyTorchVideo expects.
            
            # Slow pathway: sample every alpha frames starting from frame 0
            # This is the standard SlowFast sampling: slow pathway gets every alpha-th frame
            slow_indices = torch.arange(0, T, self.alpha, device=x.device, dtype=torch.long)
            # Ensure indices are within bounds
            slow_indices = slow_indices[slow_indices < T]
            
            # If we don't have enough frames for slow pathway, use first frame
            if len(slow_indices) == 0:
                slow_indices = torch.tensor([0], device=x.device, dtype=torch.long)
                logger.warning(
                    f"SlowFast: Input temporal dimension T={T} is less than alpha={self.alpha}. "
                    f"Using single frame for slow pathway."
                )
            
            slow_x = x[:, :, slow_indices, :, :]  # (N, C, T_slow, H, W)
            # Fast pathway: use all frames (standard SlowFast: fast pathway gets all frames)
            fast_x = x  # (N, C, T, H, W)
            
            # PyTorchVideo SlowFast processes each pathway through separate networks
            # with temporal downsampling, then fuses them. The fusion requires compatible
            # temporal dimensions. If T doesn't result in compatible dimensions, we may
            # need to adjust the input or handle it differently.
            #
            # Try the standard approach first - PyTorchVideo should handle most cases
            try:
                return self.backbone([slow_x, fast_x])
            except RuntimeError as e:
                error_msg = str(e)
                if "sizes of tensors must match" in error_msg.lower():
                    # Temporal dimension mismatch - the input T doesn't result in compatible
                    # dimensions after processing. This can happen with certain T values.
                    # 
                    # Solution: Try adjusting the temporal dimension to a compatible value
                    # by padding or truncating. However, since the user wants the architecture
                    # to accommodate their videos, we should try to make it work with the given T.
                    #
                    # Alternative: The issue might be that PyTorchVideo SlowFast expects
                    # a specific input format. Let's try using the model's expected input
                    # format more directly.
                    logger.warning(
                        f"SlowFast temporal dimension mismatch with T={T}, alpha={self.alpha}. "
                        f"Slow pathway T: {slow_x.shape[2]}, Fast pathway T: {fast_x.shape[2]}. "
                        f"Attempting to adjust input temporal dimension for compatibility..."
                    )
                    
                    # Try padding the temporal dimension to a compatible value
                    # Common compatible values for alpha=8 are multiples of 32
                    # Pad to nearest multiple of 32 that's >= T
                    target_T = ((T + 31) // 32) * 32
                    if target_T > T:
                        # Pad the last frame to reach target_T
                        padding_frames = target_T - T
                        last_frame = x[:, :, -1:, :, :]  # (N, C, 1, H, W)
                        padding = last_frame.repeat(1, 1, padding_frames, 1, 1)
                        x_padded = torch.cat([x, padding], dim=2)  # (N, C, target_T, H, W)
                        
                        # Recompute slow pathway with padded input
                        slow_indices_padded = torch.arange(0, target_T, self.alpha, device=x.device, dtype=torch.long)
                        slow_indices_padded = slow_indices_padded[slow_indices_padded < target_T]
                        slow_x_padded = x_padded[:, :, slow_indices_padded, :, :]
                        fast_x_padded = x_padded
                        
                        try:
                            return self.backbone([slow_x_padded, fast_x_padded])
                        except RuntimeError as e2:
                            # Padding didn't work, re-raise original error
                            logger.error(
                                f"SlowFast temporal dimension adjustment failed: {e2}. "
                                f"Original error: {error_msg}. "
                                f"Input shape: (N={N}, C={C}, T={T}, H={H}, W={W})"
                            )
                            raise e
                    else:
                        # T is already a compatible value, re-raise original error
                        raise e
                else:
                    # Not a temporal dimension mismatch, re-raise
                    raise e
        
        # Fallback: try list input if tensor input fails (for models that weren't detected)
        try:
            return self.backbone(x)
        except (AssertionError, TypeError, RuntimeError) as e:
            error_msg = str(e).lower()
            if "list" in error_msg or "multipathway" in error_msg:
                # Model needs list input - split into slow/fast pathways
                N, C, T, H, W = x.shape
                # Use proper indexing to ensure valid frame counts
                slow_indices = torch.arange(0, T, self.alpha, device=x.device, dtype=torch.long)
                slow_indices = slow_indices[slow_indices < T]
                if len(slow_indices) == 0:
                    slow_indices = torch.tensor([0], device=x.device, dtype=torch.long)
                slow_x = x[:, :, slow_indices, :, :]
                fast_x = x
                return self.backbone([slow_x, fast_x])
            raise
        
        # If we reach here, something went wrong - we should have handled all cases above
        raise RuntimeError(
            "CRITICAL: SlowFast forward pass reached unreachable code. "
            "This indicates the model was not properly initialized. "
            "Check that use_torchvision, use_pytorchvideo, or proper fallback was set."
        )


__all__ = ["SlowFastModel"]

