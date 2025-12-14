"""
SlowFast model: Dual-pathway network for video recognition.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        
        # Try to use torchvision's SlowFast if available
        try:
            from torchvision.models.video import slowfast_r50, SlowFast_R50_Weights
            if pretrained:
                try:
                    weights = SlowFast_R50_Weights.KINETICS400_V1
                    self.backbone = slowfast_r50(weights=weights)
                except (AttributeError, ValueError):
                    self.backbone = slowfast_r50(pretrained=True)
            else:
                self.backbone = slowfast_r50(pretrained=False)
            
            # Replace classification head for binary classification
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = True
            self.use_pytorchvideo = False
            self.use_r3d_fallback = False
            
            # Verify we actually have a SlowFast model
            model_name = str(type(self.backbone).__name__).lower()
            model_str = str(self.backbone).lower()
            
            # Check model structure to verify it's actually SlowFast
            # SlowFast models have dual pathways (pathway0/pathway1 or slow_pathway/fast_pathway)
            is_slowfast = (
                'slowfast' in model_name or 
                'slowfast' in model_str or
                hasattr(self.backbone, 'pathway0') or 
                hasattr(self.backbone, 'pathway1') or
                hasattr(self.backbone, 'slow_pathway') or
                hasattr(self.backbone, 'fast_pathway') or
                ('slow' in model_str and 'fast' in model_str and 'pathway' in model_str)
            )
            
            if not is_slowfast:
                raise RuntimeError(
                    f"CRITICAL: Loaded model does not appear to be SlowFast. "
                    f"Model type: {type(self.backbone).__name__}. "
                    f"Model structure: {list(self.backbone.named_children())[:5] if hasattr(self.backbone, 'named_children') else 'N/A'}. "
                    f"This may indicate a fallback or incorrect model was loaded."
                )
            
            logger.info(f"✓ Verified SlowFast model structure: {type(self.backbone).__name__}")
            
            # Log model structure for debugging
            if hasattr(self.backbone, 'pathway0') or hasattr(self.backbone, 'pathway1'):
                logger.debug("SlowFast model has pathway0/pathway1 (PyTorchVideo structure)")
            if hasattr(self.backbone, 'slow_pathway') or hasattr(self.backbone, 'fast_pathway'):
                logger.debug("SlowFast model has slow_pathway/fast_pathway")
            
        except (ImportError, AttributeError):
            # Try PyTorch Hub (recommended method for pytorchvideo models)
            logger.info("torchvision SlowFast not available. Trying PyTorch Hub (pytorchvideo)...")
            hub_loaded = False
            try:
                # PyTorch Hub is the recommended way to load pytorchvideo models
                import torch.hub
                if pretrained:
                    logger.info("Loading SlowFast R50 from PyTorch Hub (facebookresearch/pytorchvideo)...")
                    self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
                else:
                    self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
                
                # Replace classification head for binary classification
                # PyTorchVideo SlowFast structure varies - try multiple strategies
                head_replaced = False
                
                # Strategy 1: blocks[-1].proj (common PyTorchVideo structure)
                if hasattr(self.backbone, 'blocks') and len(self.backbone.blocks) > 0:
                    last_block = self.backbone.blocks[-1]
                    if hasattr(last_block, 'proj') and isinstance(last_block.proj, nn.Linear):
                        in_features = last_block.proj.in_features
                        last_block.proj = nn.Linear(in_features, 1)
                        head_replaced = True
                        logger.debug("Replaced SlowFast classification head at blocks[-1].proj")
                
                # Strategy 2: Direct fc attribute (torchvision-style)
                if not head_replaced and hasattr(self.backbone, 'fc'):
                    in_features = self.backbone.fc.in_features
                    self.backbone.fc = nn.Linear(in_features, 1)
                    head_replaced = True
                    logger.debug("Replaced SlowFast classification head at fc")
                
                # Strategy 3: Find last Linear layer (fallback)
                if not head_replaced:
                    last_linear_name = None
                    last_linear_in_features = None
                    for name, module in self.backbone.named_modules():
                        if isinstance(module, nn.Linear):
                            last_linear_name = name
                            last_linear_in_features = module.in_features
                    
                    if last_linear_name is not None:
                        parts = last_linear_name.split('.')
                        if len(parts) > 1:
                            parent = self.backbone
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            setattr(parent, parts[-1], nn.Linear(last_linear_in_features, 1))
                        else:
                            setattr(self.backbone, last_linear_name, nn.Linear(last_linear_in_features, 1))
                        head_replaced = True
                        logger.debug(f"Replaced SlowFast classification head at: {last_linear_name} (fallback)")
                
                if not head_replaced:
                    # Last resort: add new fc layer (may not work correctly)
                    logger.warning("Could not find SlowFast classification head, adding new fc layer (may not work correctly)")
                    self.backbone.fc = nn.Linear(2048, 1)
                
                self.use_torchvision = False  # PyTorchVideo models need list input
                self.use_pytorchvideo = True
                self.use_r3d_fallback = False
                hub_loaded = True
                logger.info("✓ Loaded SlowFast from PyTorch Hub (pytorchvideo)")
                
                # Verify we actually have a SlowFast model
                model_name = str(type(self.backbone).__name__).lower()
                model_str = str(self.backbone).lower()
                
                # Check model structure to verify it's actually SlowFast
                is_slowfast = (
                    'slowfast' in model_name or 
                    'slowfast' in model_str or
                    hasattr(self.backbone, 'pathway0') or 
                    hasattr(self.backbone, 'pathway1') or
                    hasattr(self.backbone, 'slow_pathway') or
                    hasattr(self.backbone, 'fast_pathway') or
                    ('slow' in model_str and 'fast' in model_str and 'pathway' in model_str)
                )
                
                if not is_slowfast:
                    raise RuntimeError(
                        f"CRITICAL: Loaded model does not appear to be SlowFast. "
                        f"Model type: {type(self.backbone).__name__}. "
                        f"Model structure: {list(self.backbone.named_children())[:5] if hasattr(self.backbone, 'named_children') else 'N/A'}. "
                        f"This may indicate a fallback or incorrect model was loaded."
                    )
                
                logger.info(f"✓ Verified SlowFast model structure: {type(self.backbone).__name__}")
                
                # Log model structure for debugging
                if hasattr(self.backbone, 'pathway0') or hasattr(self.backbone, 'pathway1'):
                    logger.debug("SlowFast model has pathway0/pathway1 (PyTorch Hub structure)")
                if hasattr(self.backbone, 'blocks'):
                    logger.debug(f"SlowFast model has {len(self.backbone.blocks)} blocks")
            except Exception as hub_error:
                logger.debug(f"Failed to load from PyTorch Hub: {hub_error}")
            
            # If PyTorch Hub failed, try pytorchvideo library directly
            if not hub_loaded:
                logger.info("Trying pytorchvideo library directly...")
                pytorchvideo_loaded = False
                try:
                    import pytorchvideo.models.hub as hub
                    from pytorchvideo.models import slowfast
                    # Try loading SlowFast from pytorchvideo
                    logger.info("Trying to load SlowFast from pytorchvideo...")
                    # pytorchvideo provides slowfast models
                    self.backbone = slowfast.create_slowfast(
                        model_num_class=400,  # Kinetics-400 pretrained
                        slowfast_fusion_conv_channel_ratio=1.0/8,
                        slowfast_conv_channel_fusion_ratio=2,
                    )
                    # Load pretrained weights if available
                    if pretrained:
                        try:
                            # Try to load pretrained weights
                            checkpoint = hub.load_state_dict_from_url(
                                "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth"
                            )
                            self.backbone.load_state_dict(checkpoint['model_state'], strict=False)
                            logger.info("✓ Loaded SlowFast pretrained weights from pytorchvideo")
                        except Exception as e:
                            logger.warning(f"Could not load pretrained weights: {e}, using random initialization")
                    
                    # Replace classification head for binary classification
                    # PyTorchVideo create_slowfast structure
                    head_replaced = False
                    
                    if hasattr(self.backbone, 'blocks') and len(self.backbone.blocks) > 0:
                        last_block = self.backbone.blocks[-1]
                        if hasattr(last_block, 'proj') and isinstance(last_block.proj, nn.Linear):
                            in_features = last_block.proj.in_features
                            last_block.proj = nn.Linear(in_features, 1)
                            head_replaced = True
                            logger.debug("Replaced SlowFast classification head at blocks[-1].proj (create_slowfast)")
                    
                    if not head_replaced and hasattr(self.backbone, 'fc'):
                        in_features = self.backbone.fc.in_features
                        self.backbone.fc = nn.Linear(in_features, 1)
                        head_replaced = True
                        logger.debug("Replaced SlowFast classification head at fc (create_slowfast)")
                    
                    if not head_replaced:
                        # Find last Linear layer
                        last_linear_name = None
                        last_linear_in_features = None
                        for name, module in self.backbone.named_modules():
                            if isinstance(module, nn.Linear):
                                last_linear_name = name
                                last_linear_in_features = module.in_features
                        
                        if last_linear_name is not None:
                            parts = last_linear_name.split('.')
                            if len(parts) > 1:
                                parent = self.backbone
                                for part in parts[:-1]:
                                    parent = getattr(parent, part)
                                setattr(parent, parts[-1], nn.Linear(last_linear_in_features, 1))
                            else:
                                setattr(self.backbone, last_linear_name, nn.Linear(last_linear_in_features, 1))
                            head_replaced = True
                            logger.debug(f"Replaced SlowFast classification head at: {last_linear_name} (create_slowfast fallback)")
                    
                    if not head_replaced:
                        logger.warning("Could not find SlowFast classification head in create_slowfast model, adding new fc layer")
                        self.backbone.fc = nn.Linear(2048, 1)
                    
                    self.use_torchvision = True
                    self.use_r3d_fallback = False
                    pytorchvideo_loaded = True
                    logger.info("✓ Loaded SlowFast from pytorchvideo")
                    
                    # Verify we actually have a SlowFast model
                    model_name = str(type(self.backbone).__name__).lower()
                    model_str = str(self.backbone).lower()
                    
                    # Check model structure to verify it's actually SlowFast
                    is_slowfast = (
                        'slowfast' in model_name or 
                        'slowfast' in model_str or
                        hasattr(self.backbone, 'pathway0') or 
                        hasattr(self.backbone, 'pathway1') or
                        hasattr(self.backbone, 'slow_pathway') or
                        hasattr(self.backbone, 'fast_pathway') or
                        ('slow' in model_str and 'fast' in model_str and 'pathway' in model_str)
                    )
                    
                    if not is_slowfast:
                        raise RuntimeError(
                            f"CRITICAL: Loaded model does not appear to be SlowFast. "
                            f"Model type: {type(self.backbone).__name__}. "
                            f"Model structure: {list(self.backbone.named_children())[:5] if hasattr(self.backbone, 'named_children') else 'N/A'}. "
                            f"This may indicate a fallback or incorrect model was loaded."
                        )
                    
                    logger.info(f"✓ Verified SlowFast model structure: {type(self.backbone).__name__}")
                    
                    # Log model structure for debugging
                    if hasattr(self.backbone, 'pathway0') or hasattr(self.backbone, 'pathway1'):
                        logger.debug("SlowFast model has pathway0/pathway1 (create_slowfast structure)")
                    if hasattr(self.backbone, 'blocks'):
                        logger.debug(f"SlowFast model has {len(self.backbone.blocks)} blocks")
                except ImportError:
                    logger.debug("pytorchvideo not available")
                    pytorchvideo_loaded = False
                except Exception as pv_error:
                    logger.debug(f"Failed to load from pytorchvideo: {pv_error}")
                    pytorchvideo_loaded = False
            
            # If pytorchvideo failed, try HuggingFace
            if not hub_loaded and not pytorchvideo_loaded:
                logger.info("Trying HuggingFace for SlowFast...")
                hf_loaded = False
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
                            except Exception:
                                # Fallback to AutoModel
                                model = AutoModel.from_pretrained(
                                    model_name,
                                    trust_remote_code=True
                                )
                            
                            # Extract the actual SlowFast backbone
                            if hasattr(model, 'slowfast'):
                                self.backbone = model.slowfast
                            elif hasattr(model, 'model'):
                                self.backbone = model.model
                            elif hasattr(model, 'backbone'):
                                self.backbone = model.backbone
                            else:
                                self.backbone = model
                            
                            # Replace classification head for binary classification
                            if hasattr(self.backbone, 'fc'):
                                in_features = self.backbone.fc.in_features
                                self.backbone.fc = nn.Linear(in_features, 1)
                            elif hasattr(self.backbone, 'head'):
                                in_features = self.backbone.head.in_features
                                self.backbone.head = nn.Linear(in_features, 1)
                            elif hasattr(self.backbone, 'classifier'):
                                in_features = self.backbone.classifier.in_features
                                self.backbone.classifier = nn.Linear(in_features, 1)
                            else:
                                # Add a new head if structure is different
                                feature_dim = 2048  # Common SlowFast feature dimension
                                self.backbone.fc = nn.Linear(feature_dim, 1)
                            
                            self.use_torchvision = False  # HuggingFace models may also need list input
                            self.use_pytorchvideo = True  # Assume PyTorchVideo-based models need list
                            self.use_r3d_fallback = False
                            hf_loaded = True
                            logger.info(f"✓ Loaded SlowFast from {model_name}")
                            
                            # Verify we actually have a SlowFast model
                            model_name_check = str(type(self.backbone).__name__).lower()
                            model_str_check = str(self.backbone).lower()
                            
                            # Check model structure to verify it's actually SlowFast
                            is_slowfast = (
                                'slowfast' in model_name_check or 
                                'slowfast' in model_str_check or
                                hasattr(self.backbone, 'pathway0') or 
                                hasattr(self.backbone, 'pathway1') or
                                hasattr(self.backbone, 'slow_pathway') or
                                hasattr(self.backbone, 'fast_pathway') or
                                ('slow' in model_str_check and 'fast' in model_str_check and 'pathway' in model_str_check)
                            )
                            
                            if not is_slowfast:
                                raise RuntimeError(
                                    f"CRITICAL: Loaded model does not appear to be SlowFast. "
                                    f"Model type: {type(self.backbone).__name__}. "
                                    f"Model structure: {list(self.backbone.named_children())[:5] if hasattr(self.backbone, 'named_children') else 'N/A'}. "
                                    f"This may indicate a fallback or incorrect model was loaded."
                                )
                            
                            logger.info(f"✓ Verified SlowFast model structure: {type(self.backbone).__name__}")
                            
                            # Log model structure for debugging
                            if hasattr(self.backbone, 'pathway0') or hasattr(self.backbone, 'pathway1'):
                                logger.debug("SlowFast model has pathway0/pathway1 (HuggingFace structure)")
                            break
                        except Exception as hf_error:
                            logger.debug(f"Failed to load from {model_name}: {hf_error}")
                            continue
                    
                    if not hf_loaded:
                        raise ImportError("Could not load SlowFast from any source")
                except ImportError:
                    # transformers not available
                    logger.warning("transformers library not available for HuggingFace SlowFast")
                    raise ImportError("HuggingFace SlowFast not available")
                except Exception as e:
                    logger.warning(f"Failed to load SlowFast from HuggingFace: {e}")
                    raise ImportError("HuggingFace SlowFast not available")
            
            # CRITICAL: Do NOT use r3d_18 or simplified SlowFast as fallback - require actual SlowFast model
            if not hub_loaded and not pytorchvideo_loaded and not hf_loaded:
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
                import torch.nn.functional as F
                
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
        # CRITICAL: Ensure slow and fast pathways have compatible temporal dimensions
        # The error "Expected size 63 but got size 125" suggests temporal mismatch
        # We need to ensure proper frame sampling for slow pathway
        if self.use_pytorchvideo:
            N, C, T, H, W = x.shape
            # Slow pathway: sample every alpha frames (ensures T_slow = ceil(T/alpha))
            # Use proper indexing to ensure we get valid frame counts
            slow_indices = torch.arange(0, T, self.alpha, device=x.device, dtype=torch.long)
            # Ensure we don't exceed tensor bounds
            slow_indices = slow_indices[slow_indices < T]
            slow_x = x[:, :, slow_indices, :, :]  # (N, C, T_slow, H, W)
            # Fast pathway: use all frames
            fast_x = x  # (N, C, T, H, W)
            
            # CRITICAL: PyTorchVideo SlowFast expects both pathways to have compatible shapes
            # The temporal dimensions will be processed differently, but spatial must match
            # Ensure both have the same spatial dimensions (already handled above)
            return self.backbone([slow_x, fast_x])
        
        # Fallback: try list input if tensor input fails (for models that weren't detected)
        try:
            return self.backbone(x)
        except (AssertionError, TypeError, RuntimeError) as e:
            error_msg = str(e).lower()
            if "list" in error_msg or "multipathway" in error_msg or "sizes of tensors must match" in error_msg:
                # Model needs list input - split into slow/fast pathways
                # Also handle temporal dimension mismatch errors
                N, C, T, H, W = x.shape
                # Use proper indexing to ensure valid frame counts
                slow_indices = torch.arange(0, T, self.alpha, device=x.device, dtype=torch.long)
                slow_indices = slow_indices[slow_indices < T]
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

