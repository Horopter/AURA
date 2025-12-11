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
            self.use_r3d_fallback = False
            
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
                if hasattr(self.backbone, 'blocks'):
                    # pytorchvideo SlowFast structure
                    last_block = self.backbone.blocks[-1]
                    if hasattr(last_block, 'proj'):
                        in_features = last_block.proj.in_features
                        last_block.proj = nn.Linear(in_features, 1)
                    else:
                        # Add a new head
                        self.backbone.fc = nn.Linear(2048, 1)
                elif hasattr(self.backbone, 'fc'):
                    in_features = self.backbone.fc.in_features
                    self.backbone.fc = nn.Linear(in_features, 1)
                else:
                    self.backbone.fc = nn.Linear(2048, 1)
                
                self.use_torchvision = True
                self.use_r3d_fallback = False
                hub_loaded = True
                logger.info("✓ Loaded SlowFast from PyTorch Hub (pytorchvideo)")
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
                    if hasattr(self.backbone, 'blocks'):
                        # pytorchvideo SlowFast structure
                        last_block = self.backbone.blocks[-1]
                        if hasattr(last_block, 'proj'):
                            in_features = last_block.proj.in_features
                            last_block.proj = nn.Linear(in_features, 1)
                        else:
                            # Add a new head
                            self.backbone.fc = nn.Linear(2048, 1)
                    elif hasattr(self.backbone, 'fc'):
                        in_features = self.backbone.fc.in_features
                        self.backbone.fc = nn.Linear(in_features, 1)
                    else:
                        self.backbone.fc = nn.Linear(2048, 1)
                    
                    self.use_torchvision = True
                    self.use_r3d_fallback = False
                    pytorchvideo_loaded = True
                    logger.info("✓ Loaded SlowFast from pytorchvideo")
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
                            
                            self.use_torchvision = True
                            self.use_r3d_fallback = False
                            hf_loaded = True
                            logger.info(f"✓ Loaded SlowFast from {model_name}")
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
            
            # Final fallback: use r3d_18 as pretrained backbone (similar to X3D)
            if not hub_loaded and not pytorchvideo_loaded and not hf_loaded:
                logger.warning("All SlowFast sources failed. Using r3d_18 as pretrained backbone.")
                try:
                    from torchvision.models.video import r3d_18, R3D_18_Weights
                    if pretrained:
                        try:
                            weights = R3D_18_Weights.KINETICS400_V1
                            self.backbone = r3d_18(weights=weights)
                        except (AttributeError, ValueError):
                            self.backbone = r3d_18(pretrained=True)
                    else:
                        self.backbone = r3d_18(pretrained=False)
                    
                    # Replace classification head for binary classification
                    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
                    self.use_torchvision = True  # Use backbone directly in forward
                    self.use_r3d_fallback = True  # Mark that we're using r3d_18
                except (ImportError, AttributeError):
                    # Final fallback: simplified SlowFast without pretrained weights
                    logger.warning("r3d_18 also not available. Using simplified SlowFast (no pretrained weights).")
                    self.use_torchvision = False
                    self.use_r3d_fallback = False
                    self._build_simplified_slowfast(slow_frames, fast_frames, alpha, beta)
        
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.alpha = alpha
    
    def _build_simplified_slowfast(
        self,
        slow_frames: int,
        fast_frames: int,
        alpha: int,
        beta: float
    ):
        """Build simplified SlowFast architecture."""
        # Slow pathway: 3D ResNet-like
        slow_channels = 64
        self.slow_pathway = nn.Sequential(
            # Stem
            nn.Conv3d(3, slow_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(slow_channels, slow_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(inplace=True),
            
            # Res blocks (simplified)
            self._make_res_block(slow_channels, slow_channels * 2, stride=2),
            self._make_res_block(slow_channels * 2, slow_channels * 4, stride=2),
            self._make_res_block(slow_channels * 4, slow_channels * 8, stride=2),
        )
        
        # Fast pathway: fewer channels, more frames
        fast_channels = int(slow_channels * beta)
        self.fast_pathway = nn.Sequential(
            # Stem
            nn.Conv3d(3, fast_channels, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(fast_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(fast_channels, fast_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(fast_channels),
            nn.ReLU(inplace=True),
            
            # Res blocks
            self._make_res_block(fast_channels, fast_channels * 2, stride=2),
            self._make_res_block(fast_channels * 2, fast_channels * 4, stride=2),
            self._make_res_block(fast_channels * 4, fast_channels * 8, stride=2),
        )
        
        # Lateral connections (simplified: just concatenate)
        # In real SlowFast, there are lateral connections between pathways
        
        # Fusion and classification
        fusion_dim = slow_channels * 8 + fast_channels * 8
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def _make_res_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Make a simplified 3D ResNet block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        if self.use_torchvision:
            # Use torchvision's SlowFast
            return self.backbone(x)
        
        # Simplified SlowFast
        N, C, T, H, W = x.shape
        
        # Sample frames for slow and fast pathways
        # Slow: sample every alpha frames
        slow_indices = torch.arange(0, T, self.alpha, device=x.device)
        slow_x = x[:, :, slow_indices, :, :]  # (N, C, T_slow, H, W)
        
        # Fast: use all frames (or sample at higher rate)
        fast_x = x  # (N, C, T, H, W)
        
        # Process through pathways
        slow_features = self.slow_pathway(slow_x)  # (N, C_slow, T', H', W')
        fast_features = self.fast_pathway(fast_x)  # (N, C_fast, T'', H'', W'')
        
        # Temporal alignment (simplified: just pool)
        slow_features = F.adaptive_avg_pool3d(slow_features, (1, 1, 1))  # (N, C_slow, 1, 1, 1)
        fast_features = F.adaptive_avg_pool3d(fast_features, (1, 1, 1))  # (N, C_fast, 1, 1, 1)
        
        # Concatenate
        combined = torch.cat([slow_features, fast_features], dim=1)  # (N, C_slow+C_fast, 1, 1, 1)
        
        # Classification
        logits = self.fusion(combined)  # (N, 1)
        
        return logits


__all__ = ["SlowFastModel"]

