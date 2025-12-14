"""
I3D (Inflated 3D ConvNet) model for video classification.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class I3DModel(nn.Module):
    """
    I3D (Inflated 3D ConvNet) model for video classification.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize I3D model.
        
        Args:
            pretrained: Use pretrained weights from Kinetics-400
        """
        super().__init__()
        
        # I3D is not available in torchvision, use pytorchvideo via PyTorch Hub
        hub_loaded = False
        try:
            # PyTorch Hub is the recommended way to load pytorchvideo models
            import torch.hub
            if pretrained:
                logger.info("Loading I3D R50 from PyTorch Hub (facebookresearch/pytorchvideo)...")
                self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
            else:
                self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=False)
            
            # Replace classification head for binary classification
            # pytorchvideo I3D structure uses blocks[6].proj
            if hasattr(self.backbone, 'blocks'):
                last_block = self.backbone.blocks[-1]
                if hasattr(last_block, 'proj'):
                    in_features = last_block.proj.in_features
                    last_block.proj = nn.Linear(in_features, 1)
                else:
                    # Fallback: add a new head
                    self.backbone.fc = nn.Linear(2048, 1)
            elif hasattr(self.backbone, 'fc'):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features, 1)
            else:
                # Add a new head if neither exists
                self.backbone.fc = nn.Linear(2048, 1)
            
            self.use_torchvision = False  # PyTorchVideo models may need different input handling
            self.use_pytorchvideo = True
            hub_loaded = True
            logger.info("✓ Loaded I3D from PyTorch Hub (pytorchvideo)")
        except Exception as hub_error:
            logger.debug(f"Failed to load I3D from PyTorch Hub: {hub_error}")
            hub_loaded = False
        
        # If PyTorch Hub failed, try pytorchvideo library directly (alternative method)
        if not hub_loaded:
            pytorchvideo_loaded = False
            try:
                # Try using pytorchvideo's hub module
                import pytorchvideo.models.hub as pv_hub
                logger.info("Trying to load I3D from pytorchvideo library...")
                
                # pytorchvideo hub provides i3d models
                if pretrained:
                    self.backbone = pv_hub.i3d_r50(pretrained=True)
                else:
                    self.backbone = pv_hub.i3d_r50(pretrained=False)
                
                # Replace classification head
                if hasattr(self.backbone, 'blocks'):
                    last_block = self.backbone.blocks[-1]
                    if hasattr(last_block, 'proj'):
                        in_features = last_block.proj.in_features
                        last_block.proj = nn.Linear(in_features, 1)
                    else:
                        self.backbone.fc = nn.Linear(2048, 1)
                elif hasattr(self.backbone, 'fc'):
                    in_features = self.backbone.fc.in_features
                    self.backbone.fc = nn.Linear(in_features, 1)
                else:
                    self.backbone.fc = nn.Linear(2048, 1)
                
                self.use_torchvision = False
                self.use_pytorchvideo = True
                pytorchvideo_loaded = True
                logger.info("✓ Loaded I3D from pytorchvideo library")
            except Exception as pv_error:
                logger.debug(f"pytorchvideo library not available or failed: {pv_error}")
                pytorchvideo_loaded = False
        
        # Final fallback: use R3D_18 as approximation (available in torchvision)
        if not hub_loaded and not pytorchvideo_loaded:
            logger.warning(
                "I3D not available from pytorchvideo. Using R3D_18 as fallback. "
                "Install pytorchvideo for true I3D: pip install pytorchvideo"
            )
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
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
                self.use_torchvision = True
                self.use_pytorchvideo = False
            except ImportError:
                raise ImportError(
                    "I3D requires either pytorchvideo or torchvision. "
                    "Install pytorchvideo: pip install pytorchvideo"
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # CRITICAL: Handle small spatial dimensions like X3D and SlowFast
        # I3D also requires minimum spatial dimensions for proper processing
        if x.dim() == 5:
            N, C, T, H, W = x.shape
            min_spatial_size = 32  # Minimum required for I3D
            
            if H < min_spatial_size or W < min_spatial_size:
                import torch.nn.functional as F
                
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
        
        return self.backbone(x)


__all__ = ["I3DModel"]

