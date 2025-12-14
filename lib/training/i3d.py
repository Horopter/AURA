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
        
        # CRITICAL: Do NOT use r3d_18 as fallback - require actual I3D model
        if not hub_loaded and not pytorchvideo_loaded:
            raise RuntimeError(
                f"CRITICAL: Failed to load I3D model. "
                f"Tried PyTorch Hub (pytorchvideo) and pytorchvideo library. "
                f"I3D model is required - please install pytorchvideo: pip install pytorchvideo. "
                f"Fallback to r3d_18 is disabled to ensure proper model implementation."
            )
        
        # Verify we actually have an I3D model
        if hub_loaded or pytorchvideo_loaded:
            model_name = str(type(self.backbone).__name__).lower()
            model_str = str(self.backbone).lower()
            
            # Check model structure to verify it's actually I3D
            is_i3d = (
                'i3d' in model_name or 
                'i3d' in model_str or
                hasattr(self.backbone, 'blocks')  # PyTorchVideo I3D has blocks
            )
            
            if not is_i3d:
                raise RuntimeError(
                    f"CRITICAL: Loaded model does not appear to be I3D. "
                    f"Model type: {type(self.backbone).__name__}. "
                    f"Model structure: {list(self.backbone.named_children())[:5] if hasattr(self.backbone, 'named_children') else 'N/A'}. "
                    f"This may indicate a fallback or incorrect model was loaded."
                )
            
            logger.info(f"✓ Verified I3D model structure: {type(self.backbone).__name__}")
            
            # Log model structure for debugging
            if hasattr(self.backbone, 'blocks'):
                logger.debug(f"I3D model has {len(self.backbone.blocks)} blocks (PyTorchVideo structure)")
    
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

