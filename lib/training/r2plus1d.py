"""
R(2+1)D model: Factorized 3D convolutions for video classification.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class R2Plus1DModel(nn.Module):
    """
    R(2+1)D (Factorized 3D Convolutions) model for video classification.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize R(2+1)D model.
        
        Args:
            pretrained: Use pretrained weights from Kinetics-400
        """
        super().__init__()
        
        try:
            from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
            
            if pretrained:
                try:
                    weights = R2Plus1D_18_Weights.KINETICS400_V1
                    self.backbone = r2plus1d_18(weights=weights)
                except (AttributeError, ValueError):
                    self.backbone = r2plus1d_18(pretrained=True)
            else:
                self.backbone = r2plus1d_18(pretrained=False)
            
            # Replace classification head for binary classification
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = True
            
        except ImportError:
            logger.warning("torchvision R(2+1)D not available. Using fallback implementation.")
            # Fallback: use R3D_18 as approximation
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        return self.backbone(x)


__all__ = ["R2Plus1DModel"]

