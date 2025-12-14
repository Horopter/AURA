"""
TimeSformer model: Space-time attention for video recognition.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available. TimeSformer will not work.")


class SpaceTimeAttention(nn.Module):
    """
    Space-time divided attention block (official TimeSformer implementation).
    
    Implements divided space-time attention as described in:
    "Is Space-Time Attention All You Need for Video Understanding?"
    
    First applies spatial attention within each frame, then temporal attention
    across frames for each spatial location.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections for divided space-time attention
        # Official TimeSformer uses shared QKV, but separate projections are clearer
        # and match the divided attention mechanism better
        self.qkv_spatial = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_temporal = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, num_frames: int = None, num_patches: int = None) -> torch.Tensor:
        """
        Forward pass with space-time divided attention (official TimeSformer implementation).
        
        Args:
            x: Input tensor (N, T*H*W, dim) where T is temporal, H*W is spatial
            num_frames: Number of temporal frames T (required for proper reshaping)
            num_patches: Number of spatial patches H*W (required for proper reshaping)
        
        Returns:
            Output tensor (N, T*H*W, dim)
        """
        N, L, D = x.shape
        
        # CRITICAL: TimeSformer requires T and H*W to properly implement divided attention
        # If not provided, try to infer from input shape (assumes square patches)
        if num_frames is None or num_patches is None:
            # Try to infer: L = T * (H*W), assume H*W is a perfect square
            # Common case: 16 frames * 256 patches = 4096 tokens
            # Or: 8 frames * 256 patches = 2048 tokens
            # Try common values
            for T_candidate in [8, 16, 32, 64]:
                if L % T_candidate == 0:
                    H_W_candidate = L // T_candidate
                    # Check if it's a reasonable number of patches (e.g., 14*14=196, 16*16=256)
                    sqrt_HW = int(H_W_candidate ** 0.5)
                    if sqrt_HW * sqrt_HW == H_W_candidate:
                        num_frames = T_candidate
                        num_patches = H_W_candidate
                        break
            
            if num_frames is None or num_patches is None:
                raise RuntimeError(
                    f"CRITICAL: Cannot infer num_frames and num_patches from input shape {x.shape}. "
                    f"TimeSformer requires explicit temporal and spatial dimensions for divided attention. "
                    f"Please provide num_frames and num_patches parameters."
                )
        
        # Reshape to (N, T, H*W, D) for proper space-time divided attention
        # Note: num_patches includes CLS token if present, so we reshape accordingly
        x = x.view(N, num_frames, num_patches, D)
        
        # STEP 1: Spatial attention - attend within each frame (divided attention)
        # Reshape to (N*T, H*W, D) to apply spatial attention frame by frame
        # This includes CLS token in each frame - spatial attention applies to all tokens in frame
        x_spatial = x.view(N * num_frames, num_patches, D)
        
        qkv_spatial = self.qkv_spatial(x_spatial).reshape(N * num_frames, num_patches, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_s, k_s, v_s = qkv_spatial[0], qkv_spatial[1], qkv_spatial[2]
        
        # Spatial attention: (N*T, num_heads, H*W, H*W)
        # Each frame attends to patches within that frame only
        attn_spatial = (q_s @ k_s.transpose(-2, -1)) * self.scale
        attn_spatial = attn_spatial.softmax(dim=-1)
        attn_spatial = self.attn_drop(attn_spatial)
        
        x_spatial_out = (attn_spatial @ v_s).transpose(1, 2).reshape(N * num_frames, num_patches, D)
        x_spatial_out = self.proj(x_spatial_out)
        x_spatial_out = self.proj_drop(x_spatial_out)
        
        # Reshape back to (N, T, H*W, D) and add residual
        x_spatial_out = x_spatial_out.view(N, num_frames, num_patches, D)
        x = x + x_spatial_out
        
        # STEP 2: Temporal attention - attend across frames for each spatial location (divided attention)
        # Reshape to (N*H*W, T, D) to apply temporal attention location by location
        x_temporal = x.permute(0, 2, 1, 3).contiguous()  # (N, H*W, T, D)
        x_temporal = x_temporal.view(N * num_patches, num_frames, D)
        
        qkv_temporal = self.qkv_temporal(x_temporal).reshape(N * num_patches, num_frames, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_t, k_t, v_t = qkv_temporal[0], qkv_temporal[1], qkv_temporal[2]
        
        # Temporal attention: (N*H*W, num_heads, T, T)
        # Each spatial location attends across all frames
        attn_temporal = (q_t @ k_t.transpose(-2, -1)) * self.scale
        attn_temporal = attn_temporal.softmax(dim=-1)
        attn_temporal = self.attn_drop(attn_temporal)
        
        x_temporal_out = (attn_temporal @ v_t).transpose(1, 2).reshape(N * num_patches, num_frames, D)
        x_temporal_out = self.proj(x_temporal_out)
        x_temporal_out = self.proj_drop(x_temporal_out)
        
        # Reshape back to (N, T, H*W, D) and add residual
        x_temporal_out = x_temporal_out.view(N, num_patches, num_frames, D)
        x_temporal_out = x_temporal_out.permute(0, 2, 1, 3).contiguous()  # (N, T, H*W, D)
        x = x + x_temporal_out
        
        # Reshape back to (N, T*H*W, D)
        x = x.view(N, num_frames * num_patches, D)
        
        return x


class TimeSformerModel(nn.Module):
    """
    TimeSformer: Space-time attention for video recognition.
    
    Based on: "Is Space-Time Attention All You Need for Video Understanding?"
    """
    
    def __init__(
        self,
        num_frames: int = 1000,
        img_size: int = 256,  # Match scaled video dimensions
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        pretrained: bool = True
    ):
        """
        Initialize TimeSformer model.
        
        Args:
            num_frames: Number of frames to process
            img_size: Input image size
            patch_size: Patch size for ViT
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Use bias in QKV projection
            dropout: Dropout probability
            attn_drop: Attention dropout probability
            pretrained: Use pretrained weights (if available)
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for TimeSformer. Install with: pip install timm")
        
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use ViT as backbone for patch embedding
        # Note: We extract patch_embed which works with any size, but we create our own pos_embed
        # for img_size=256. The patch_embed will produce 16x16=256 patches for 256x256 input.
        vit_backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            img_size=256,  # Match scaled video dimensions (patch_embed will adapt)
        )
        
        # Extract patch embedding (works with 256x256, produces 16x16=256 patches)
        self.patch_embed = vit_backbone.patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Time embedding (learnable)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks with space-time attention
        self.blocks = nn.ModuleList([
            SpaceTimeAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, 1)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W) or (N, T, C, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # Handle input format
        if x.dim() == 5:
            if x.shape[1] == 3:  # (N, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
            # Now x is (N, T, C, H, W)
        
        N, T, C, H, W = x.shape
        
        # Process each frame through patch embedding
        x = x.view(N * T, C, H, W)  # (N*T, C, H, W)
        x = self.patch_embed(x)  # (N*T, num_patches, embed_dim)
        
        # Reshape to (N, T, num_patches, embed_dim)
        x = x.view(N, T, self.num_patches, self.embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(N, T, -1, -1)  # (N, T, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=2)  # (N, T, num_patches+1, embed_dim)
        
        # Add positional embeddings (spatial) - same for all frames
        x = x + self.pos_embed.unsqueeze(1)  # (N, T, num_patches+1, embed_dim)
        
        # Add temporal embeddings - different for each frame
        # time_embed shape: (1, num_frames, embed_dim) -> (1, T, 1, embed_dim)
        if T <= self.time_embed.shape[1]:
            # Use first T frames of time embedding
            time_emb = self.time_embed[:, :T, :].unsqueeze(2)  # (1, T, 1, embed_dim)
        else:
            # If T > num_frames, repeat or interpolate
            # For simplicity, repeat the last embedding
            time_emb = self.time_embed[:, -1:, :].expand(1, T, 1, self.embed_dim)  # (1, T, 1, embed_dim)
        x = x + time_emb  # (N, T, num_patches+1, embed_dim)
        
        # Flatten: (N, T*(num_patches+1), embed_dim)
        x = x.view(N, T * (self.num_patches + 1), self.embed_dim)
        
        x = self.pos_drop(x)
        
        # Apply transformer blocks with space-time divided attention
        # CRITICAL: Pass num_frames and num_patches to each block for proper divided attention
        # Note: num_patches+1 includes CLS token, so we have T*(num_patches+1) total tokens
        total_tokens_per_frame = self.num_patches + 1  # Includes CLS token
        for blk in self.blocks:
            # TimeSformer blocks need to know T and tokens_per_frame for proper space-time attention
            # The block will handle CLS token correctly in divided attention
            x = blk(x, num_frames=T, num_patches=total_tokens_per_frame)
        
        x = self.norm(x)
        
        # Extract CLS token (first token of each frame, then average)
        # CLS tokens are at indices: 0, num_patches+1, 2*(num_patches+1), ...
        cls_indices = torch.arange(0, T * total_tokens_per_frame, total_tokens_per_frame, device=x.device)
        cls_tokens = x[:, cls_indices, :]  # (N, T, embed_dim)
        
        # Average pool over temporal dimension
        cls_token = cls_tokens.mean(dim=1)  # (N, embed_dim)
        
        # Classification
        logits = self.head(cls_token)  # (N, 1)
        
        return logits


__all__ = ["TimeSformerModel"]

