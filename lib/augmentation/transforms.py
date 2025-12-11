"""
Augmentation transforms for video frames.

Provides spatial and temporal augmentation transforms.
"""

from __future__ import annotations

import random
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F_transforms
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial Augmentations
# ---------------------------------------------------------------------------


class RandomRotation:
    """Random rotation with small angles."""
    
    def __init__(self, degrees: float = 10.0, p: float = 0.5):
        self.degrees = degrees
        self.p = p
        self.angle = 0.0
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            self.angle = random.uniform(-self.degrees, self.degrees)
            # Use fillcolor for compatibility with older PIL versions
            # For RGB images, use (0, 0, 0) for black, for grayscale use 0
            if img.mode == 'RGB':
                fillcolor = (0, 0, 0)
            else:
                fillcolor = 0
            try:
                return img.rotate(
                    self.angle, 
                    resample=Image.Resampling.BILINEAR, 
                    fillcolor=fillcolor
                )
            except TypeError:
                # Fallback for very old PIL versions that don't support fillcolor
                return img.rotate(self.angle, resample=Image.Resampling.BILINEAR)
        return img


class RandomAffine:
    """Random affine transformation."""
    
    def __init__(
        self, 
        translate: Tuple[float, float] = (0.1, 0.1), 
        scale: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ):
        self.translate = translate
        self.scale = scale
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
                return F_transforms.affine(
                img,
                angle=0,
                translate=[int(img.width * t) for t in self.translate],
                scale=random.uniform(*self.scale),
                shear=0,
                fill=0
            )
        return img


class RandomGaussianNoise:
    """Add Gaussian noise to image."""
    
    def __init__(self, std: float = 10.0, p: float = 0.5):
        self.std = std
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_array = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, self.std, img_array.shape).astype(np.float32)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        return img


class RandomGaussianBlur:
    """Apply Gaussian blur."""
    
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 2.0), p: float = 0.5):
        self.radius_range = radius_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomCutout:
    """Random cutout (random rectangular regions set to zero)."""
    
    def __init__(self, num_holes: int = 1, length: int = 16, p: float = 0.5):
        self.num_holes = num_holes
        self.length = length
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            for _ in range(self.num_holes):
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)
                y1 = max(0, y - self.length // 2)
                y2 = min(h, y + self.length // 2)
                x1 = max(0, x - self.length // 2)
                x2 = min(w, x + self.length // 2)
                img_array[y1:y2, x1:x2] = 0
            
            return Image.fromarray(img_array)
        return img


class LetterboxResize:
    """Resize with letterboxing to maintain aspect ratio."""
    
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        result = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        result.paste(img, (paste_x, paste_y))
        
        return result


# ---------------------------------------------------------------------------
# Temporal Augmentations
# ---------------------------------------------------------------------------


def temporal_frame_drop(frames: List[torch.Tensor], drop_prob: float = 0.1) -> List[torch.Tensor]:
    """Randomly drop frames from sequence."""
    if random.random() < drop_prob:
        indices = list(range(len(frames)))
        num_to_drop = random.randint(1, max(1, len(frames) // 4))
        indices_to_drop = random.sample(indices, min(num_to_drop, len(indices)))
        return [f for i, f in enumerate(frames) if i not in indices_to_drop]
    return frames


def temporal_frame_duplicate(frames: List[torch.Tensor], dup_prob: float = 0.1) -> List[torch.Tensor]:
    """Randomly duplicate frames in sequence."""
    if random.random() < dup_prob:
        result = []
        for frame in frames:
            result.append(frame)
            if random.random() < 0.3:  # 30% chance to duplicate this frame
                result.append(frame.clone())
        return result
    return frames


def temporal_reverse(frames: List[torch.Tensor], reverse_prob: float = 0.1) -> List[torch.Tensor]:
    """Randomly reverse frame sequence."""
    if random.random() < reverse_prob:
        return frames[::-1]
    return frames


# ---------------------------------------------------------------------------
# Simple Frame Augmentation (for Stage 1)
# ---------------------------------------------------------------------------


def apply_simple_augmentation(
    frame: np.ndarray, 
    aug_type: str, 
    seed: int
) -> np.ndarray:
    """
    Apply a simple augmentation to a single frame.
    
    Used in Stage 1 pipeline for quick augmentation.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if aug_type == 'none':
        return frame
    
    pil_image = Image.fromarray(frame)
    
    if aug_type == 'rotation':
        angle = random.uniform(-15, 15)
        # Use fillcolor for older PIL versions, fill for newer ones
        try:
            pil_image = pil_image.rotate(angle, fill=0)
        except TypeError:
            # Fallback for older PIL versions
            pil_image = pil_image.rotate(angle, fillcolor=0)
    elif aug_type == 'flip':
        if random.random() < 0.5:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif aug_type == 'brightness':
        factor = random.uniform(0.7, 1.3)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(factor)
    elif aug_type == 'contrast':
        factor = random.uniform(0.7, 1.3)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(factor)
    elif aug_type == 'saturation':
        factor = random.uniform(0.7, 1.3)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(factor)
    elif aug_type == 'gaussian_noise':
        frame_array = np.array(pil_image, dtype=np.float32)
        noise = np.random.normal(0, 10, frame_array.shape).astype(np.float32)
        frame_array = np.clip(frame_array + noise, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_array)
    elif aug_type == 'gaussian_blur':
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    elif aug_type == 'affine':
        transform = transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=0
        )
        tensor = transforms.ToTensor()(pil_image)
        tensor = transform(tensor)
        pil_image = transforms.ToPILImage()(tensor)
    elif aug_type == 'elastic':
        # Simplified elastic transform
        frame_array = np.array(pil_image)
        # Just return original for now (can be enhanced)
        pass
    elif aug_type == 'cutout':
        # Random cutout augmentation
        frame_array = np.array(pil_image)
        h, w = frame_array.shape[:2]
        num_holes = random.randint(1, 3)
        length = random.randint(16, 32)
        for _ in range(num_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            y1 = max(0, y - length // 2)
            y2 = min(h, y + length // 2)
            x1 = max(0, x - length // 2)
            x2 = min(w, x + length // 2)
            frame_array[y1:y2, x1:x2] = 0
        pil_image = Image.fromarray(frame_array)
    
    return np.array(pil_image)


# ---------------------------------------------------------------------------
# Comprehensive Transform Builders
# ---------------------------------------------------------------------------


def build_comprehensive_frame_transforms(
    train: bool = True,
    fixed_size: Optional[int] = None,
    max_size: Optional[int] = None,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build comprehensive frame transforms for video augmentation.
    
    Args:
        train: If True, apply data augmentations
        fixed_size: Fixed size for both dimensions (e.g., 256). Uses letterboxing to maintain aspect ratio.
        max_size: Maximum size for longer edge when resizing (maintains aspect ratio, but variable output size).
                  Only used if fixed_size is None.
        augmentation_config: Optional configuration dict for augmentation types
    
    Returns:
        Tuple of (spatial_transform, post_tensor_transform)
    """
    transform_list = [transforms.functional.to_pil_image]
    
    # Resize strategy: prefer fixed_size (letterboxing) over max_size (variable)
    if fixed_size is not None:
        # Fixed size with letterboxing: resize to fit within fixed_size x fixed_size while maintaining aspect ratio
        def letterbox_resize(img):
            """Resize image to fit within fixed_size x fixed_size, adding black bars if needed."""
            w, h = img.size
            scale = min(fixed_size / w, fixed_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            
            # Create a black canvas of fixed_size x fixed_size
            canvas = Image.new('RGB', (fixed_size, fixed_size), (0, 0, 0))
            # Center the resized image on the canvas
            paste_x = (fixed_size - new_w) // 2
            paste_y = (fixed_size - new_h) // 2
            canvas.paste(img_resized, (paste_x, paste_y))
            return canvas
        
        transform_list.append(letterbox_resize)
    elif max_size is not None:
        # Variable size: resize longer edge to max_size (maintains aspect ratio)
        transform_list.append(transforms.Resize(max_size, antialias=True))
    
    # Add spatial augmentations based on config (only during training)
    if train:
        if augmentation_config is None:
            augmentation_config = {}
        
        # Add spatial augmentations based on config
        if augmentation_config.get('rotation', True):
            transform_list.append(RandomRotation(degrees=15.0, p=0.5))
        
        if augmentation_config.get('affine', True):
            transform_list.append(RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5))
        
        if augmentation_config.get('gaussian_noise', False):
            transform_list.append(RandomGaussianNoise(std=10.0, p=0.3))
        
        if augmentation_config.get('gaussian_blur', False):
            transform_list.append(RandomGaussianBlur(radius_range=(0.5, 2.0), p=0.3))
        
        if augmentation_config.get('cutout', False):
            transform_list.append(RandomCutout(num_holes=1, length=16, p=0.3))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Build spatial transform
    spatial_transform = transforms.Compose(transform_list)
    
    # Post-tensor transforms (normalization)
    post_tensor_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return spatial_transform, post_tensor_transform


def apply_temporal_augmentations(
    frames: List[torch.Tensor],
    temporal_config: Optional[Dict[str, Any]] = None
) -> List[torch.Tensor]:
    """
    Apply temporal augmentations to frame sequence.
    
    Args:
        frames: List of frame tensors
        temporal_config: Optional temporal augmentation config
    
    Returns:
        Augmented frame list
    """
    if temporal_config is None:
        return frames
    
    result = frames.copy()
    
    if temporal_config.get('frame_drop', False):
        result = temporal_frame_drop(result, drop_prob=0.1)
    
    if temporal_config.get('frame_duplicate', False):
        result = temporal_frame_duplicate(result, dup_prob=0.1)
    
    if temporal_config.get('reverse', False):
        result = temporal_reverse(result, reverse_prob=0.1)
    
    return result
