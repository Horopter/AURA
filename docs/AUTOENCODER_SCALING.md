# Autoencoder-Based Video Scaling

This document describes how to use pretrained Hugging Face autoencoders for video scaling while preserving aspect ratio.

## Overview

The pipeline now supports using pretrained autoencoders from Hugging Face for high-quality video scaling. The default model is Stable Diffusion's VAE (`stabilityai/sd-vae-ft-mse`), which is designed for high-quality image/video encoding and decoding.

## Features

- **Aspect Ratio Preservation**: The autoencoder maintains the original aspect ratio of videos
- **High Quality**: Uses pretrained models optimized for visual quality
- **Automatic Padding**: Handles variable input sizes by padding to model requirements (multiples of 8 for Stable Diffusion VAE)
- **Fallback Support**: Automatically falls back to letterbox resize if autoencoder fails

## Installation

Install the required dependencies:

```bash
pip install diffusers transformers accelerate
```

Or update your requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

Use the `--method autoencoder` flag with Stage 3:

```bash
# Use default Stable Diffusion VAE
python src/scripts/run_stage3_scaling.py --method autoencoder

# Use a custom Hugging Face model
python src/scripts/run_stage3_scaling.py \
    --method autoencoder \
    --autoencoder-model "stabilityai/sd-vae-ft-ema"

# With custom target size
python src/scripts/run_stage3_scaling.py \
    --method autoencoder \
    --target-size 112
```

### Python API

```python
from lib.scaling import load_hf_autoencoder, stage3_scale_videos

# Load autoencoder
autoencoder = load_hf_autoencoder("stabilityai/sd-vae-ft-mse")

# Use in pipeline
stage3_scale_videos(
    project_root=".",
    augmented_metadata_path="data/augmented_videos/augmented_metadata.arrow",
    method="autoencoder",
    autoencoder_model="stabilityai/sd-vae-ft-mse",
    target_size=224
)
```

### Available Models

Recommended models from Hugging Face:

1. **`stabilityai/sd-vae-ft-mse`** (default)
   - Stable Diffusion VAE optimized with MSE loss
   - High quality, good for general use
   - VAE compression factor: 8x

2. **`stabilityai/sd-vae-ft-ema`**
   - Stable Diffusion VAE with EMA weights
   - Alternative to MSE version
   - VAE compression factor: 8x

3. **Custom Models**
   - Any Hugging Face model with `AutoencoderKL` interface
   - Must support variable aspect ratios or be padded appropriately

## How It Works

1. **Frame Loading**: Frames are loaded from video files
2. **Preprocessing**: 
   - Frames are normalized to [-1, 1] range
   - Padded to multiples of 8 (for Stable Diffusion VAE)
3. **Encoding**: Frames are encoded to latent space (8x smaller)
4. **Decoding**: Latents are decoded back to pixel space
5. **Postprocessing**:
   - Padding is removed
   - Aspect ratio is preserved
   - Final resize to target size with letterboxing

## Performance Considerations

- **Memory**: Autoencoder models require GPU memory (typically 1-2GB for Stable Diffusion VAE)
- **Speed**: Slower than letterbox resize (~10-50x depending on GPU)
- **Quality**: Higher quality than simple resize, especially for complex scenes

## Comparison: Letterbox vs Autoencoder

| Method | Speed | Quality | Memory | Aspect Ratio |
|--------|-------|---------|--------|--------------|
| Letterbox | Fast | Good | Low | Preserved |
| Autoencoder | Slow | Excellent | High | Preserved |

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce `chunk_size` (default: 250)
2. Use CPU instead of GPU (slower but less memory)
3. Use a smaller model or fall back to letterbox

### Model Download Issues

If model download fails:
1. Check internet connection
2. Set `HF_HOME` environment variable for custom cache location
3. Manually download model from Hugging Face

### Import Errors

If you see `ModuleNotFoundError`:
```bash
pip install diffusers transformers accelerate
```

## Example Output

```
Stage 3: Loading Hugging Face autoencoder: stabilityai/sd-vae-ft-mse
✓ Loaded autoencoder: stabilityai/sd-vae-ft-mse
Stage 3: Processing 2980 videos
Stage 3: Target size: 224x224
Stage 3: Method: autoencoder
Stage 3: Chunk size: 250 frames (optimized for memory)
```

## Integration with Full Pipeline

The autoencoder method integrates seamlessly with the 5-stage pipeline:

```python
# In src/run_new_pipeline.py
stage3_df = stage3_scale_videos(
    project_root=str(project_root),
    augmented_metadata_path=str(stage1_metadata_path),
    output_dir="data/scaled_videos",
    method="autoencoder",  # Changed from "resolution"
    target_size=224,
    chunk_size=250,
    autoencoder_model="stabilityai/sd-vae-ft-mse"  # Optional
)
```

## References

- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [Transformers Library](https://huggingface.co/docs/transformers)

