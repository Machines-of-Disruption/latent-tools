# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latent Tools is a ComfyUI custom nodes extension for manipulating latent tensors in Stable Diffusion workflows. It provides nodes for generating, visualizing, blending, and transforming latent space representations.

## Architecture

### ComfyUI Custom Node Structure

This is a ComfyUI extension that follows the standard custom node pattern:

- `__init__.py` - Main entry point that exports `NODE_CLASS_MAPPINGS` dictionary and `WEB_DIRECTORY`
- Individual `.py` files contain node class definitions (one class per file typically)
- `web/js/` - Frontend JavaScript for custom UI components
- Node classes must define `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, and `CATEGORY` class attributes

### Node Registration

All nodes are registered in `__init__.py` via `NODE_CLASS_MAPPINGS`. The naming convention:
- Python class names use CamelCase (e.g., `LTRandomGaussian`)
- ComfyUI node names use the "LT" prefix (e.g., `LTGaussianLatent`)
- Float step nodes are dynamically generated via `LTFloatSteps` list

### Latent Tensor Format

ComfyUI passes latents as dictionaries with this structure:
```python
{
    "samples": torch.Tensor,  # Shape: [batch, channels, height/8, width/8]
    "noise_mask": torch.Tensor (optional),  # For inpainting
    "batch_index": list (optional)  # For batch processing
}
```

All nodes must preserve this dictionary structure, copying it and updating the "samples" field.

### Key Dependencies

- `comfy` - Core ComfyUI library (not in requirements.txt, provided by ComfyUI)
- `lovely_tensors` - Tensor visualization library used by `LTPreviewLatent`
- `torch` - PyTorch for tensor operations
- `matplotlib` - Used for latent visualizations (set to 'Agg' backend for non-interactive use)

## Code Patterns

### Node Implementation Pattern

All node classes follow this structure:
1. `INPUT_TYPES` classmethod - defines input schema with types and tooltips
2. Class attributes: `CATEGORY`, `DESCRIPTION`, `FUNCTION`, `RETURN_TYPES`, `OUTPUT_TOOLTIPS` (optional)
3. Implementation method (name matches `FUNCTION` attribute)
4. Return tuple of outputs matching `RETURN_TYPES`

### Latent Validation

Always validate latent inputs:
```python
assert isinstance(latent, dict), "Inputs must be dictionaries"
samples = latent["samples"]
assert isinstance(samples, torch.Tensor), "latent['samples'] must be torch.Tensor"
```

### Shape Handling

- Latent height/width are 1/8th of final image resolution
- Default channels: 4 (standard SD latent space)
- When broadcasting noise to multiple samples: use `noise.broadcast_to(size=latent_image_samples.shape)`
- Check dimension compatibility before operations

## Development Commands

This project has no build system - it's pure Python that gets loaded by ComfyUI at runtime.

Install dependencies:
```bash
pip install -r requirements.txt
```

Testing is done within ComfyUI - there are no unit tests or test commands.

## Node Categories

### Latent Generation
- `LTGaussianLatent` - Generate Gaussian distributed noise (most common for SD)
- `LTUniformLatent` - Generate uniform distributed noise (experimental)

### Latent Operations
- `LTBlendLatent` - Blend two latents using 8 modes (interpolate, add, multiply, abs_max, abs_min, max, min, sample)
- `LTLatentOp` - Apply mathematical operations (add, mul, pow, exp, abs, clamp, norm, sigmoid, etc.)
- `LTLatentsConcatenate` - Concatenate along any dimension (useful for video/batch)

### Shape Manipulation
- `LTReshapeLatent` - Reshape latent tensors (supports strict mode for exact size matching)
- `LTLatentToShape` - Extract shape information (returns 7 dimensions, unused ones as 0)

### Custom Sampling
- `LTKSampler` - Modified KSampler accepting custom latent noise input instead of seed-based generation
  - Takes both `latent_image` (what to denoise) and `latent_noise` (starting noise)
  - Uses `lt_prepare_noise` for batch index handling

### Visualization
- `LTPreviewLatent` - Debug visualization showing distribution plots and channel views
  - Returns HTML with embedded base64 images
  - Uses lovely_tensors for tensor summaries
- `LTVisualizeWeights` - Visualize model weights with SVD factorization (Distill Circuits style)
  - Extracts weights from specific SDXL layers
  - Shows raw weight matrices and expanded (factorized) views
  - Navigate layers by leaving layer_name empty
- `LTListLayers` - List all available layers with weights in a model
- `LTFeatureVisualization` - Generate images that maximize neuron activation (Zoom In)
  - Uses gradient ascent with regularization
  - Shows what patterns/textures individual neurons detect
  - Includes total variation, L2, jitter, scaling, rotation augmentations
- `LTActivationAtlas` - Spatial activation heatmaps for given inputs
  - Shows where neurons activate across spatial dimensions
  - Multi-channel comparison view

### Batch Helpers
- `LTNumberRangeUniform` / `LTNumberRangeGaussian` - Random parameter generation for batch workflows
- `LTFloat_Steps_*` - Increment/decrement values for parameter sweeps (11 variants with different step sizes)

## License Notes

Most files are under standard license, but `samplers.py` is GPL v3 (derived from ComfyUI core).

## Installation Context

Users install this by cloning into `ComfyUI/custom_nodes/latent-tools/` or via ComfyUI Manager. The extension is auto-discovered by ComfyUI on startup via `__init__.py`.
