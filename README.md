[![](https://alexey.work/badge/)](https://alexey.work?ref=latent-tools-md)

# Latent Tools for ComfyUI

A collection of nodes for manipulating latent tensors in ComfyUI. These tools provide various operations for working with latent representations in stable diffusion workflows.

Made in collaboration with [fzayguler](https://github.com/fzayguler)

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory
2. Restart ComfyUI

Or use the ComfyUI extension manager.

## Available Nodes

### Preview and Debug

#### LTPreviewLatent
Visualizes latent tensors for debugging and inspection.

![alt text](assets/PreviewLatent.png)

### Weight Visualization

Visualize neural network weights to understand what features the model has learned. Inspired by [Distill's Visualizing Weights](https://distill.pub/2020/circuits/visualizing-weights/).

#### LTVisualizeWeights

Visualizes weights from SDXL UNet layers to reveal learned image features, textures, and patterns.

| **Inputs** |
|------------|
| - `model`: The model to visualize weights from |
| - `layer_name`: Layer name to visualize (e.g., 'input_blocks.1.1.conv') - leave empty to list available layers |
| - `num_filters`: Number of filters/channels to visualize (1-256) |
| - `visualization_mode`: Choose from 'weights' (raw weight matrices), 'expanded' (SVD factorized view), or 'both' |

**Features:**
- **Weight Matrices View**: Shows raw convolutional filter weights organized by input/output channels
- **Expanded Weights View**: Uses SVD factorization to reveal the structure and patterns in weights
- **Interactive Layer Exploration**: List all available layers by leaving layer_name empty
- **SVD Components**: Shows how weights decompose into interpretable patterns with singular values

**Usage:**
1. Connect a MODEL node to the input
2. Leave `layer_name` empty and run to see all available layers
3. Copy a layer name (e.g., 'input_blocks.4.1.transformer_blocks.0.attn1.to_k') and paste it into the `layer_name` field
4. Choose visualization mode and number of filters to display
5. View weight matrices and their factorized components to understand what the model has learned

**Quick Start:**
- Load the example workflow: `workflow_weight_visualization.json`
- See the detailed guide: [WEIGHT_VISUALIZATION_GUIDE.md](WEIGHT_VISUALIZATION_GUIDE.md)

#### LTListLayers

Helper node that lists all layers with weights in a model.

| **Inputs** |
|------------|
| - `model`: The model to list layers from |
| - `filter`: Filter layer names by substring (e.g., 'conv' or 'attn') |
| **Outputs** |
| - `layer_names`: Newline-separated list of layer names |

### Feature Visualization (Zoom In)

Understand what individual neurons detect by generating synthetic images that maximally activate them. Inspired by [Distill's Zoom In](https://distill.pub/2020/circuits/zoom-in/).

#### LTFeatureVisualization

Generates synthetic images that maximize activation of a specific neuron using gradient-based optimization.

| **Inputs** |
|------------|
| - `model`: The model to visualize features from |
| - `layer_name`: Layer name (e.g., 'middle_block.1.transformer_blocks.0.attn1.to_q') |
| - `channel`: Channel/neuron index to maximize (0-2048) |
| - `num_iterations`: Number of optimization steps (default: 512) |
| - `learning_rate`: Learning rate for optimization (default: 0.05) |
| - `image_size`: Size of generated visualization (default: 224) |
| - `tv_weight`: Total variation regularization weight (default: 1e-4) |
| - `l2_weight`: L2 regularization weight (default: 1e-5) |
| - `use_augmentation`: Use jitter, scaling, rotation (default: True) |
| - `seed`: Random seed for initialization |

**Features:**
- **Activation Maximization**: Generates images that make a neuron fire strongly
- **Regularization**: Total variation + L2 for clean, interpretable images
- **Augmentation**: Jitter, scaling, rotation for robust visualizations
- **Progress Tracking**: Shows snapshots and activation history over time

**What You'll See:**
- Edge detectors, texture patterns, color gradients
- What specific neurons/channels "look for" in images
- How features evolve during optimization

**Usage:**
1. Use `LTListLayers` to find interesting layers
2. Pick a layer and channel index
3. Run optimization (512 steps is usually enough)
4. View the generated feature visualization

#### LTActivationAtlas

Visualizes spatial activation patterns for multiple channels on a given input.

| **Inputs** |
|------------|
| - `model`: The model |
| - `latent`: Input latent to analyze |
| - `layer_name`: Layer to visualize activations from |
| - `channels`: Comma-separated channel indices (e.g., "0,1,2,3,4,5,6,7") |
| - `timestep`: Diffusion timestep (default: 500) |

**Features:**
- **Spatial Heatmaps**: Shows where each neuron activates on the input
- **Multi-Channel View**: Compare multiple channels side-by-side
- **Activation Atlas**: Understand the spatial structure of activations

**Usage:**
1. Connect a latent input (your starting noise or image)
2. Specify layer name and which channels to visualize
3. View heatmaps showing where each neuron activates

### Frequency and Edge Analysis

Analyze frequency-specific properties of neurons. Inspired by [Distill: Frequency and Edges](https://distill.pub/2020/circuits/frequency-edges/).

#### LTFrequencyResponse

Measure neuron response across spatial frequencies to determine if it's a high-frequency (fine detail) or low-frequency (coarse feature) detector.

| **Inputs** |
|------------|
| - `model`: The model |
| - `layer_name`: Layer to analyze |
| - `channel`: Neuron/channel index |
| - `min_frequency`: Minimum spatial frequency (default: 1.0 cycles/image) |
| - `max_frequency`: Maximum spatial frequency (default: 20.0 cycles/image) |
| - `num_frequencies`: Number of test frequencies (default: 20) |
| - `orientation`: Test orientation in degrees (default: 45°) |

**What You'll See:**
- Frequency response curve showing activation vs spatial frequency
- Classification: High-frequency, Low-frequency, or Band-pass
- Peak frequency and example test patterns
- Determines if neuron detects fine details vs coarse features

#### LTEdgeDetectorAnalysis

Analyze orientation tuning to understand if neuron is an oriented edge detector.

| **Inputs** |
|------------|
| - `model`: The model |
| - `layer_name`: Layer to analyze |
| - `channel`: Neuron/channel index |
| - `num_orientations`: Number of test orientations (default: 12) |
| - `edge_sharpness`: Edge sharpness for testing (default: 5.0) |

**What You'll See:**
- Polar plot of orientation tuning
- Preferred orientation and bandwidth
- Response to edges at different orientations
- Determines orientation selectivity

#### LTGaborFit

Fit Gabor filter parameters to convolutional weights to reveal their properties.

| **Inputs** |
|------------|
| - `model`: The model |
| - `layer_name`: Layer containing Conv2d weights |
| - `channel`: Output channel index |
| - `input_channel`: Input channel index (for Conv2d) |

**What You'll See:**
- Learned weights vs fitted Gabor filter
- Gabor parameters:
  - Wavelength (λ): Spatial frequency preference
  - Orientation (θ): Preferred angle
  - Phase (φ): Sine vs cosine
  - Sigma (σ): Spatial extent
  - Aspect Ratio (γ): Ellipticity
- Fit quality (R² and MSE)
- Residual showing deviation from Gabor model

**Usage:**
1. Use on early convolutional layers (e.g., `input_blocks.0.0`, `input_blocks.1.1.conv`)
2. Early layers often learn Gabor-like filters
3. Parameters reveal what the filter detects

### KSampler with additional noise input

#### LTKSampler

A KSampler variant that accepts an additional input for starting latent space noise.

| <img src="assets/KSampler.png" alt="LTKSampler" width="70%"> |
|------------|
| **Inputs** |
| - `model`: The model used for denoising |
| - `extra_seed`: See for any other noise used by the sampler |
| - `steps`: Number of steps in the denoising process |
| - `cfg`: Classifier-Free Guidance scale |
| - `sampler_name`: Algorithm used for sampling |
| - `scheduler`: Controls how noise is gradually removed |
| - `positive`: Positive conditioning |
| - `negative`: Negative conditioning |
| - `latent_image`: The latent image to denoise |
| - `latent_noise`: Starting noise for the sampler |
| - `denoise`: Amount of denoising to apply |
| **Outputs** |
| - `latent`: The denoised latent tensor |


#### LTGaussianLatent

Generates latent tensors filled with random values from a normal (Gaussian) distribution.

| <img src="assets/GaussianLatent.png" alt="LTGaussianLatent" width="30%"> |
|------------|
| **Inputs** |
| - `channels`: Number of channels (default: 4) |
| - `width`: Width of the latent space (will be divided by 8) |
| - `height`: Height of the latent space (will be divided by 8) |
| - `batch_size`: Number of samples to generate |
| - `mean`: Mean of the normal distribution |
| - `std`: Standard deviation of the normal distribution |
| - `seed`: Random seed |
| **Outputs** |
| - `latent`: Generated latent tensor |
| ![Gaussian Latent Node](assets/GaussianPlot.png) |


**Example:**

| "quick brown fox",  σ=0.9 μ=0 | "quick brown fox", σ=1.05, μ=0 | "quick brown fox", σ=1, μ=0 | "quick brown fox", σ=1, μ=-0.1 | "quick brown fox", σ=1, μ=0.1 |
|---|---|---|---|---|
![Random Range Gaussian Example](assets/fox_ddpm_2m-karras_mean_0.0_std_0.9_00001_.png) | ![Random Range Gaussian Example](assets/fox_ddpm_2m-karras_mean_0.0_std_1.05_00001_.png) | ![Random Range Gaussian Example](assets/fox_ddpm_2m-karras_mean_0.0_std_1.0_00001_.png) | ![Random Range Gaussian Example](assets/fox_ddpm_2m-karras_mean_-0.122_std_1.0_00001_.png) | ![Random Range Gaussian Example](assets/fox_ddpm_2m-karras_mean_0.122_std_1.0_00001_.png) |

#### LTUniformLatent
Generates latent tensors with values uniformly distributed between min and max.

| <img src="assets/UniformLatent.png" alt="LTUniformLatent" width="30%"> |
|------------|
| **Inputs** |
| - `channels`: Number of channels (default: 4) |
| - `width`: Width of the latent space (will be divided by 8) |
| - `height`: Height of the latent space (will be divided by 8) |
| - `batch_size`: Number of samples to generate |
| - `min`: Minimum value |
| - `max`: Maximum value |
| - `seed`: Random seed |
| **Outputs** |
| - `latent`: Generated latent tensor |
| ![Uniform Latent Node](assets/UniformPlot.png) |


> Note: Stable Diffusion models are usually trained with Gaussian noise, so the generations from Uniform noise will look unusual.

**Example:**

| "quick brown fox", -1.67 to 1.67 | "quick brown fox", -1.81 to 1.81 |
|---|---|
![Random Range Uniform Example](assets/FoxUniform_-1.67_1.67.png) | ![Random Range Uniform Example](assets/FoxUniform_-1.81_1.81.png) |

### Latent Operations

#### LTBlendLatent
Blends two latent tensors using various blending modes.

| <img src="assets/BlendLatent.png" alt="LTBlendLatent" width="40%"> |
|------------|
| **Inputs** |
| - `latent1`: First latent tensor |
| - `latent2`: Second latent tensor |
| - `mode`: Blending mode |
| - `ratio`: Blend ratio (0.0 to 1.0) **Only used for mode=sample or mode=interpolate** |
| - `seed`: Random seed **Only used for mode=sample** |
| **Outputs** |
| - `latent`: Blended latent tensor |


| **Blending Modes** | **Description** |
|-------------------|-----------------|
| `interpolate`     | Linear interpolation between latents |
| `add`             | Additive blending |
| `multiply`        | Multiplicative blending |
| `abs_max`         | Maximum of absolute values |
| `abs_min`         | Minimum of absolute values |
| `max`             | Element-wise maximum |
| `min`             | Element-wise minimum |
| `sample`          | Randomly sample from either latent based on ratio |

Example: \
Inputs: Random Gaussian σ=0.1 μ=0 (top) and Random Uniform [-1s, 1] (bottom) \
Blend modes: interpolate (top) and sample (bottom) \
![Blend Latent Node](assets/BlendLatentExample.png)


#### LTLatentOp
Applies mathematical operations to a latent tensor.

| <img src="assets/LatentOp-ops.png" alt="LTLatentOp" width="80%"> |
|------------|
| **Inputs** |
| - `latent`: Input latent tensor |
| - `op`: Operation to apply |
| - `arg`: Argument to apply (for operations that require an argument) |
| **Outputs** |
| - `latent`: Resulting latent tensor |

| **Operation** | **Description** |
|---------------|-----------------|
| `add`         | Add a value |
| `mul`         | Multiply by a value |
| `pow`         | Raise to a power |
| `exp`         | Exponential |
| `abs`         | Absolute value |
| `clamp_bottom` | Clamp minimum value |
| `clamp_top`   | Clamp maximum value |
| `norm`        | Normalize (zero mean, unit variance) |
| `mean`        | Set mean to specified value |
| `std`         | Set standard deviation to specified value |
| `sigmoid`     | Apply sigmoid function |
| `nop`         | No operation |

Example:
Inputs: Random Gaussian σ=1 μ=0
Op: abs
![Latent Op Example](assets/LatentOpExample.png)


#### LTLatentsConcatenate
Concatenates two latent tensors along a specified dimension.

|  <img src="assets/LatentsConcatenate.png" alt="LTLatentsConcatenate" width="50%"> |
|------------|
| **Inputs** |
| - `latent1`: First latent tensor |
| - `latent2`: Second latent tensor |
| - `dim`: Dimension to concatenate along (supports negative indexing) |
| **Outputs** |
| - `latent`: Concatenated latent tensor |

Example1:
2 images, concatenated along x axis:
![Latent Concatenate Example1](assets/LatentsConcatenateExample1.png)

Example2:
Stable Video Diffusion xt (24 frames total), concatenating
- 10 frames Gaussian noise (σ=1 μ=0)
- 4  frames Gaussian noise (σ=1.2 μ=0)
- 10 frames Gaussian noise (σ=1 μ=0)
![Latent Concatenate Example2](assets/LatentsConcatenateExample2.png)


| 10f (σ=1 μ=0) + 4f (σ=1.2 μ=0) + 10f (σ=1 μ=0) | 24f (σ=1 μ=0) |
|---|---|
| ![Latent Concatenate Example2](assets/LatentsConcatenateExample2a.gif) | ![Latent Concatenate Example2](assets/LatentsConcatenateExample2b.gif) |



#### LTLatentToShape
Extracts the shape of a latent tensor.

| <img src="assets/LatentToShape.png" alt="LTLatentToShape" width="40%"> |
|------------|
| **Inputs** |
| - `input`: Input latent tensor |
| **Outputs:** |
| - Return 7 dimensions of the input latent shape. Non-existing ones are returned as 0 |


#### LTReshapeLatent
Reshapes a latent tensor to new dimensions.

| <img src="assets/ReshapeLatent.png" alt="LTReshapeLatent" width="50%"> |
|------------|
| **Inputs** |
| - `input`: Input latent tensor |
| - `strict`: If True, requires exact size match |
| - `dim0`-`dim6`: Target dimensions (0 values are ignored) |
| **Outputs** |
| - `latent`: Reshaped latent tensor |

**Example:**
Reshape one latent to match another one:

<img src="assets/ShapeExample.png" alt="LTReshapeLatent" width="70%">
<!-- ![Latent Reshape Example](assets/ShapeExample.png) -->


## Batch helpers
### Parameter Randomization

LTNumberRangeUniform and LTRandomRangeGaussian are used to randomize inputs to other nodes when scheduling multiple images.

#### LTNumberRangeUniform
Generates random float values from a uniform distribution.

| <img src="assets/NumberRangeUniform.png" alt="LTNumberRangeUniform" width="50%"> |
|------------|
| **Inputs** |
| - `min`: Minimum value |
| - `max`: Maximum value |
| - `seed`: Random seed - randomize it to get different values for each image |
| **Outputs** |
| - `float`: Generated number as float |
| - `int`: Generated number as int |

#### LTNumberRangeGaussian
Generates random values from a Gaussian distribution.

| <img src="assets/NumberRangeGaussian.png" alt="LTRandomRangeGaussian" width="50%"> |
|------------|
| **Inputs** |
| - `mean`: Mean of the normal distribution |
| - `std`: Standard deviation of the normal distribution |
| - `seed`: Random seed - randomize it to get different values for each image |
| **Outputs** |
| - `float`: Generated number as float |
| - `int`: Generated number as int |


**Example:**

Let's randomize a bumch of parameters for batch generation:
- For the Gaussian Noise that is used as input:
   - the mean value will be itself a random uniform value between -0.2 and 0.2.
   - The standard deviation will be itself a random value with mean 1 and std 0.5, clipped to 0.1 at the low end (using a ComfyMath node).
- The number of diffusion steps is random uniform value betwee 5 and 40.
- The cfg is random gussian with mean 8 and std 1, clipped to 0.1 at the low end (using a ComfyMath node).

Using this setup, generate 100 images with different parameters.

| | |
|---|---|
| ![alt text](assets/NumberRangeExample.png) |  <img src="assets/random_params.gif" alt="random_params" width="100%"> |


### LTFloat_Steps_0001
### LTFloat_Steps_0001
### LTFloat_Steps_0002
### LTFloat_Steps_0005
### LTFloat_Steps_001
### LTFloat_Steps_002
### LTFloat_Steps_005
### LTFloat_Steps_01
### LTFloat_Steps_02
### LTFloat_Steps_05
### LTFloat_Steps_1

These nodes are used to increment/decrement a float value by a fixed amount when scheduling multiple images.

| ![Float Steps](assets/Float_Step_XXX.png) |
|------------|
| **Inputs** |
| - `value`: float |
| **Outputs** |
| - `float`: float |
| - `string`: string |

**Example:**

Generate images with fixed seed Gaussian noise, starting with σ etween 0.8 and 1.2 with 0.001 step increments:

![alt text](assets/Float_StepExample.png)

Generate images with fixed seed Gaussian noise, starting with μ between -0.22 and 0.3 with 0.001 step increments:

![alt text](assets/Float_StepExample2.png)

Result:
| Sweeping Standard Deviation (σ) | Sweeping Mean (μ) |
|---|---|
| ![std sweep](assets/std_sweep.gif) | ![mean sweep](assets/mean_sweep.gif) |
