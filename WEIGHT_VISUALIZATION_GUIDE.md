# Weight Visualization Guide

This guide shows you how to use the new weight visualization nodes to explore SDXL model internals.

## Quick Start

### Method 1: Load the Workflow

1. Open ComfyUI
2. Click "Load" and select `workflow_weight_visualization.json` from this repository
3. Update the checkpoint name in the `CheckpointLoaderSimple` node to your SDXL model
4. Click "Queue Prompt" to run

### Method 2: Build Manually

#### Step 1: List Available Layers

1. Add these nodes to your canvas:
   - `CheckpointLoaderSimple` (to load your SDXL model)
   - `LTListLayers` (LatentTools → LTListLayers)

2. Connect:
   - CheckpointLoaderSimple → MODEL → LTListLayers → model

3. Set the filter to narrow down layers (optional):
   - `conv` - shows all convolutional layers
   - `attn` - shows all attention layers
   - `transformer` - shows transformer blocks
   - Leave empty to see all layers

4. Run the workflow - you'll get a list of all available layer names

#### Step 2: Visualize Specific Layers

1. Add `LTVisualizeWeights` node (LatentTools → LTVisualizeWeights)

2. Connect:
   - CheckpointLoaderSimple → MODEL → LTVisualizeWeights → model

3. Leave `layer_name` empty and run once to see all available layers in the preview

4. Pick an interesting layer name from the list, for example:
   - `input_blocks.1.1.conv` - early conv layer
   - `input_blocks.4.1.transformer_blocks.0.attn1.to_q` - attention query projection
   - `middle_block.1.transformer_blocks.0.ff.net.0.proj` - feedforward layer
   - `output_blocks.5.1.conv` - later conv layer

5. Paste the layer name into the `layer_name` field

6. Configure visualization:
   - `num_filters`: How many filters/channels to visualize (16 is a good start)
   - `visualization_mode`:
     - `weights` - Shows raw weight matrices
     - `expanded` - Shows SVD factorized view (reveals patterns)
     - `both` - Shows both views (recommended)

7. Run the workflow - you'll see:
   - Layer shape and parameter count
   - Weight matrix visualization
   - SVD factorized components showing learned patterns

## Understanding the Visualizations

### Weight Matrices View

Shows the raw convolutional filter weights as a grid:
- **Rows**: Output channels (what this layer produces)
- **Columns**: Input channels (what this layer receives)
- **Colors**:
  - Red = positive weights
  - Blue = negative weights
  - White = near zero

Look for:
- Patterns in the filters (edges, textures, gradients)
- Diagonal structures (channel-wise operations)
- Smooth vs. noisy filters

### Expanded Weights View (SVD Factorization)

Shows how weights decompose into interpretable components:
- **Component 1, 2, 3...**: Ranked by importance (singular value σ)
- **Left plot**: How this component affects each output channel
- **Middle plot**: Spatial pattern (averaged across input channels)
- **Right plot**: All input channel patterns

Look for:
- What textures/patterns each component detects
- Diagonal patterns, edges, color gradients
- How components combine to form the full filter

Higher σ (sigma) values = more important components

## Interesting Layers to Explore

### Early Layers (Input Blocks)
- Detect low-level features: edges, colors, simple textures
- Example: `input_blocks.1.1.conv`
- Look for: Edge detectors, color channels, Gabor-like filters

### Middle Layers
- Detect mid-level features: complex textures, patterns
- Example: `middle_block.1.transformer_blocks.0.attn1.to_k`
- Look for: Texture patterns, more abstract features

### Attention Layers
- Learn what parts of the image to focus on
- Example: `*.attn1.to_q`, `*.attn1.to_k`, `*.attn1.to_v`
- Look for: Query/Key/Value transformations

### Late Layers (Output Blocks)
- Detect high-level features: object parts, semantic content
- Example: `output_blocks.5.1.conv`
- Look for: More complex, abstract patterns

## Tips

1. **Start broad, then zoom in**:
   - First run with `layer_name=""` to see all layers
   - Pick interesting ones based on name
   - Try different `num_filters` values (8, 16, 32)

2. **Compare related layers**:
   - Compare `attn1.to_q` vs `attn1.to_k` vs `attn1.to_v` in the same block
   - See how attention mechanisms transform inputs

3. **Use expanded view for interpretation**:
   - Raw weights can be hard to interpret
   - SVD factorization reveals the actual patterns being detected

4. **Filter by type**:
   - Use LTListLayers with filter="conv" to see only conv layers
   - Use filter="attn" to see only attention layers

## Example Workflow Structure

```
CheckpointLoaderSimple
  ├─→ MODEL → LTListLayers (filter="conv") → [layer list]
  └─→ MODEL → LTVisualizeWeights (layer_name="input_blocks.1.1.conv")
                ↓
            [Weight Visualization Preview]
```

## Troubleshooting

**Error: "Layer not found"**
- Copy the exact layer name from LTListLayers output
- Layer names are case-sensitive
- Check for extra spaces

**Empty visualization**
- Check that your checkpoint loaded correctly
- Try a different layer name
- Ensure you're using an SDXL model

**Too many/too few filters**
- Adjust `num_filters` parameter
- Start with 16, increase for more detail
- Some layers have fewer channels than requested

## Advanced: Finding Specific Features

Want to find layers that detect specific features?

1. Use `LTListLayers` with different filters
2. Try these layer patterns:
   - Early edge detectors: `input_blocks.0` to `input_blocks.3`
   - Texture layers: `input_blocks.4` to `input_blocks.6`
   - Semantic layers: `output_blocks`
   - Cross-attention (text): `*.attn2.*`
   - Self-attention (image): `*.attn1.*`

3. Visualize and look for patterns matching your target feature

## References

- [Distill: Visualizing Weights](https://distill.pub/2020/circuits/visualizing-weights/)
- [InceptionV1 Weight Explorer](https://storage.googleapis.com/distill-circuits/inceptionv1-weight-explorer/conv2d2_89.html)
