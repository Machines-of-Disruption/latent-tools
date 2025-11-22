import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def total_variation_loss(img):
    """
    Total variation loss for spatial smoothness.
    Encourages neighboring pixels to be similar.
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def l2_regularization(img):
    """L2 regularization to prevent extreme pixel values."""
    return torch.mean(img ** 2)


class ActivationHook:
    """Helper class to capture layer activations during forward pass."""

    def __init__(self):
        self.activations = {}
        self.handles = []

    def get_activation(self, name):
        """Create a hook function that saves activations."""
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def register_hook(self, model, layer_name):
        """Register a hook on the specified layer."""
        parts = layer_name.split('.')
        current = model

        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        handle = current.register_forward_hook(self.get_activation(layer_name))
        self.handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()


def apply_jitter(img, jitter_px=8):
    """Apply random spatial jitter to the image."""
    ox = np.random.randint(-jitter_px, jitter_px + 1)
    oy = np.random.randint(-jitter_px, jitter_px + 1)
    return torch.roll(img, shifts=(ox, oy), dims=(2, 3))


def apply_random_scale(img, scale_range=(0.95, 1.05)):
    """Apply random scaling to the image."""
    scale = np.random.uniform(*scale_range)
    _, _, h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)

    scaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Crop or pad back to original size
    if scale > 1:
        # Crop
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return scaled[:, :, start_h:start_h+h, start_w:start_w+w]
    else:
        # Pad
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return F.pad(scaled, (pad_w, w - new_w - pad_w, pad_h, h - new_h - pad_h))


def apply_random_rotation(img, max_angle=5):
    """Apply small random rotation."""
    angle = np.random.uniform(-max_angle, max_angle)
    angle_rad = angle * np.pi / 180

    # Simple rotation using affine transform
    theta = torch.tensor([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0]
    ], dtype=img.dtype, device=img.device).unsqueeze(0)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode='bilinear', align_corners=False)


def normalize_image(img):
    """Normalize image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img


class LTFeatureVisualization:
    """
    Generate synthetic images that maximize activation of a specific neuron.
    Uses gradient ascent with regularization to create interpretable visualizations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to visualize features from"}),
                "layer_name": ("STRING", {"default": "middle_block.1.transformer_blocks.0.attn1.to_q",
                                         "multiline": False,
                                         "tooltip": "Layer name to visualize"}),
                "channel": ("INT", {"default": 0, "min": 0, "max": 2048,
                                   "tooltip": "Channel/neuron index to maximize"}),
                "num_iterations": ("INT", {"default": 512, "min": 1, "max": 2000,
                                          "tooltip": "Number of optimization steps"}),
                "learning_rate": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001,
                                           "tooltip": "Learning rate for optimization"}),
                "image_size": ("INT", {"default": 224, "min": 64, "max": 512, "step": 64,
                                      "tooltip": "Size of generated visualization"}),
                "tv_weight": ("FLOAT", {"default": 1e-4, "min": 0, "max": 1.0, "step": 1e-5,
                                       "tooltip": "Total variation regularization weight"}),
                "l2_weight": ("FLOAT", {"default": 1e-5, "min": 0, "max": 1.0, "step": 1e-6,
                                       "tooltip": "L2 regularization weight"}),
                "use_augmentation": ("BOOLEAN", {"default": True,
                                                "tooltip": "Use jitter, scaling, rotation"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                "tooltip": "Random seed for initialization"}),
            }
        }

    CATEGORY = "LatentTools"
    DESCRIPTION = "Generate images that maximally activate a specific neuron (Distill Zoom In)"
    FUNCTION = "visualize"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def visualize(self, model, layer_name, channel, num_iterations, learning_rate,
                  image_size, tv_weight, l2_weight, use_augmentation, seed):
        """Generate feature visualization using activation maximization."""

        torch.manual_seed(seed)
        np.random.seed(seed)

        device = model.load_device

        # Initialize with random noise
        # Use spectral initialization for better results
        img = torch.randn(1, 3, image_size, image_size, device=device) * 0.01
        img.requires_grad = True

        # Setup optimizer
        optimizer = torch.optim.Adam([img], lr=learning_rate)

        # Setup activation hook
        hook = ActivationHook()
        try:
            hook.register_hook(model.model.diffusion_model, layer_name)
        except Exception as e:
            error_html = f"""
            <div class="flex flex-col gap-2 text-red-600">
                <div class="text-lg font-bold">Error:</div>
                <div>Could not register hook on layer '{layer_name}': {str(e)}</div>
            </div>
            """
            return {"ui": {"html": (error_html,)}}

        # Track progress
        activation_history = []
        snapshots = []
        snapshot_intervals = [num_iterations // 4, num_iterations // 2,
                            3 * num_iterations // 4, num_iterations - 1]

        try:
            for i in range(num_iterations):
                optimizer.zero_grad()

                # Apply augmentations
                if use_augmentation:
                    img_aug = apply_jitter(img, jitter_px=8)
                    img_aug = apply_random_scale(img_aug, scale_range=(0.95, 1.05))
                    if i % 4 == 0:  # Rotate less frequently
                        img_aug = apply_random_rotation(img_aug, max_angle=5)
                else:
                    img_aug = img

                # Forward pass through model
                # Note: We need to be careful here as we're working with the diffusion model directly
                # For now, we'll just pass the image through to get activations
                with torch.enable_grad():
                    # Normalize to typical image range
                    img_normalized = torch.sigmoid(img_aug)

                    # Forward pass - this will trigger our hook
                    try:
                        _ = model.model.diffusion_model(img_normalized,
                                                       timesteps=torch.zeros(1, device=device),
                                                       context=None)
                    except (AttributeError, RuntimeError, TypeError) as e:
                        # Some models need different inputs, try simpler approach
                        # Log the specific error for debugging
                        print(f"Primary forward pass failed: {e}, trying alternative approach")
                        _ = model.model.diffusion_model.input_blocks[0](img_normalized)

                    # Get activation from our target layer and channel
                    if layer_name not in hook.activations:
                        raise ValueError(f"Layer {layer_name} not found in activations")

                    activation = hook.activations[layer_name]

                    # Handle different activation shapes
                    if len(activation.shape) == 4:  # [batch, channels, h, w]
                        target_activation = activation[0, channel, :, :].mean()
                    elif len(activation.shape) == 3:  # [batch, seq, features]
                        target_activation = activation[0, :, channel].mean()
                    else:
                        target_activation = activation[0, channel].mean()

                    # Loss: negative activation (we want to maximize)
                    loss_activation = -target_activation

                    # Regularization
                    loss_tv = total_variation_loss(img) * tv_weight
                    loss_l2 = l2_regularization(img) * l2_weight

                    total_loss = loss_activation + loss_tv + loss_l2

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Track progress
                activation_history.append(target_activation.item())

                # Save snapshots
                if i in snapshot_intervals:
                    with torch.no_grad():
                        snapshot = torch.sigmoid(img.clone()).cpu()
                        snapshots.append((i, snapshot))

                # Print progress
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i+1}/{num_iterations}, "
                          f"Activation: {target_activation.item():.4f}, "
                          f"Loss: {total_loss.item():.4f}")

        finally:
            hook.remove_hooks()

        # Generate visualizations
        html = self.create_html_output(img, snapshots, activation_history,
                                       layer_name, channel)

        return {"ui": {"html": (html,)}}

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image."""
        img = torch.sigmoid(tensor).squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    def image_to_base64(self, img_pil):
        """Convert PIL Image to base64."""
        buf = BytesIO()
        img_pil.save(buf, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    def create_html_output(self, final_img, snapshots, activation_history,
                          layer_name, channel):
        """Create HTML visualization of results."""

        # Final result
        final_pil = self.tensor_to_image(final_img)
        final_b64 = self.image_to_base64(final_pil)

        # Snapshots
        snapshot_html = ""
        for i, (step, snap) in enumerate(snapshots):
            snap_pil = self.tensor_to_image(snap)
            snap_b64 = self.image_to_base64(snap_pil)
            snapshot_html += f"""
            <div class="flex flex-col items-center">
                <img src="{snap_b64}" style="width: 150px; height: 150px;">
                <div class="text-xs">Step {step}</div>
            </div>
            """

        # Activation history plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(activation_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Activation')
        ax.set_title('Activation Over Time')
        ax.grid(True, alpha=0.3)
        history_b64 = self.fig_to_base64(fig)

        html = f"""
        <div class="flex flex-col gap-4">
            <div class="text-lg font-bold">Feature Visualization</div>
            <div class="text-sm">Layer: {layer_name} | Channel: {channel}</div>

            <div class="flex flex-col gap-2">
                <div class="font-bold">Final Result:</div>
                <img src="{final_b64}" style="width: 300px; height: 300px; image-rendering: pixelated;">
                <div class="text-xs">Final Activation: {activation_history[-1]:.4f}</div>
            </div>

            <div class="flex flex-col gap-2">
                <div class="font-bold">Optimization Progress:</div>
                <div class="flex gap-2">
                    {snapshot_html}
                </div>
            </div>

            <div class="flex flex-col gap-2">
                <div class="font-bold">Activation History:</div>
                <img src="{history_b64}">
            </div>
        </div>
        """

        return html


class LTActivationAtlas:
    """
    Visualize spatial activation patterns for a given input across multiple channels.
    Shows heatmaps of where different neurons activate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model"}),
                "latent": ("LATENT", {"tooltip": "Input latent to analyze"}),
                "layer_name": ("STRING", {"default": "middle_block.1.transformer_blocks.0.attn1.to_q",
                                         "multiline": False}),
                "channels": ("STRING", {"default": "0,1,2,3,4,5,6,7",
                                       "multiline": False,
                                       "tooltip": "Comma-separated channel indices"}),
                "timestep": ("INT", {"default": 500, "min": 0, "max": 999,
                                    "tooltip": "Diffusion timestep"}),
            }
        }

    CATEGORY = "LatentTools"
    DESCRIPTION = "Show spatial activation patterns (activation atlas)"
    FUNCTION = "visualize"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def visualize(self, model, latent, layer_name, channels, timestep):
        """Create activation atlas visualization."""

        device = model.load_device
        latent_samples = latent["samples"].to(device)

        # Parse channel indices
        try:
            channel_indices = [int(c.strip()) for c in channels.split(',')]
        except (ValueError, AttributeError) as e:
            return {"ui": {"html": (f"<div class='text-red-600'>Invalid channel format. Use comma-separated integers. Error: {str(e)}</div>",)}}

        # Setup hook
        hook = ActivationHook()
        try:
            hook.register_hook(model.model.diffusion_model, layer_name)
        except Exception as e:
            return {"ui": {"html": (f"<div class='text-red-600'>Error: {str(e)}</div>",)}}

        try:
            # Forward pass
            timesteps = torch.tensor([timestep], device=device)
            with torch.no_grad():
                _ = model.model.diffusion_model(latent_samples, timesteps=timesteps, context=None)

            # Get activations
            if layer_name not in hook.activations:
                raise ValueError(f"Layer {layer_name} not activated")

            activation = hook.activations[layer_name]

            # Create visualization
            html = self.create_activation_atlas(activation, channel_indices, layer_name)

        finally:
            hook.remove_hooks()

        return {"ui": {"html": (html,)}}

    def create_activation_atlas(self, activation, channel_indices, layer_name):
        """Create heatmap visualization of activations."""

        activation = activation.cpu().numpy()

        # Handle different shapes
        if len(activation.shape) == 4:  # [batch, channels, h, w]
            batch, total_channels, h, w = activation.shape

            # Create grid of heatmaps
            n_channels = len(channel_indices)
            cols = min(4, n_channels)
            rows = (n_channels + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(rows, cols)

            for idx, ch in enumerate(channel_indices):
                if ch >= total_channels:
                    continue

                row = idx // cols
                col = idx % cols
                ax = axes[row, col]

                act_map = activation[0, ch, :, :]
                im = ax.imshow(act_map, cmap='hot', interpolation='nearest')
                ax.set_title(f'Channel {ch}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)

            # Hide unused subplots
            for idx in range(n_channels, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].axis('off')

            plt.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            img_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            html = f"""
            <div class="flex flex-col gap-2">
                <div class="text-lg font-bold">Activation Atlas</div>
                <div class="text-sm">Layer: {layer_name}</div>
                <div class="text-sm">Shape: {activation.shape}</div>
                <img src="{img_b64}">
            </div>
            """
        else:
            html = f"""
            <div class="text-yellow-600">
                Activation shape {activation.shape} not yet supported for atlas visualization.
            </div>
            """

        return html
