import torch
import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt

class LTVisualizeWeights:
    """
    Visualize neural network weights similar to Distill's weight visualization.
    Targets SDXL UNet layers to show learned features.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to visualize weights from"}),
                "layer_name": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Layer name to visualize (e.g., 'input_blocks.1.1.conv')"}),
                "num_filters": ("INT", {"default": 16, "min": 1, "max": 256,
                                       "tooltip": "Number of filters to visualize"}),
                "visualization_mode": (["weights", "expanded", "both"],
                                      {"default": "both",
                                       "tooltip": "Visualization mode: weights, expanded (factorized), or both"}),
            }
        }

    CATEGORY = "LatentTools"
    DESCRIPTION = "Visualize model weights to understand learned features (Distill Circuits style)"
    FUNCTION = "visualize"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def get_layer_weights(self, model, layer_name):
        """Extract weights from a specific layer in the model."""
        try:
            # Access the underlying diffusion model
            diffusion_model = model.model.diffusion_model

            # Navigate to the layer
            parts = layer_name.split('.')
            current = diffusion_model

            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)

            # Get the weight tensor
            if hasattr(current, 'weight'):
                return current.weight.data.cpu()
            else:
                raise ValueError(f"Layer {layer_name} does not have weights")

        except Exception as e:
            raise ValueError(f"Error accessing layer '{layer_name}': {str(e)}")

    def list_available_layers(self, model):
        """List all available layers with weights in the model."""
        layers = []

        def recurse_model(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if hasattr(child, 'weight'):
                    layers.append(full_name)
                recurse_model(child, full_name)

        try:
            diffusion_model = model.model.diffusion_model
            recurse_model(diffusion_model)
        except Exception as e:
            print(f"Error listing layers: {e}")

        return layers

    def visualize_weight_matrix(self, weights, num_filters=16):
        """Create a visualization of weight matrices."""
        # weights shape is typically [out_channels, in_channels, height, width]
        weights = weights[:num_filters]

        if weights.dim() == 4:
            # Conv2d weights
            out_c, in_c, h, w = weights.shape

            # Create grid of weight visualizations
            fig, axes = plt.subplots(num_filters, in_c,
                                    figsize=(min(in_c * 2, 20), num_filters * 2))

            # Handle edge cases for axes array shape
            if num_filters == 1 and in_c == 1:
                axes = np.array([[axes]])
            elif num_filters == 1:
                axes = axes.reshape(1, -1)
            elif in_c == 1:
                axes = axes.reshape(-1, 1)

            for i in range(min(num_filters, out_c)):
                for j in range(in_c):
                    ax = axes[i, j]
                    weight_slice = weights[i, j].numpy()

                    # Normalize for visualization, handle zero weights
                    vmax = np.abs(weight_slice).max()
                    if vmax == 0:
                        vmax = 1e-8  # Avoid division by zero, use small epsilon
                    ax.imshow(weight_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                    ax.axis('off')

                    if i == 0:
                        ax.set_title(f'In:{j}', fontsize=8)
                    if j == 0:
                        ax.set_ylabel(f'Out:{i}', fontsize=8)

            plt.tight_layout()

        elif weights.dim() == 2:
            # Linear layer weights
            fig, ax = plt.subplots(figsize=(12, 8))
            weight_np = weights.numpy()
            vmax = np.abs(weight_np).max()
            if vmax == 0:
                vmax = 1e-8  # Avoid division by zero
            im = ax.imshow(weight_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_xlabel('Input Features')
            ax.set_ylabel('Output Features')
            ax.set_title('Weight Matrix')
            plt.colorbar(im, ax=ax)
        else:
            # Fallback for other dimensions
            fig, ax = plt.subplots(figsize=(12, 8))
            weight_flat = weights.reshape(-1, weights.shape[-1]).numpy()
            vmax = np.abs(weight_flat).max()
            if vmax == 0:
                vmax = 1e-8  # Avoid division by zero
            im = ax.imshow(weight_flat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            plt.colorbar(im, ax=ax)

        return fig

    def visualize_expanded_weights(self, weights, num_filters=16, rank=3):
        """
        Create expanded weights visualization using SVD factorization.
        This reveals the structure of learned features.
        """
        weights = weights[:num_filters]

        if weights.dim() != 4:
            # For now, only handle conv2d weights
            return None

        out_c, in_c, h, w = weights.shape

        # Reshape for factorization: [out_c, in_c*h*w]
        weights_2d = weights.reshape(out_c, -1)

        # Apply SVD
        try:
            U, S, Vh = torch.linalg.svd(weights_2d, full_matrices=False)

            # Keep only top-k singular values
            k = min(rank, min(U.shape[1], Vh.shape[0]))
            U_k = U[:, :k]
            S_k = S[:k]
            Vh_k = Vh[:k, :]

            # Visualize the factorized components
            fig = plt.figure(figsize=(15, 5 * k))

            for i in range(k):
                # Left factor (output features)
                ax1 = plt.subplot(k, 3, i*3 + 1)
                u_i = U_k[:, i].numpy()
                ax1.barh(range(len(u_i)), u_i)
                ax1.set_title(f'Component {i+1}: Output Weights (σ={S_k[i]:.2f})')
                ax1.set_xlabel('Weight')
                ax1.set_ylabel('Output Channel')

                # Right factor (input features) - reshaped to spatial
                ax2 = plt.subplot(k, 3, i*3 + 2)
                v_i = Vh_k[i, :].reshape(in_c, h, w)

                # Show average across input channels
                v_avg = v_i.mean(dim=0).numpy()
                vmax = np.abs(v_avg).max()
                if vmax == 0:
                    vmax = 1e-8  # Avoid division by zero
                im = ax2.imshow(v_avg, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax2.set_title(f'Spatial Pattern (avg across {in_c} channels)')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2)

                # Show all input channels
                ax3 = plt.subplot(k, 3, i*3 + 3)
                if in_c <= 16:
                    # Create montage of all channels
                    rows = int(np.ceil(np.sqrt(in_c)))
                    cols = int(np.ceil(in_c / rows))
                    montage = np.zeros((rows * h, cols * w))

                    for ch in range(in_c):
                        r = ch // cols
                        c = ch % cols
                        montage[r*h:(r+1)*h, c*w:(c+1)*w] = v_i[ch].numpy()

                    vmax = np.abs(montage).max()
                    if vmax == 0:
                        vmax = 1e-8  # Avoid division by zero
                    im = ax3.imshow(montage, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                    ax3.set_title(f'All {in_c} Input Channels')
                    ax3.axis('off')
                    plt.colorbar(im, ax=ax3)
                else:
                    # Too many channels, just show first 16
                    ax3.text(0.5, 0.5, f'{in_c} channels\n(too many to display)',
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.axis('off')

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error in SVD factorization: {e}")
            return None

    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 encoded PNG."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    def visualize(self, model, layer_name, num_filters, visualization_mode):
        """Main visualization function."""

        # If no layer specified, list available layers
        if not layer_name or layer_name.strip() == "":
            available_layers = self.list_available_layers(model)
            layers_html = "<br>".join(available_layers[:100])  # Limit to first 100
            if len(available_layers) > 100:
                layers_html += f"<br>... and {len(available_layers) - 100} more layers"

            html = f"""
            <div class="flex flex-col gap-2">
                <div class="text-lg font-bold">Available Layers ({len(available_layers)} total):</div>
                <div class="text-sm font-mono">
                    {layers_html}
                </div>
                <div class="text-sm text-gray-600 mt-2">
                    Enter a layer name to visualize its weights.
                </div>
            </div>
            """
            return {"ui": {"html": (html,)}}

        # Extract weights from the specified layer
        try:
            weights = self.get_layer_weights(model, layer_name)
        except Exception as e:
            error_html = f"""
            <div class="flex flex-col gap-2 text-red-600">
                <div class="text-lg font-bold">Error:</div>
                <div>{str(e)}</div>
            </div>
            """
            return {"ui": {"html": (error_html,)}}

        # Generate visualizations
        html_parts = [f"""
        <div class="flex flex-col gap-2">
            <div class="text-lg font-bold">Layer: {layer_name}</div>
            <div class="text-sm">Shape: {list(weights.shape)} | {weights.numel():,} parameters</div>
        """]

        if visualization_mode in ["weights", "both"]:
            fig_weights = self.visualize_weight_matrix(weights, num_filters)
            img_data = self.fig_to_base64(fig_weights)
            html_parts.append(f"""
            <div class="flex flex-col gap-1">
                <div class="font-bold">Weight Matrices:</div>
                <img src="{img_data}">
            </div>
            """)

        if visualization_mode in ["expanded", "both"]:
            fig_expanded = self.visualize_expanded_weights(weights, num_filters)
            if fig_expanded is not None:
                img_data = self.fig_to_base64(fig_expanded)
                html_parts.append(f"""
                <div class="flex flex-col gap-1">
                    <div class="font-bold">Expanded Weights (SVD Factorization):</div>
                    <img src="{img_data}">
                </div>
                """)

        html_parts.append("</div>")
        html = "\n".join(html_parts)

        return {"ui": {"html": (html,)}}


class LTListLayers:
    """
    Helper node to list all available layers in a model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to list layers from"}),
                "filter": ("STRING", {"default": "", "multiline": False,
                                     "tooltip": "Filter layer names (substring match)"}),
            }
        }

    CATEGORY = "LatentTools"
    DESCRIPTION = "List all layers in a model"
    FUNCTION = "list_layers"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_names",)

    def list_layers(self, model, filter=""):
        """List all layers with weights in the model."""
        layers = []

        def recurse_model(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if hasattr(child, 'weight'):
                    layers.append(full_name)
                recurse_model(child, full_name)

        try:
            diffusion_model = model.model.diffusion_model
            recurse_model(diffusion_model)
        except Exception as e:
            return (f"Error: {str(e)}",)

        # Apply filter if specified
        if filter and filter.strip():
            layers = [l for l in layers if filter in l]

        # Return as newline-separated string
        return ("\n".join(layers),)
