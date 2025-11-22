import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
import base64
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def generate_sinusoid_pattern(frequency, orientation, size=224, phase=0):
    """
    Generate oriented sinusoidal grating at specific spatial frequency.

    Args:
        frequency: Spatial frequency (cycles per image)
        orientation: Orientation in degrees (0 = horizontal)
        size: Image size
        phase: Phase offset
    """
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    # Rotate coordinates
    theta = orientation * np.pi / 180
    x_rot = x * np.cos(theta) + y * np.sin(theta)

    # Create sinusoid
    pattern = np.sin(2 * np.pi * frequency * x_rot / size + phase)
    return pattern


def generate_edge_pattern(sharpness, orientation, size=224):
    """
    Generate oriented edge with controllable sharpness.

    Args:
        sharpness: Edge sharpness (higher = sharper, lower = blurrier)
        orientation: Orientation in degrees
        size: Image size
    """
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    # Rotate coordinates
    theta = orientation * np.pi / 180
    x_rot = x * np.cos(theta) + y * np.sin(theta)

    # Create step edge
    edge = (x_rot > size / 2).astype(float)

    # Apply Gaussian blur (lower sharpness = more blur)
    sigma = max(0.1, 20.0 / sharpness) if sharpness > 0 else 10.0
    edge_blurred = gaussian_filter(edge, sigma=sigma)

    # Normalize to [-1, 1]
    edge_blurred = (edge_blurred - 0.5) * 2
    return edge_blurred


def gabor_filter(wavelength, orientation, phase, sigma, gamma, size=11):
    """
    Generate Gabor filter.

    Args:
        wavelength: Wavelength of sinusoid (pixels)
        orientation: Orientation in degrees
        phase: Phase offset
        sigma: Gaussian envelope standard deviation
        gamma: Aspect ratio
        size: Filter size
    """
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(size) - size//2, np.arange(size) - size//2)

    # Rotate
    theta = orientation * np.pi / 180
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor formula
    gaussian = np.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * x_rot / wavelength + phase)

    gabor = gaussian * sinusoid

    # Normalize
    gabor = gabor / (np.abs(gabor).sum() + 1e-8)
    return gabor


class LTFrequencyResponse:
    """
    Measure neuron response across spatial frequencies.
    Determines if neuron is a high-freq, low-freq, or band-pass filter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model"}),
                "layer_name": ("STRING", {"default": "input_blocks.1.1.conv",
                                         "multiline": False}),
                "channel": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "min_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 50.0, "step": 0.1,
                                           "tooltip": "Minimum spatial frequency (cycles per image)"}),
                "max_frequency": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 50.0, "step": 0.1,
                                           "tooltip": "Maximum spatial frequency"}),
                "num_frequencies": ("INT", {"default": 20, "min": 5, "max": 50,
                                            "tooltip": "Number of test frequencies"}),
                "orientation": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 180.0, "step": 15.0,
                                         "tooltip": "Test orientation in degrees"}),
            }
        }

    CATEGORY = "LatentTools"
    DESCRIPTION = "Measure neuron response to different spatial frequencies"
    FUNCTION = "analyze"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def analyze(self, model, layer_name, channel, min_frequency, max_frequency,
                num_frequencies, orientation):
        """Measure frequency response."""

        from .feature_visualization import ActivationHook

        device = model.load_device

        # Generate test frequencies (log scale)
        frequencies = np.logspace(np.log10(min_frequency), np.log10(max_frequency),
                                 num_frequencies)

        # Setup hook
        hook = ActivationHook()
        try:
            hook.register_hook(model.model.diffusion_model, layer_name)
        except Exception as e:
            return {"ui": {"html": (f"<div class='text-red-600'>Error: {str(e)}</div>",)}}

        responses = []

        try:
            for freq in frequencies:
                # Generate sinusoidal pattern
                pattern = generate_sinusoid_pattern(freq, orientation, size=224)

                # Convert to tensor [1, 3, H, W]
                img_tensor = torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0)
                img_tensor = img_tensor.repeat(1, 3, 1, 1).float().to(device)

                # Normalize to [0, 1]
                img_tensor = (img_tensor + 1) / 2

                # Forward pass
                with torch.no_grad():
                    try:
                        # First, validate that input_blocks exists and is accessible
                        if hasattr(model.model.diffusion_model, 'input_blocks'):
                            _ = model.model.diffusion_model.input_blocks[0](img_tensor)
                        else:
                            # Use full model forward pass as primary approach
                            _ = model.model.diffusion_model(img_tensor,
                                                           timesteps=torch.zeros(1, device=device),
                                                           context=None)
                    except (AttributeError, RuntimeError, TypeError) as e:
                        # If both approaches fail, provide informative error
                        raise ValueError(f"Failed to forward pass through model: {str(e)}. "
                                       f"Model structure may be incompatible.")

                    # Get activation
                    if layer_name not in hook.activations:
                        raise ValueError(f"Layer {layer_name} not activated")

                    activation = hook.activations[layer_name]

                    # Extract channel activation
                    if len(activation.shape) == 4:
                        response = activation[0, channel, :, :].mean().item()
                    elif len(activation.shape) == 3:
                        response = activation[0, :, channel].mean().item()
                    else:
                        response = activation[0, channel].mean().item()

                    responses.append(response)

        finally:
            hook.remove_hooks()

        # Analyze response curve
        responses = np.array(responses)
        peak_idx = np.argmax(responses)
        peak_frequency = frequencies[peak_idx]
        peak_response = responses[peak_idx]

        # Classify neuron type
        if peak_idx < len(frequencies) * 0.3:
            neuron_type = "Low-frequency (coarse features)"
        elif peak_idx > len(frequencies) * 0.7:
            neuron_type = "High-frequency (fine details)"
        else:
            neuron_type = "Band-pass (mid-range features)"

        # Create visualization
        html = self.create_visualization(frequencies, responses, peak_frequency,
                                         neuron_type, layer_name, channel, orientation)

        return {"ui": {"html": (html,)}}

    def create_visualization(self, frequencies, responses, peak_frequency,
                            neuron_type, layer_name, channel, orientation):
        """Create frequency response visualization."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Frequency response curve
        ax1.semilogx(frequencies, responses, 'b-o', linewidth=2, markersize=6)
        ax1.axvline(peak_frequency, color='r', linestyle='--', label=f'Peak: {peak_frequency:.2f} cycles/image')
        ax1.set_xlabel('Spatial Frequency (cycles per image)', fontsize=12)
        ax1.set_ylabel('Activation', fontsize=12)
        ax1.set_title(f'Frequency Response\n{neuron_type}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Example patterns at key frequencies
        test_freqs = [frequencies[0], peak_frequency, frequencies[-1]]
        labels = ['Low Freq', 'Peak Freq', 'High Freq']

        for i, (freq, label) in enumerate(zip(test_freqs, labels)):
            pattern = generate_sinusoid_pattern(freq, orientation, size=100)

            # Create subplot
            ax = fig.add_subplot(2, 3, i + 4)
            ax.imshow(pattern, cmap='gray', vmin=-1, vmax=1)
            ax.set_title(f'{label}\n{freq:.1f} cyc/img', fontsize=10)
            ax.axis('off')

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        img_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        html = f"""
        <div class="flex flex-col gap-2">
            <div class="text-lg font-bold">Frequency Response Analysis</div>
            <div class="text-sm">Layer: {layer_name} | Channel: {channel} | Orientation: {orientation}°</div>
            <div class="text-sm font-bold">Classification: {neuron_type}</div>
            <div class="text-sm">Peak Frequency: {peak_frequency:.2f} cycles/image</div>
            <img src="{img_b64}">
        </div>
        """

        return html


class LTEdgeDetectorAnalysis:
    """
    Analyze neuron as oriented edge detector.
    Measures orientation tuning and frequency preference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "layer_name": ("STRING", {"default": "input_blocks.1.1.conv", "multiline": False}),
                "channel": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "num_orientations": ("INT", {"default": 12, "min": 4, "max": 36,
                                            "tooltip": "Number of orientations to test"}),
                "edge_sharpness": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1,
                                            "tooltip": "Edge sharpness (higher = sharper)"}),
            }
        }

    CATEGORY = "LatentTools"
    DESCRIPTION = "Analyze neuron orientation tuning (edge detector properties)"
    FUNCTION = "analyze"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def analyze(self, model, layer_name, channel, num_orientations, edge_sharpness):
        """Measure orientation tuning."""

        from .feature_visualization import ActivationHook

        device = model.load_device

        # Test orientations
        orientations = np.linspace(0, 180, num_orientations, endpoint=False)

        # Setup hook
        hook = ActivationHook()
        try:
            hook.register_hook(model.model.diffusion_model, layer_name)
        except Exception as e:
            return {"ui": {"html": (f"<div class='text-red-600'>Error: {str(e)}</div>",)}}

        responses = []
        test_patterns = []

        try:
            for orientation in orientations:
                # Generate edge pattern
                pattern = generate_edge_pattern(edge_sharpness, orientation, size=224)
                test_patterns.append(pattern)

                # Convert to tensor
                img_tensor = torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0)
                img_tensor = img_tensor.repeat(1, 3, 1, 1).float().to(device)
                img_tensor = (img_tensor + 1) / 2

                # Forward pass
                with torch.no_grad():
                    try:
                        # First, validate that input_blocks exists and is accessible
                        if hasattr(model.model.diffusion_model, 'input_blocks'):
                            _ = model.model.diffusion_model.input_blocks[0](img_tensor)
                        else:
                            # Use full model forward pass as primary approach
                            _ = model.model.diffusion_model(img_tensor,
                                                           timesteps=torch.zeros(1, device=device),
                                                           context=None)
                    except (AttributeError, RuntimeError, TypeError) as e:
                        # If both approaches fail, provide informative error
                        raise ValueError(f"Failed to forward pass through model: {str(e)}. "
                                       f"Model structure may be incompatible.")

                    activation = hook.activations[layer_name]

                    if len(activation.shape) == 4:
                        response = activation[0, channel, :, :].mean().item()
                    elif len(activation.shape) == 3:
                        response = activation[0, :, channel].mean().item()
                    else:
                        response = activation[0, channel].mean().item()

                    responses.append(response)

        finally:
            hook.remove_hooks()

        # Find preferred orientation
        responses = np.array(responses)
        preferred_idx = np.argmax(responses)
        preferred_orientation = orientations[preferred_idx]

        # Compute orientation bandwidth (FWHM)
        half_max = (responses.max() - responses.min()) / 2 + responses.min()
        above_half = responses > half_max
        bandwidth = np.sum(above_half) * (180 / num_orientations)

        # Create visualization
        html = self.create_visualization(orientations, responses, test_patterns,
                                         preferred_orientation, bandwidth,
                                         layer_name, channel, edge_sharpness)

        return {"ui": {"html": (html,)}}

    def create_visualization(self, orientations, responses, patterns,
                            preferred_orientation, bandwidth, layer_name,
                            channel, sharpness):
        """Create orientation tuning visualization."""

        fig = plt.figure(figsize=(15, 6))

        # Plot 1: Polar plot of orientation tuning
        ax1 = plt.subplot(1, 3, 1, projection='polar')

        # Convert to radians and duplicate for full circle
        theta = np.concatenate([orientations, orientations + 180]) * np.pi / 180
        r = np.concatenate([responses, responses])

        ax1.plot(theta, r, 'b-o', linewidth=2)
        ax1.fill(theta, r, alpha=0.3)
        ax1.set_theta_zero_location('E')
        ax1.set_theta_direction(1)
        ax1.set_title(f'Orientation Tuning\nPreferred: {preferred_orientation:.1f}°\nBandwidth: {bandwidth:.1f}°',
                     pad=20)

        # Plot 2: Cartesian plot
        ax2 = plt.subplot(1, 3, 2)
        ax2.plot(orientations, responses, 'b-o', linewidth=2)
        ax2.axvline(preferred_orientation, color='r', linestyle='--',
                   label=f'Preferred: {preferred_orientation:.1f}°')
        ax2.set_xlabel('Orientation (degrees)')
        ax2.set_ylabel('Activation')
        ax2.set_title('Orientation Response')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Example test patterns
        ax3 = plt.subplot(1, 3, 3)

        # Show 4 example orientations
        n_examples = 4
        indices = np.linspace(0, len(patterns)-1, n_examples, dtype=int)

        combined = np.hstack([patterns[i] for i in indices])
        ax3.imshow(combined, cmap='gray', vmin=-1, vmax=1)
        ax3.set_title(f'Test Patterns (sharpness={sharpness:.1f})')
        ax3.axis('off')

        # Add orientation labels
        for i, idx in enumerate(indices):
            x_pos = (i + 0.5) * patterns[0].shape[1]
            ax3.text(x_pos, -10, f'{orientations[idx]:.0f}°',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        img_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        html = f"""
        <div class="flex flex-col gap-2">
            <div class="text-lg font-bold">Edge Detector Analysis</div>
            <div class="text-sm">Layer: {layer_name} | Channel: {channel}</div>
            <div class="text-sm font-bold">Preferred Orientation: {preferred_orientation:.1f}° | Bandwidth: {bandwidth:.1f}°</div>
            <img src="{img_b64}">
        </div>
        """

        return html


class LTGaborFit:
    """
    Fit Gabor filter to convolutional weights.
    Reveals frequency, orientation, and phase properties.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "layer_name": ("STRING", {"default": "input_blocks.1.1.conv", "multiline": False}),
                "channel": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "input_channel": ("INT", {"default": 0, "min": 0, "max": 16,
                                         "tooltip": "Which input channel to analyze (for Conv2d)"}),
            }
        }

    CATEGORY = "LatentTools"
    DESCRIPTION = "Fit Gabor filter to neuron weights"
    FUNCTION = "fit"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def get_layer_weights(self, model, layer_name):
        """Extract weights from layer."""
        diffusion_model = model.model.diffusion_model
        parts = layer_name.split('.')
        current = diffusion_model

        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        if hasattr(current, 'weight'):
            return current.weight.data.cpu()
        else:
            raise ValueError(f"Layer {layer_name} has no weights")

    def fit(self, model, layer_name, channel, input_channel):
        """Fit Gabor to weights."""

        try:
            weights = self.get_layer_weights(model, layer_name)
        except Exception as e:
            return {"ui": {"html": (f"<div class='text-red-600'>Error: {str(e)}</div>",)}}

        # Extract specific filter
        if weights.dim() == 4:  # Conv2d [out_ch, in_ch, h, w]
            if channel >= weights.shape[0] or input_channel >= weights.shape[1]:
                return {"ui": {"html": ("<div class='text-red-600'>Channel index out of range</div>",)}}

            filter_weights = weights[channel, input_channel, :, :].numpy()
        else:
            return {"ui": {"html": ("<div class='text-yellow-600'>Only Conv2d weights supported</div>",)}}

        # Fit Gabor
        size = filter_weights.shape[0]

        try:
            params = self.fit_gabor(filter_weights)
            wavelength, orientation, phase, sigma, gamma = params

            # Generate fitted Gabor
            fitted_gabor = gabor_filter(wavelength, orientation, phase, sigma, gamma, size)

            # Compute fit quality
            residual = filter_weights - fitted_gabor
            mse = np.mean(residual ** 2)
            r_squared = 1 - (np.sum(residual**2) / (np.sum((filter_weights - filter_weights.mean())**2) + 1e-8))

            # Create visualization
            html = self.create_visualization(filter_weights, fitted_gabor, residual,
                                            params, mse, r_squared, layer_name, channel)

        except Exception as e:
            html = f"<div class='text-red-600'>Fitting failed: {str(e)}</div>"

        return {"ui": {"html": (html,)}}

    def fit_gabor(self, weights):
        """Fit Gabor parameters to weights."""
        size = weights.shape[0]

        def gabor_error(params):
            wavelength, orientation, phase, sigma, gamma = params
            gabor = gabor_filter(wavelength, orientation, phase, sigma, gamma, size)
            return np.mean((weights - gabor)**2)

        # Initial guess
        x0 = [size / 3, 45.0, 0.0, size / 4, 0.5]

        # Bounds
        bounds = [
            (2, size),           # wavelength
            (0, 180),            # orientation
            (-np.pi, np.pi),     # phase
            (0.5, size),         # sigma
            (0.1, 2.0)           # gamma
        ]

        result = minimize(gabor_error, x0, bounds=bounds, method='L-BFGS-B')
        return result.x

    def create_visualization(self, weights, fitted, residual, params,
                            mse, r_squared, layer_name, channel):
        """Create Gabor fit visualization."""

        wavelength, orientation, phase, sigma, gamma = params

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        vmax = max(np.abs(weights).max(), np.abs(fitted).max())

        # Actual weights
        im1 = axes[0].imshow(weights, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0].set_title('Learned Weights')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        # Fitted Gabor
        im2 = axes[1].imshow(fitted, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1].set_title('Fitted Gabor')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        # Residual
        im3 = axes[2].imshow(residual, cmap='RdBu_r')
        axes[2].set_title(f'Residual (MSE={mse:.4f})')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        img_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        html = f"""
        <div class="flex flex-col gap-2">
            <div class="text-lg font-bold">Gabor Filter Fit</div>
            <div class="text-sm">Layer: {layer_name} | Channel: {channel}</div>
            <div class="text-sm font-bold">Gabor Parameters:</div>
            <div class="text-sm ml-4">
                • Wavelength (λ): {wavelength:.2f} pixels<br>
                • Orientation (θ): {orientation:.1f}°<br>
                • Phase (φ): {phase:.2f} rad<br>
                • Sigma (σ): {sigma:.2f}<br>
                • Aspect Ratio (γ): {gamma:.2f}
            </div>
            <div class="text-sm">
                <strong>Fit Quality:</strong> R² = {r_squared:.3f} | MSE = {mse:.6f}
            </div>
            <img src="{img_b64}">
        </div>
        """

        return html
