"""
╔════════════════════════════════════════════════════════════════════════════════╗
║              PHOTONIC IMAGE ENCODER                                            ║
║              Maps images to wavelength-mode representations                    ║
╚════════════════════════════════════════════════════════════════════════════════╝

Strategy:
    - Image is divided into patches (one patch per V1 switch)
    - Each patch is encoded across wavelengths as different features:
        λ₀-λ₇:   Edge orientations (0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°)
        λ₈-λ₁₅:  Spatial frequencies (fine to coarse)
        λ₁₆-λ₂₃: Local contrast at different scales
        λ₂₄-λ₃₁: Intensity/luminance bands
        λ₃₂-λ₄₇: Color channels (if RGB)
        λ₄₈-λ₆₃: Texture/Gabor responses
    - Spatial modes encode position within the patch

This mimics how biological V1 encodes visual information:
    - Different neurons (wavelengths) respond to different features
    - Retinotopic organization (patches map to spatial locations)
"""

import math
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    """Configuration for image encoding."""
    n_wavelengths: int = 64
    n_modes: int = 32
    patch_size: int = 7          # Size of image patches
    n_orientations: int = 8      # Number of edge orientations
    n_frequencies: int = 4       # Number of spatial frequencies
    n_scales: int = 4            # Number of contrast scales


class GaborFilterBank(nn.Module):
    """
    Bank of Gabor filters for edge/texture detection.
    These mimic simple cells in V1.
    """
    
    def __init__(self, config: EncoderConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Create Gabor filters at different orientations and frequencies
        self.filters = self._create_filters()
    
    def _create_filters(self) -> torch.Tensor:
        """Create a bank of Gabor filters."""
        size = self.config.patch_size
        filters = []
        
        # Create coordinate grid
        x = torch.linspace(-1, 1, size, device=self.device)
        y = torch.linspace(-1, 1, size, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        for freq_idx in range(self.config.n_frequencies):
            frequency = 2 + freq_idx * 2  # Spatial frequency
            
            for ori_idx in range(self.config.n_orientations):
                theta = ori_idx * math.pi / self.config.n_orientations
                
                # Rotate coordinates
                x_rot = xx * math.cos(theta) + yy * math.sin(theta)
                y_rot = -xx * math.sin(theta) + yy * math.cos(theta)
                
                # Gabor = Gaussian envelope * sinusoidal carrier
                sigma = 0.3
                gaussian = torch.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
                sinusoid = torch.cos(2 * math.pi * frequency * x_rot)
                
                gabor = gaussian * sinusoid
                gabor = gabor / (gabor.abs().max() + 1e-8)  # Normalize
                
                filters.append(gabor)
        
        # Stack: [n_filters, patch_size, patch_size]
        return torch.stack(filters)
    
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply Gabor filters to a patch.
        
        patch: [patch_size, patch_size]
        returns: [n_filters] response magnitudes
        """
        # Convolve (element-wise multiply and sum)
        responses = (self.filters * patch.unsqueeze(0)).sum(dim=(1, 2))
        return responses.abs()


class PhotonicImageEncoder(nn.Module):
    """
    Encodes images into wavelength-mode representations for the photonic network.
    """
    
    def __init__(self, config: EncoderConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Gabor filter bank for feature extraction
        self.gabor_bank = GaborFilterBank(config, self.device)
        
        # Number of patches that fit in an image (computed per image)
        self.n_patches_h = None
        self.n_patches_w = None
    
    def _extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract non-overlapping patches from an image.
        
        image: [H, W] or [C, H, W]
        returns: [n_patches, patch_size, patch_size]
        """
        if image.dim() == 3:
            # Convert to grayscale if needed
            image = image.mean(dim=0)
        
        H, W = image.shape
        ps = self.config.patch_size
        
        # Pad if necessary
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))
        
        H, W = image.shape
        self.n_patches_h = H // ps
        self.n_patches_w = W // ps
        
        # Reshape into patches
        patches = image.unfold(0, ps, ps).unfold(1, ps, ps)
        patches = patches.contiguous().view(-1, ps, ps)
        
        return patches
    
    def _encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Encode a single patch into wavelength-mode representation.
        
        patch: [patch_size, patch_size]
        returns: [n_wavelengths, n_modes]
        """
        n_λ = self.config.n_wavelengths
        n_m = self.config.n_modes
        ps = self.config.patch_size
        
        # Output tensor
        encoded = torch.zeros(n_λ, n_m, dtype=torch.complex64, device=self.device)
        
        # Flatten patch to modes (resample if needed)
        patch_flat = F.interpolate(
            patch.unsqueeze(0).unsqueeze(0), 
            size=(1, n_m), 
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        # λ₀-λ₇: Edge orientations (from Gabor responses)
        gabor_responses = self.gabor_bank(patch)
        n_gabor = min(8, len(gabor_responses))
        for i in range(n_gabor):
            # Encode response magnitude, modulated by patch structure
            encoded[i, :] = gabor_responses[i] * patch_flat
        
        # λ₈-λ₁₅: Spatial frequencies (FFT of patch)
        fft = torch.fft.fft2(patch)
        fft_flat = F.interpolate(
            fft.abs().unsqueeze(0).unsqueeze(0),
            size=(1, n_m),
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        for i in range(8):
            freq_band = i / 8  # 0 to 1
            weight = torch.exp(-((torch.linspace(0, 1, n_m, device=self.device) - freq_band) ** 2) / 0.1)
            encoded[8 + i, :] = fft_flat * weight
        
        # λ₁₆-λ₂₃: Local contrast at different scales
        for i in range(8):
            scale = 2 ** min(i, 3)  # 1, 2, 4, 8, 8, 8, 8, 8
            scale = max(1, min(scale, ps - 1))
            
            # Simple contrast: std of patch at different blur levels
            if scale > 1:
                # Use simple box blur
                kernel = torch.ones(1, 1, scale, scale, device=self.device) / (scale * scale)
                padded = F.pad(patch.unsqueeze(0).unsqueeze(0), 
                              (scale//2, scale//2, scale//2, scale//2), mode='reflect')
                blurred = F.conv2d(padded, kernel)
                # Crop to original size
                blurred = blurred[:, :, :ps, :ps]
                contrast = (patch.unsqueeze(0).unsqueeze(0) - blurred).abs().squeeze()
            else:
                contrast = patch.abs()
            
            contrast_flat = F.interpolate(
                contrast.unsqueeze(0).unsqueeze(0),
                size=(1, n_m),
                mode='bilinear',
                align_corners=True
            ).squeeze()
            
            encoded[16 + i, :] = contrast_flat
        
        # λ₂₄-λ₃₁: Intensity/luminance bands
        patch_min = patch.min()
        patch_max = patch.max()
        patch_range = patch_max - patch_min + 1e-8
        patch_norm = (patch - patch_min) / patch_range
        
        for i in range(8):
            # Different intensity bands
            band_center = i / 7
            band_width = 0.2
            band_response = torch.exp(-((patch_norm - band_center) ** 2) / (2 * band_width ** 2))
            band_flat = F.interpolate(
                band_response.unsqueeze(0).unsqueeze(0),
                size=(1, n_m),
                mode='bilinear',
                align_corners=True
            ).squeeze()
            encoded[24 + i, :] = band_flat
        
        # λ₃₂-λ₄₇: Additional Gabor responses (different frequencies)
        for i in range(min(16, len(gabor_responses) - 8)):
            encoded[32 + i, :] = gabor_responses[8 + i] * patch_flat
        
        # λ₄₈-λ₆₃: Phase information
        fft_phase = torch.angle(torch.fft.fft2(patch))
        phase_flat = F.interpolate(
            fft_phase.unsqueeze(0).unsqueeze(0),
            size=(1, n_m),
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        for i in range(16):
            phase_band = (i - 8) / 8 * math.pi
            weight = torch.exp(-((phase_flat - phase_band) ** 2) / 0.5)
            encoded[48 + i, :] = weight * patch_flat
        
        return encoded
    
    def encode(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Encode an entire image.
        
        image: [H, W] or [C, H, W]
        returns: (encoded_patches, n_patches)
            encoded_patches: [n_patches, n_wavelengths, n_modes]
            n_patches: number of patches (= number of V1 switches needed)
        """
        patches = self._extract_patches(image)
        n_patches = patches.shape[0]
        
        encoded = torch.zeros(
            n_patches, self.config.n_wavelengths, self.config.n_modes,
            dtype=torch.complex64, device=self.device
        )
        
        for i in range(n_patches):
            encoded[i] = self._encode_patch(patches[i])
        
        return encoded, n_patches
    
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images.
        
        images: [batch, H, W] or [batch, C, H, W]
        returns: [batch, n_patches, n_wavelengths, n_modes]
        """
        batch_size = images.shape[0]
        
        # Encode first image to get n_patches
        first_encoded, n_patches = self.encode(images[0])
        
        all_encoded = torch.zeros(
            batch_size, n_patches, self.config.n_wavelengths, self.config.n_modes,
            dtype=torch.complex64, device=self.device
        )
        
        all_encoded[0] = first_encoded
        
        for b in range(1, batch_size):
            all_encoded[b], _ = self.encode(images[b])
        
        return all_encoded


class MNISTPhotonicsDemo:
    """
    Demo: MNIST classification with the photonic network.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_network_for_mnist(self):
        """Create a photonic network sized for MNIST (28x28) images."""
        from photonic_gpu import (
            TopologyBuilder, SwitchType, GPUConfig, 
            PhotonicNetworkGPU, HolographicMemoryBank
        )
        
        # MNIST is 28x28, with patch_size=7 we get 4x4=16 patches
        patch_size = 7
        n_v1_switches = 16  # One per patch
        
        # Build topology
        builder = TopologyBuilder(n_wavelengths=64, n_modes=32)
        
        # V1: One switch per image patch
        builder.add_region("V1", SwitchType.VISUAL, count=n_v1_switches)
        
        # V2: Combines adjacent patches
        builder.add_region("V2", SwitchType.VISUAL, count=8)
        
        # V4: Higher-level shape features
        builder.add_region("V4", SwitchType.VISUAL, count=6)
        
        # IT: Object/digit representation
        builder.add_region("IT", SwitchType.VISUAL, count=4)
        
        # Memory: Store digit templates
        builder.add_region("MEM", SwitchType.MEMORY, count=10)  # One per digit class
        
        # Readout: Classification
        builder.add_region("OUT", SwitchType.INTEGRATOR, count=10)  # One per class
        
        # Feedforward
        builder.connect_regions("V1", "V2", density=0.6, strength=0.3)
        builder.connect_regions("V2", "V4", density=0.5, strength=0.3)
        builder.connect_regions("V4", "IT", density=0.5, strength=0.3)
        builder.connect_regions("IT", "MEM", density=0.4, strength=0.25)
        builder.connect_regions("MEM", "OUT", density=0.5, strength=0.3)
        
        # Lateral within V1 (neighboring patches talk)
        builder.connect_regions("V1", "V1", density=0.3, strength=0.2)
        
        config, types, conn = builder.build(self.device)
        
        net = PhotonicNetworkGPU(config)
        net.switch_types = types
        net.connectivity = conn
        
        # Encoder
        enc_config = EncoderConfig(
            n_wavelengths=64,
            n_modes=32,
            patch_size=patch_size
        )
        encoder = PhotonicImageEncoder(enc_config, self.device)
        
        return net, encoder, builder
    
    def run_demo(self):
        """Run a demo with synthetic MNIST-like patterns."""
        print("\n" + "=" * 70)
        print("DEMO: MNIST-STYLE IMAGE PROCESSING")
        print("=" * 70)
        
        # Create network
        net, encoder, builder = self.create_network_for_mnist()
        print(f"\n{builder.summary()}")
        
        # Create synthetic digit-like patterns (28x28)
        print("\nCreating synthetic digit patterns...")
        
        digits = {}
        for d in range(10):
            img = torch.zeros(28, 28, device=self.device)
            
            if d == 0:
                # Circle
                xx, yy = torch.meshgrid(
                    torch.linspace(-1, 1, 28, device=self.device),
                    torch.linspace(-1, 1, 28, device=self.device),
                    indexing='ij'
                )
                r = torch.sqrt(xx**2 + yy**2)
                img = ((r > 0.4) & (r < 0.7)).float()
            
            elif d == 1:
                # Vertical line
                img[:, 12:16] = 1.0
            
            elif d == 2:
                # Horizontal bars
                img[5:9, 5:23] = 1.0
                img[12:16, 5:23] = 1.0
                img[19:23, 5:23] = 1.0
            
            elif d == 3:
                # Three horizontal lines connected on right
                img[5:9, 5:23] = 1.0
                img[12:16, 10:23] = 1.0
                img[19:23, 5:23] = 1.0
                img[5:23, 19:23] = 1.0
            
            elif d == 4:
                # L shape + vertical
                img[:14, 5:9] = 1.0
                img[10:14, 5:20] = 1.0
                img[:, 15:19] = 1.0
            
            elif d == 5:
                # S shape
                img[3:7, 5:23] = 1.0
                img[3:14, 5:9] = 1.0
                img[11:15, 5:23] = 1.0
                img[11:24, 19:23] = 1.0
                img[20:24, 5:23] = 1.0
            
            elif d == 6:
                # 6-like
                img[3:7, 5:23] = 1.0
                img[3:24, 5:9] = 1.0
                img[12:16, 5:23] = 1.0
                img[12:24, 19:23] = 1.0
                img[20:24, 5:23] = 1.0
            
            elif d == 7:
                # 7 shape
                img[3:7, 5:23] = 1.0
                img[3:24, 19:23] = 1.0
            
            elif d == 8:
                # 8-like (two circles stacked)
                xx, yy = torch.meshgrid(
                    torch.linspace(-1, 1, 28, device=self.device),
                    torch.linspace(-1, 1, 28, device=self.device),
                    indexing='ij'
                )
                r1 = torch.sqrt(xx**2 + (yy - 0.45)**2)
                r2 = torch.sqrt(xx**2 + (yy + 0.45)**2)
                img = (((r1 > 0.2) & (r1 < 0.4)) | ((r2 > 0.2) & (r2 < 0.4))).float()
            
            elif d == 9:
                # 9-like
                img[3:7, 5:23] = 1.0
                img[3:16, 5:9] = 1.0
                img[12:16, 5:23] = 1.0
                img[3:16, 19:23] = 1.0
                img[12:24, 19:23] = 1.0
            
            digits[d] = img
        
        print(f"  Created {len(digits)} synthetic digit patterns")
        
        # Encode each digit
        print("\nEncoding digits into wavelength-mode representation...")
        
        for d, img in digits.items():
            encoded, n_patches = encoder.encode(img)
            print(f"  Digit {d}: encoded into {n_patches} patches, "
                  f"shape {encoded.shape}")
        
        # Process through network
        print("\nProcessing digits through photonic network...")
        
        net.reset(batch_size=1)
        v1_indices = list(range(builder.regions["V1"]["start_idx"], 
                                builder.regions["V1"]["start_idx"] + builder.regions["V1"]["count"]))
        
        for test_digit in [0, 3, 7]:
            encoded, _ = encoder.encode(digits[test_digit])
            net.reset(batch_size=1)
            
            print(f"\n  Processing digit {test_digit}:")
            
            # Inject encoded patches into V1
            for t in range(30):
                if t < 5:
                    # Inject each patch into corresponding V1 switch
                    for patch_idx, v1_idx in enumerate(v1_indices):
                        if patch_idx < encoded.shape[0]:
                            patch_signal = encoded[patch_idx].unsqueeze(0)  # [1, λ, m]
                            net.inject(v1_idx, patch_signal)
                
                net.step()
                
                if t % 10 == 0:
                    # Read energy at each region
                    energies = {}
                    for name, region in builder.regions.items():
                        start = region["start_idx"]
                        end = start + region["count"]
                        e = (torch.abs(net.output[:, start:end, :, :]) ** 2).sum().item()
                        energies[name] = e
                    
                    print(f"    t={t:2d}:", end="")
                    for name, e in energies.items():
                        print(f" {name}={e:6.0f}", end="")
                    print()
        
        print("\n  ✓ Image encoding and processing complete!")
        print("\n  The signal flows through the visual hierarchy:")
        print("    V1 (patches) → V2 (combinations) → V4 (shapes) → IT (objects)")
        print("    → MEM (template matching) → OUT (classification readout)")
        
        return net, encoder, builder, digits


def demo_encoding():
    """Demonstrate image encoding."""
    print("\n" + "=" * 70)
    print("DEMO: PHOTONIC IMAGE ENCODING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    config = EncoderConfig()
    encoder = PhotonicImageEncoder(config, device)
    
    # Create a test image
    print(f"\nEncoder configuration:")
    print(f"  Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  Wavelengths: {config.n_wavelengths}")
    print(f"  Modes: {config.n_modes}")
    print(f"  Orientations: {config.n_orientations}")
    
    # Create test pattern
    print("\nCreating test image (28x28 with bars pattern)...")
    test_image = torch.zeros(28, 28, device=device)
    test_image[5:10, :] = 1.0
    test_image[15:20, :] = 1.0
    test_image[:, 10:15] = 0.5
    
    # Encode
    encoded, n_patches = encoder.encode(test_image)
    
    print(f"\nEncoding results:")
    print(f"  Image size: 28x28")
    print(f"  Number of patches: {n_patches} ({encoder.n_patches_h}x{encoder.n_patches_w})")
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Total elements: {encoded.numel():,}")
    
    # Show energy distribution across wavelength bands
    print("\nWavelength band energy distribution (first patch):")
    
    bands = [
        ("Edges (λ₀-₇)", 0, 8),
        ("Frequencies (λ₈-₁₅)", 8, 16),
        ("Contrast (λ₁₆-₂₃)", 16, 24),
        ("Intensity (λ₂₄-₃₁)", 24, 32),
        ("Gabor (λ₃₂-₄₇)", 32, 48),
        ("Phase (λ₄₈-₆₃)", 48, 64),
    ]
    
    first_patch = encoded[0]
    total_energy = (torch.abs(first_patch) ** 2).sum().item()
    
    for name, start, end in bands:
        band_energy = (torch.abs(first_patch[start:end, :]) ** 2).sum().item()
        pct = band_energy / total_energy * 100 if total_energy > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {name:25s}: {pct:5.1f}%  {bar}")
    
    return encoder, encoded


if __name__ == "__main__":
    demo_encoding()
    
    print("\n" + "=" * 70)
    print("Running MNIST-style demo...")
    print("=" * 70)
    
    demo = MNISTPhotonicsDemo()
    net, encoder, builder, digits = demo.run_demo()
    
    print("\n" + "=" * 70)
    print("IMAGE ENCODING COMPLETE")
    print("=" * 70)
    print("""
    The photonic image encoder maps images to wavelength-mode representations:
    
    1. PATCH EXTRACTION
       - Image divided into 7x7 patches
       - Each patch → one V1 switch
       - Preserves retinotopic organization
    
    2. WAVELENGTH ENCODING
       - λ₀-₇:   Edge orientations (Gabor responses)
       - λ₈-₁₅:  Spatial frequencies (FFT)
       - λ₁₆-₂₃: Local contrast (multi-scale)
       - λ₂₄-₃₁: Intensity bands
       - λ₃₂-₄₇: Additional texture features
       - λ₄₈-₆₃: Phase information
    
    3. BIOLOGICAL CORRESPONDENCE
       - V1 simple cells → Gabor filters
       - V1 complex cells → Frequency/contrast
       - Retinotopy → Patch organization
       - Wavelength multiplexing → Parallel feature channels
    
    To use with real MNIST:
        from torchvision import datasets, transforms
        mnist = datasets.MNIST('./data', download=True)
        image = mnist[0][0]  # First digit
        encoded, n_patches = encoder.encode(image)
    """)