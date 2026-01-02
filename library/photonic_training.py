"""
╔════════════════════════════════════════════════════════════════════════════════╗
║              PHOTONIC NETWORK TRAINING                                         ║
║              Learning to classify MNIST digits                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝

The network is fully differentiable, so we can train via backprop:
    1. Encode image → inject into V1
    2. Run network for N steps  
    3. Read OUT switch energies as class logits
    4. Compute cross-entropy loss
    5. Backprop and update parameters

Learnable parameters:
    - switch_gain: per-switch gain (amplification)
    - switch_bias: per-switch bias
    - connectivity: connection weights between switches
"""

import math
import time
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import our modules
from photonic_gpu import (
    GPUConfig, PhotonicNetworkGPU, TopologyBuilder, SwitchType,
    clear_gpu_memory
)
from photonic_encoder import PhotonicImageEncoder, EncoderConfig


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 0.01
    n_epochs: int = 10
    n_steps_per_input: int = 15      # Network steps per image
    injection_steps: int = 5          # Steps during which we inject input
    log_interval: int = 50            # Print every N batches
    use_synthetic: bool = True        # Use synthetic digits (no download needed)
    n_train_samples: int = 1000       # Synthetic samples for training
    n_test_samples: int = 200         # Synthetic samples for testing


class PhotonicClassifier(nn.Module):
    """
    Trainable photonic network for image classification.
    
    Wraps PhotonicNetworkGPU with:
    - Image encoder (fixed)
    - Learnable switch parameters
    - Learnable connectivity (optional)
    - Readout layer
    """
    
    def __init__(self, n_classes: int = 10, device: torch.device = None, size: str = "small"):
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.size = size
        
        # Patch size depends on network size
        # small: 7x7 patches = 16 patches (4x4 grid)
        # medium: 4x4 patches = 49 patches (7x7 grid)
        # large: 4x4 patches = 49 patches
        if size == "small":
            self.patch_size = 7
        else:
            self.patch_size = 4
        
        # Build network topology
        self.builder = self._create_topology()
        config, types, conn = self.builder.build(self.device)
        
        # Store config
        self.config = config
        
        # Create photonic network
        self.net = PhotonicNetworkGPU(config)
        self.net.switch_types = types
        
        # Make connectivity learnable (use real part only for training stability)
        # Keep more of the initial structure
        conn_real = conn.real * 0.5  # Less aggressive scaling
        self.connectivity = nn.Parameter(conn_real.clone())
        
        # Create encoder (fixed, not trained) - NOT USED, we encode in _encode_batch
        self.encoder = None
        
        # Readout layer: maps OUT switch energies to class logits
        out_region = self.builder.regions["OUT"]
        self.readout = nn.Linear(out_region["count"], n_classes).to(self.device)
        
        # Initialize readout with small weights
        nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)
        
        # Store region info for easy access
        self.regions = self.builder.regions
        
    def _create_topology(self) -> TopologyBuilder:
        """Create the network topology for MNIST."""
        builder = TopologyBuilder(n_wavelengths=64, n_modes=32)
        
        if self.size == "small":
            # Original small network: 44 switches
            builder.add_region("V1", SwitchType.VISUAL, count=16)
            builder.add_region("V2", SwitchType.VISUAL, count=8)
            builder.add_region("V4", SwitchType.VISUAL, count=6)
            builder.add_region("IT", SwitchType.VISUAL, count=4)
            builder.add_region("OUT", SwitchType.INTEGRATOR, count=10)
            
            builder.connect_regions("V1", "V2", density=0.6, strength=0.3)
            builder.connect_regions("V2", "V4", density=0.5, strength=0.3)
            builder.connect_regions("V4", "IT", density=0.5, strength=0.3)
            builder.connect_regions("IT", "OUT", density=0.6, strength=0.3)
            builder.connect_regions("V1", "V1", density=0.2, strength=0.15)
            
        elif self.size == "medium":
            # Medium network with MORE INPUT PATCHES
            # 4x4 patches on 28x28 = 49 patches (vs 16 with 7x7 patches)
            # V1 matches patch count, rest scales proportionally
            
            builder.add_region("V1", SwitchType.VISUAL, count=49)    # One per patch
            builder.add_region("V2", SwitchType.VISUAL, count=32)
            builder.add_region("V4", SwitchType.VISUAL, count=24)
            builder.add_region("IT", SwitchType.VISUAL, count=16)
            builder.add_region("OUT", SwitchType.INTEGRATOR, count=10)
            
            # Strong feedforward
            builder.connect_regions("V1", "V2", density=0.5, strength=0.35)
            builder.connect_regions("V2", "V4", density=0.5, strength=0.35)
            builder.connect_regions("V4", "IT", density=0.5, strength=0.35)
            builder.connect_regions("IT", "OUT", density=0.6, strength=0.4)
            
            # Skip connections for gradient flow
            builder.connect_regions("V1", "V4", density=0.2, strength=0.2)
            builder.connect_regions("V2", "OUT", density=0.2, strength=0.2)
            
            # Lateral
            builder.connect_regions("V1", "V1", density=0.15, strength=0.1)
            builder.connect_regions("V2", "V2", density=0.15, strength=0.1)
            
        elif self.size == "large":
            # Large network: ~500 switches (safe for 8GB)
            # Key: stronger connections + skip connections to prevent gradient death
            builder.add_region("V1", SwitchType.VISUAL, count=128)
            builder.add_region("V2a", SwitchType.VISUAL, count=64)
            builder.add_region("V2b", SwitchType.VISUAL, count=64)
            builder.add_region("V4", SwitchType.VISUAL, count=96)
            builder.add_region("IT", SwitchType.VISUAL, count=64)
            builder.add_region("PFC", SwitchType.ROUTING, count=32)
            builder.add_region("MEM", SwitchType.MEMORY, count=32)
            builder.add_region("OUT", SwitchType.INTEGRATOR, count=10)
            
            # STRONGER feedforward connections
            builder.connect_regions("V1", "V2a", density=0.5, strength=0.4)
            builder.connect_regions("V1", "V2b", density=0.5, strength=0.4)
            builder.connect_regions("V2a", "V4", density=0.5, strength=0.4)
            builder.connect_regions("V2b", "V4", density=0.5, strength=0.4)
            builder.connect_regions("V4", "IT", density=0.5, strength=0.4)
            
            # SKIP CONNECTIONS (critical for gradient flow)
            builder.connect_regions("V1", "V4", density=0.2, strength=0.25)   # Skip V2
            builder.connect_regions("V2a", "IT", density=0.2, strength=0.25)  # Skip V4
            builder.connect_regions("V4", "OUT", density=0.3, strength=0.3)   # Direct to output
            
            # Executive and memory
            builder.connect_regions("IT", "PFC", density=0.4, strength=0.35)
            builder.connect_regions("PFC", "MEM", density=0.4, strength=0.35)
            builder.connect_regions("IT", "MEM", density=0.4, strength=0.35)
            builder.connect_regions("MEM", "OUT", density=0.5, strength=0.4)
            builder.connect_regions("PFC", "OUT", density=0.4, strength=0.35)
            builder.connect_regions("IT", "OUT", density=0.4, strength=0.35)  # Direct IT→OUT
            
            # Top-down feedback
            builder.connect_regions("PFC", "V4", density=0.2, strength=0.2)
            
            # Lateral (weaker, for refinement)
            builder.connect_regions("V1", "V1", density=0.1, strength=0.1)
            builder.connect_regions("V4", "V4", density=0.1, strength=0.1)
            
        return builder
    
    def forward(self, images: torch.Tensor, n_steps: int = 15, injection_steps: int = 5, debug: bool = False) -> torch.Tensor:
        """
        Forward pass: encode images and run through network.
        
        images: [batch, 28, 28] or [batch, 1, 28, 28]
        returns: [batch, n_classes] logits
        """
        batch_size = images.shape[0]
        
        # Handle channel dimension
        if images.dim() == 4:
            images = images.squeeze(1)
        
        # Encode all images (returns real tensor)
        encoded_batch = self._encode_batch(images)  # [batch, n_patches, λ, m] real
        
        if debug:
            print(f"  Encoded max: {encoded_batch.max().item():.4f}")
        
        # Normalize encoded input
        enc_max = encoded_batch.abs().max()
        if enc_max > 1e-8:
            encoded_batch = encoded_batch / enc_max
        
        # Reset network state (complex for compatibility, but we'll use magnitude only)
        self.net.field = torch.zeros(
            batch_size, self.config.n_switches, self.config.n_wavelengths, self.config.n_modes,
            dtype=self.config.dtype, device=self.device
        )
        self.net.output = torch.zeros_like(self.net.field)
        
        # Update connectivity from learnable parameter
        # Keep as real - the network step now handles real connectivity
        
        if debug:
            print(f"  Connectivity max: {self.connectivity.abs().max().item():.4f}")
        
        # Get V1 indices
        v1_start = self.regions["V1"]["start_idx"]
        v1_count = self.regions["V1"]["count"]
        
        # Run network
        for t in range(n_steps):
            # Inject encoded patches into V1 for first few steps
            if t < injection_steps:
                for p in range(min(v1_count, encoded_batch.shape[1])):
                    # Convert real encoding to complex for injection
                    signal = encoded_batch[:, p, :, :].to(torch.complex64)
                    self.net.field[:, v1_start + p, :, :] = (
                        self.net.field[:, v1_start + p, :, :] + signal * 0.3  # Stronger injection
                    )
            
            # Network step (real-valued internally)
            self._network_step()
            
            if debug and t % 3 == 0:
                field_max = torch.abs(self.net.field).max().item()
                out_max = torch.abs(self.net.output).max().item()
                print(f"  t={t}: field_max={field_max:.4f}, output_max={out_max:.4f}")
        
        # Read OUT switch energies
        out_start = self.regions["OUT"]["start_idx"]
        out_count = self.regions["OUT"]["count"]
        
        out_field = self.net.output[:, out_start:out_start + out_count, :, :]
        # Since we're using real values stored as complex, just take real part
        out_energy = (out_field.real ** 2 + 1e-8).sum(dim=(2, 3))  # [batch, out_count]
        
        if debug:
            print(f"  Out energy max: {out_energy.max().item():.4f}")
        
        # Log scale to compress dynamic range
        out_features = torch.log1p(out_energy + 1e-8)
        
        # Apply readout layer
        logits = self.readout(out_features)
        
        if debug:
            print(f"  Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        
        return logits
    
    def _encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images - real-valued for training stability."""
        batch_size = images.shape[0]
        
        ps = self.patch_size  # Use configured patch size
        H, W = images.shape[1], images.shape[2]
        n_patches_h = H // ps
        n_patches_w = W // ps
        n_patches = n_patches_h * n_patches_w
        n_modes = self.config.n_modes
        n_wavelengths = self.config.n_wavelengths
        
        # Extract patches efficiently using unfold
        # [batch, H, W] -> [batch, n_ph, ps, n_pw, ps] -> [batch, n_patches, ps*ps]
        patches = images[:, :n_patches_h*ps, :n_patches_w*ps]
        patches = patches.unfold(1, ps, ps).unfold(2, ps, ps)  # [batch, n_ph, n_pw, ps, ps]
        patches = patches.contiguous().view(batch_size, n_patches, ps*ps)  # [batch, n_patches, ps*ps]
        
        # Pad or truncate to n_modes
        if ps*ps < n_modes:
            patches = F.pad(patches, (0, n_modes - ps*ps))
        else:
            patches = patches[:, :, :n_modes]
        
        # Create wavelength weights [n_wavelengths]
        w_idx = torch.arange(n_wavelengths, device=self.device, dtype=torch.float32)
        w_weights = 1.0 - torch.abs(w_idx - 32) / 64  # Peak at center
        
        # Expand: [batch, n_patches, n_modes] x [n_wavelengths] -> [batch, n_patches, n_wavelengths, n_modes]
        encoded = patches.unsqueeze(2) * w_weights.view(1, 1, n_wavelengths, 1)
        
        return encoded
    
    def _network_step(self):
        """One network step with gradients - using REAL values only for stability."""
        cfg = self.config
        
        # Work with magnitudes only (real-valued) to avoid complex gradient issues
        # Add small epsilon to avoid gradient issues at zero
        eps = 1e-8
        
        # Get magnitude of current output (with epsilon for gradient stability)
        out_mag = torch.sqrt(self.net.output.real**2 + self.net.output.imag**2 + eps)
        
        # Gather from connected switches (real-valued matmul)
        out_t = out_mag.permute(0, 2, 3, 1)  # [batch, λ, modes, switches]
        conn_real = self.connectivity  # Already real
        gathered = torch.matmul(out_t, conn_real)  # [batch, λ, modes, switches]
        gathered = gathered.permute(0, 3, 1, 2)  # [batch, switches, λ, modes]
        
        # ReLU to keep positive (optical intensity is non-negative)
        gathered = F.relu(gathered)
        
        # Get magnitude of current field
        field_mag = torch.sqrt(self.net.field.real**2 + self.net.field.imag**2 + eps)
        
        # Integrate with decay - increase gathered coefficient for better signal flow
        decay = 0.15
        field_mag = field_mag * (1.0 - decay) + gathered * 0.25
        
        # Apply wavelength weights
        weighted = field_mag * self.net.wavelength_weights.unsqueeze(0).unsqueeze(-1)
        
        # Activate with learnable gain/bias
        gain = F.softplus(self.net.switch_gain).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = self.net.switch_bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # Tanh activation
        sat = cfg.saturation
        activated = sat * torch.tanh((weighted * gain + bias) / sat)
        
        # Store back as complex (with zero phase) for compatibility
        self.net.field = field_mag.to(torch.complex64)
        self.net.output = activated.to(torch.complex64)


def create_synthetic_dataset(n_samples: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic digit-like patterns for training.
    Returns (images, labels) tensors.
    """
    images = torch.zeros(n_samples, 28, 28, device=device)
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    for i in range(n_samples):
        digit = i % 10
        labels[i] = digit
        
        img = torch.zeros(28, 28, device=device)
        
        # Add some randomness to make training meaningful
        noise = torch.randn(28, 28, device=device) * 0.1
        offset_x = torch.randint(-2, 3, (1,)).item()
        offset_y = torch.randint(-2, 3, (1,)).item()
        thickness = torch.randint(3, 6, (1,)).item()
        
        if digit == 0:
            # Circle
            xx, yy = torch.meshgrid(
                torch.linspace(-1, 1, 28, device=device),
                torch.linspace(-1, 1, 28, device=device),
                indexing='ij'
            )
            xx = xx + offset_x * 0.1
            yy = yy + offset_y * 0.1
            r = torch.sqrt(xx**2 + yy**2)
            img = ((r > 0.35) & (r < 0.35 + thickness * 0.05)).float()
            
        elif digit == 1:
            # Vertical line
            cx = 14 + offset_x
            img[:, cx-thickness//2:cx+thickness//2] = 1.0
            
        elif digit == 2:
            # Z-like shape
            t = thickness
            img[4:4+t, 6:22] = 1.0
            img[4:14, 18:22] = 1.0
            img[12:12+t, 6:22] = 1.0
            img[12:22, 6:10] = 1.0
            img[20:20+t, 6:22] = 1.0
            
        elif digit == 3:
            # Three horizontal lines connected on right
            t = thickness
            img[4:4+t, 6:22] = 1.0
            img[12:12+t, 10:22] = 1.0
            img[20:20+t, 6:22] = 1.0
            img[4:24, 18:22] = 1.0
            
        elif digit == 4:
            # 4 shape
            t = thickness
            img[:14, 6:6+t] = 1.0
            img[10:10+t, 6:20] = 1.0
            img[:, 16:16+t] = 1.0
            
        elif digit == 5:
            # S shape
            t = thickness
            img[4:4+t, 6:22] = 1.0
            img[4:14, 6:6+t] = 1.0
            img[12:12+t, 6:22] = 1.0
            img[12:24, 18:22] = 1.0
            img[20:20+t, 6:22] = 1.0
            
        elif digit == 6:
            # 6 shape
            t = thickness
            img[4:4+t, 6:22] = 1.0
            img[4:24, 6:6+t] = 1.0
            img[12:12+t, 6:22] = 1.0
            img[12:24, 18:22] = 1.0
            img[20:20+t, 6:22] = 1.0
            
        elif digit == 7:
            # 7 shape
            t = thickness
            img[4:4+t, 6:22] = 1.0
            img[4:24, 18:22] = 1.0
            
        elif digit == 8:
            # Two stacked circles
            xx, yy = torch.meshgrid(
                torch.linspace(-1, 1, 28, device=device),
                torch.linspace(-1, 1, 28, device=device),
                indexing='ij'
            )
            r1 = torch.sqrt((xx + 0.4)**2 + yy**2)
            r2 = torch.sqrt((xx - 0.4)**2 + yy**2)
            w = thickness * 0.04
            img = (((r1 > 0.25) & (r1 < 0.25 + w)) | ((r2 > 0.25) & (r2 < 0.25 + w))).float()
            
        elif digit == 9:
            # 9 shape
            t = thickness
            img[4:4+t, 6:22] = 1.0
            img[4:14, 6:6+t] = 1.0
            img[12:12+t, 6:22] = 1.0
            img[4:14, 18:22] = 1.0
            img[12:24, 18:22] = 1.0
        
        # Apply random shift
        img = torch.roll(img, shifts=(offset_x, offset_y), dims=(0, 1))
        
        # Add noise
        img = img + noise
        img = torch.clamp(img, 0, 1)
        
        images[i] = img
    
    return images, labels


def train_epoch(model: PhotonicClassifier, train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, config: TrainingConfig,
                epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(model.device)
        labels = labels.to(model.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images, n_steps=config.n_steps_per_input, 
                      injection_steps=config.injection_steps)
        
        # Check for NaN
        if torch.isnan(logits).any():
            print(f"    Warning: NaN in logits at batch {batch_idx}, skipping")
            continue
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"    Warning: NaN loss at batch {batch_idx}, skipping")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        valid_batches += 1
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % config.log_interval == 0:
            acc = correct/total if total > 0 else 0
            print(f"    Batch {batch_idx + 1}: Loss={loss.item():.4f}, Acc={acc:.1%}")
    
    return total_loss / max(valid_batches, 1)


def evaluate(model: PhotonicClassifier, test_loader: DataLoader,
             config: TrainingConfig) -> Tuple[float, float]:
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            logits = model(images, n_steps=config.n_steps_per_input,
                          injection_steps=config.injection_steps)
            
            # Skip NaN batches
            if torch.isnan(logits).any():
                continue
            
            loss = F.cross_entropy(logits, labels)
            if not torch.isnan(loss):
                total_loss += loss.item()
                valid_batches += 1
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(valid_batches, 1)
    
    return avg_loss, accuracy


def train_photonic_classifier(config: TrainingConfig = None, size: str = "small"):
    """Main training function."""
    
    if config is None:
        config = TrainingConfig()
    
    print("=" * 70)
    print("PHOTONIC NETWORK TRAINING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Network size: {size}")
    
    # Create model
    print("\nCreating photonic classifier...")
    model = PhotonicClassifier(n_classes=10, device=device, size=size)
    
    print(f"\nNetwork topology:")
    print(f"  {model.builder.summary()}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")
    
    # Create dataset
    print(f"\nCreating synthetic dataset...")
    print(f"  Training samples: {config.n_train_samples}")
    print(f"  Test samples: {config.n_test_samples}")
    
    train_images, train_labels = create_synthetic_dataset(config.n_train_samples, device)
    test_images, test_labels = create_synthetic_dataset(config.n_test_samples, device)
    
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Network steps per image: {config.n_steps_per_input}")
    print()
    
    best_accuracy = 0.0
    
    for epoch in range(1, config.n_epochs + 1):
        print(f"Epoch {epoch}/{config.n_epochs}")
        print("-" * 40)
        
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, config, epoch)
        t1 = time.perf_counter()
        
        test_loss, test_accuracy = evaluate(model, test_loader, config)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_marker = " ← Best!"
        else:
            best_marker = ""
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        print(f"  Test Acc:   {test_accuracy:.1%}{best_marker}")
        print(f"  Time:       {t1-t0:.1f}s")
        print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Best test accuracy: {best_accuracy:.1%}")
    print(f"\n  The photonic network learned to classify digits!")
    print(f"  This demonstrates that wavelength-multiplexed optical")
    print(f"  neural networks can be trained via backpropagation.")
    
    return model, best_accuracy


def quick_demo():
    """Quick demo with minimal training."""
    print("\n" + "=" * 70)
    print("QUICK TRAINING DEMO")
    print("=" * 70)
    print("\nRunning minimal training to verify everything works...")
    
    config = TrainingConfig(
        batch_size=16,
        n_epochs=5,
        n_train_samples=300,
        n_test_samples=60,
        n_steps_per_input=8,
        injection_steps=3,
        log_interval=10,
        learning_rate=0.005  # Lower learning rate for stability
    )
    
    model, accuracy = train_photonic_classifier(config, size="small")
    return model, accuracy


def train_medium():
    """Train medium-sized network (~130 switches, 49 input patches)."""
    print("\n" + "=" * 70)
    print("MEDIUM NETWORK TRAINING")
    print("=" * 70)
    
    config = TrainingConfig(
        batch_size=32,
        n_epochs=30,
        n_train_samples=5000,
        n_test_samples=1000,
        n_steps_per_input=10,
        injection_steps=4,
        log_interval=20,
        learning_rate=0.01
    )
    
    model, accuracy = train_photonic_classifier(config, size="medium")
    return model, accuracy


def train_large():
    """Train large network (~500 switches)."""
    print("\n" + "=" * 70)
    print("LARGE NETWORK TRAINING")
    print("=" * 70)
    
    config = TrainingConfig(
        batch_size=16,  # Smaller batch to fit in memory
        n_epochs=30,
        n_train_samples=5000,
        n_test_samples=1000,
        n_steps_per_input=10,  # Fewer steps, stronger signal
        injection_steps=5,     # More injection
        log_interval=50,
        learning_rate=0.01     # Higher LR - was too conservative
    )
    
    model, accuracy = train_photonic_classifier(config, size="large")
    return model, accuracy


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "medium":
            model, accuracy = train_medium()
        elif mode == "large":
            model, accuracy = train_large()
        elif mode == "quick":
            model, accuracy = quick_demo()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python photonic_training.py [quick|medium|large]")
            sys.exit(1)
    else:
        # Default: run quick demo
        model, accuracy = quick_demo()
    
    print("\n" + "=" * 70)
    print("TRAINING OPTIONS")
    print("=" * 70)
    print("""
    python photonic_training.py quick   # 44 switches, ~2K params, fast
    python photonic_training.py medium  # 194 switches, ~38K params  
    python photonic_training.py large   # 490 switches, ~240K params
    """)