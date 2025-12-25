"""
╔════════════════════════════════════════════════════════════════════════════════╗
║              PHASE-AWARE PHOTONIC TRAINING                                     ║
║              Enabling interference-based computation                           ║
╚════════════════════════════════════════════════════════════════════════════════╝

The magnitude-only training hit ~90% but can't use interference.
This version keeps phase information throughout:

    - Complex connectivity (amplitude AND phase)
    - Phase-preserving activation
    - Constructive/destructive interference between paths
    - Careful numerics to avoid gradient explosions

Key insight: work in (real, imag) space, not (mag, phase) space.
This avoids the discontinuity in angle() at zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import gzip
import urllib.request
import numpy as np

from photonic_gpu import (
    GPUConfig, PhotonicNetworkGPU, TopologyBuilder, SwitchType
)


class PhaseAwareClassifier(nn.Module):
    """
    Photonic classifier with phase-aware training.
    
    Key differences from magnitude-only:
    1. Complex connectivity (learns interference patterns)
    2. Phase-preserving nonlinearity
    3. Operates in (real, imag) not (mag, phase)
    """
    
    def __init__(self, n_classes: int = 10, device: torch.device = None, size: str = "medium"):
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.size = size
        
        # Patch size
        if size == "small":
            self.patch_size = 7
        else:  # medium or large - both use 4x4 = 49 patches
            self.patch_size = 4
        
        # Build topology
        self.builder = self._create_topology()
        config, types, conn = self.builder.build(self.device)
        self.config = config
        
        # Create network
        self.net = PhotonicNetworkGPU(config)
        self.net.switch_types = types
        
        # COMPLEX connectivity - learns both amplitude AND phase
        # Initialize with small random phase
        n_switches = config.n_switches
        conn_mag = conn.real.abs() * 0.3
        conn_phase = torch.randn(n_switches, n_switches, device=self.device) * 0.1
        
        # Store as two real parameters (more stable than complex parameter)
        self.conn_real = nn.Parameter(conn_mag * torch.cos(conn_phase))
        self.conn_imag = nn.Parameter(conn_mag * torch.sin(conn_phase))
        
        # Learnable phase shifts per switch (models optical path length differences)
        self.switch_phase = nn.Parameter(torch.zeros(n_switches, device=self.device))
        
        # Learnable gain (real-valued, applied to magnitude)
        self.switch_gain = nn.Parameter(torch.ones(n_switches, device=self.device))
        
        # Readout 
        out_region = self.builder.regions["OUT"]
        self.readout = nn.Linear(out_region["count"], n_classes).to(self.device)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)
        
        self.regions = self.builder.regions
    
    def _create_topology(self) -> TopologyBuilder:
        """Create network topology."""
        builder = TopologyBuilder(n_wavelengths=64, n_modes=32)
        
        if self.size == "small":
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
            # 131 switches - the baseline that hit 93.6%
            builder.add_region("V1", SwitchType.VISUAL, count=49)
            builder.add_region("V2", SwitchType.VISUAL, count=32)
            builder.add_region("V4", SwitchType.VISUAL, count=24)
            builder.add_region("IT", SwitchType.VISUAL, count=16)
            builder.add_region("OUT", SwitchType.INTEGRATOR, count=10)
            
            builder.connect_regions("V1", "V2", density=0.5, strength=0.35)
            builder.connect_regions("V2", "V4", density=0.5, strength=0.35)
            builder.connect_regions("V4", "IT", density=0.5, strength=0.35)
            builder.connect_regions("IT", "OUT", density=0.6, strength=0.4)
            builder.connect_regions("V1", "V4", density=0.2, strength=0.2)
            builder.connect_regions("V2", "OUT", density=0.2, strength=0.2)
            builder.connect_regions("V1", "V1", density=0.15, strength=0.1)
            builder.connect_regions("V2", "V2", density=0.15, strength=0.1)
        
        elif self.size == "large":
            # ~200 switches - conservative scale-up from medium
            builder.add_region("V1", SwitchType.VISUAL, count=49)     # Match patches
            builder.add_region("V2", SwitchType.VISUAL, count=64)     # 2x medium
            builder.add_region("V4", SwitchType.VISUAL, count=48)     # 2x medium
            builder.add_region("IT", SwitchType.VISUAL, count=32)     # 2x medium
            builder.add_region("OUT", SwitchType.INTEGRATOR, count=10)
            
            # Strong feedforward
            builder.connect_regions("V1", "V2", density=0.5, strength=0.35)
            builder.connect_regions("V2", "V4", density=0.5, strength=0.35)
            builder.connect_regions("V4", "IT", density=0.5, strength=0.35)
            builder.connect_regions("IT", "OUT", density=0.6, strength=0.4)
            
            # Skip connections (critical for gradient flow)
            builder.connect_regions("V1", "V4", density=0.25, strength=0.25)
            builder.connect_regions("V2", "IT", density=0.25, strength=0.25)
            builder.connect_regions("V2", "OUT", density=0.2, strength=0.2)
            builder.connect_regions("V4", "OUT", density=0.25, strength=0.25)
            
            # Lateral refinement
            builder.connect_regions("V1", "V1", density=0.15, strength=0.1)
            builder.connect_regions("V2", "V2", density=0.15, strength=0.1)
            
        return builder
    
    def forward(self, images: torch.Tensor, n_steps: int = 10, injection_steps: int = 4) -> torch.Tensor:
        """Forward pass with phase-aware dynamics."""
        batch_size = images.shape[0]
        
        if images.dim() == 4:
            images = images.squeeze(1)
        
        # Encode images
        encoded = self._encode_batch(images)  # [batch, patches, λ, modes] real
        
        # Normalize
        enc_max = encoded.abs().max()
        if enc_max > 1e-8:
            encoded = encoded / enc_max
        
        # Initialize state as complex
        n_sw = self.config.n_switches
        n_λ = self.config.n_wavelengths
        n_m = self.config.n_modes
        
        # State: (real, imag) separate for numerical stability
        field_r = torch.zeros(batch_size, n_sw, n_λ, n_m, device=self.device)
        field_i = torch.zeros(batch_size, n_sw, n_λ, n_m, device=self.device)
        out_r = torch.zeros_like(field_r)
        out_i = torch.zeros_like(field_i)
        
        # Build complex connectivity
        conn = torch.complex(self.conn_real, self.conn_imag)
        
        # V1 indices
        v1_start = self.regions["V1"]["start_idx"]
        v1_count = self.regions["V1"]["count"]
        
        # Run dynamics
        for t in range(n_steps):
            # Inject input (real-valued encoding becomes real part)
            if t < injection_steps:
                n_patches = min(v1_count, encoded.shape[1])
                # Create injection tensor (avoid inplace modification)
                injection = torch.zeros_like(field_r)
                injection[:, v1_start:v1_start+n_patches, :, :] = encoded[:, :n_patches, :, :] * 0.3
                field_r = field_r + injection
            
            # Phase-aware network step
            field_r, field_i, out_r, out_i = self._phase_step(
                field_r, field_i, out_r, out_i, conn
            )
        
        # Read output energy (magnitude squared)
        out_start = self.regions["OUT"]["start_idx"]
        out_count = self.regions["OUT"]["count"]
        
        out_r_slice = out_r[:, out_start:out_start+out_count, :, :]
        out_i_slice = out_i[:, out_start:out_start+out_count, :, :]
        
        # Energy = |field|^2 = real^2 + imag^2
        energy = (out_r_slice ** 2 + out_i_slice ** 2).sum(dim=(2, 3))
        
        # Log scale and readout
        features = torch.log1p(energy + 1e-8)
        logits = self.readout(features)
        
        return logits
    
    def _phase_step(self, field_r, field_i, out_r, out_i, conn):
        """
        One network step preserving phase.
        
        Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        """
        # Gather with complex connectivity
        # out: [batch, switches, λ, modes]
        # conn: [switches, switches] complex
        
        # Reshape for matmul: [batch, λ, modes, switches]
        out_r_t = out_r.permute(0, 2, 3, 1)
        out_i_t = out_i.permute(0, 2, 3, 1)
        
        # Complex matmul: gathered = out @ conn
        # (out_r + i*out_i) @ (conn_r + i*conn_i)
        # = (out_r @ conn_r - out_i @ conn_i) + i*(out_r @ conn_i + out_i @ conn_r)
        conn_r = conn.real
        conn_i = conn.imag
        
        gathered_r = torch.matmul(out_r_t, conn_r) - torch.matmul(out_i_t, conn_i)
        gathered_i = torch.matmul(out_r_t, conn_i) + torch.matmul(out_i_t, conn_r)
        
        # Back to [batch, switches, λ, modes]
        gathered_r = gathered_r.permute(0, 3, 1, 2)
        gathered_i = gathered_i.permute(0, 3, 1, 2)
        
        # Apply per-switch phase shift (models optical path differences)
        # Multiply by exp(i * phase) = cos(phase) + i*sin(phase)
        phase = self.switch_phase.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        
        gathered_r_new = gathered_r * cos_p - gathered_i * sin_p
        gathered_i_new = gathered_r * sin_p + gathered_i * cos_p
        gathered_r, gathered_i = gathered_r_new, gathered_i_new
        
        # Integrate with decay
        decay = 0.15
        field_r = field_r * (1 - decay) + gathered_r * 0.25
        field_i = field_i * (1 - decay) + gathered_i * 0.25
        
        # Phase-preserving activation: saturate magnitude, preserve phase
        # |z| -> tanh(gain * |z|), angle(z) preserved
        mag = torch.sqrt(field_r ** 2 + field_i ** 2 + 1e-8)
        
        gain = F.softplus(self.switch_gain).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        new_mag = torch.tanh(mag * gain)
        
        # Scale factor to preserve direction
        scale = new_mag / (mag + 1e-8)
        
        out_r = field_r * scale
        out_i = field_i * scale
        
        return field_r, field_i, out_r, out_i
    
    def _encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to patches."""
        batch_size = images.shape[0]
        ps = self.patch_size
        H, W = images.shape[1], images.shape[2]
        
        n_patches_h = H // ps
        n_patches_w = W // ps
        n_patches = n_patches_h * n_patches_w
        n_modes = self.config.n_modes
        n_λ = self.config.n_wavelengths
        
        # Extract patches
        patches = images[:, :n_patches_h*ps, :n_patches_w*ps]
        patches = patches.unfold(1, ps, ps).unfold(2, ps, ps)
        patches = patches.contiguous().view(batch_size, n_patches, ps*ps)
        
        # Pad to n_modes
        if ps*ps < n_modes:
            patches = F.pad(patches, (0, n_modes - ps*ps))
        else:
            patches = patches[:, :, :n_modes]
        
        # Wavelength weights
        w_idx = torch.arange(n_λ, device=self.device, dtype=torch.float32)
        w_weights = 1.0 - torch.abs(w_idx - 32) / 64
        
        encoded = patches.unsqueeze(2) * w_weights.view(1, 1, n_λ, 1)
        return encoded


def download_mnist():
    """Download MNIST."""
    urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    os.makedirs("mnist_data", exist_ok=True)
    
    for name, filename in files.items():
        filepath = os.path.join("mnist_data", filename)
        if not os.path.exists(filepath):
            print(f"  Downloading {filename}...", end=" ", flush=True)
            for base_url in urls:
                try:
                    urllib.request.urlretrieve(base_url + filename, filepath)
                    print("OK")
                    break
                except:
                    continue
    
    def load_images(filename):
        with gzip.open(os.path.join("mnist_data", filename), 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    def load_labels(filename):
        with gzip.open(os.path.join("mnist_data", filename), 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    return (load_images("train-images-idx3-ubyte.gz"),
            load_labels("train-labels-idx1-ubyte.gz"),
            load_images("t10k-images-idx3-ubyte.gz"),
            load_labels("t10k-labels-idx1-ubyte.gz"))


def train_phase_aware(size="medium", n_epochs=30):
    """Train with phase-aware dynamics."""
    
    print("=" * 60)
    print("PHASE-AWARE PHOTONIC TRAINING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Size: {size}")
    
    # Load data
    print("\nLoading MNIST...")
    train_img, train_lbl, test_img, test_lbl = download_mnist()
    
    train_img = torch.from_numpy(train_img)
    train_lbl = torch.from_numpy(train_lbl.astype(np.int64))
    test_img = torch.from_numpy(test_img)
    test_lbl = torch.from_numpy(test_lbl.astype(np.int64))
    
    # Scale data with network size
    if size == "small":
        n_train, n_test = 5000, 1000
        batch_size = 32
        lr = 0.005
    elif size == "medium":
        n_train, n_test = 10000, 2000
        batch_size = 32
        lr = 0.005
    else:  # large
        n_train, n_test = 10000, 2000  # Same as medium, just more capacity
        batch_size = 32
        lr = 0.004
    
    train_loader = DataLoader(
        TensorDataset(train_img[:n_train], train_lbl[:n_train]),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_img[:n_test], test_lbl[:n_test]),
        batch_size=batch_size
    )
    
    print(f"  Train: {n_train}, Test: {n_test}")
    
    # Create model
    print("\nCreating phase-aware classifier...")
    model = PhaseAwareClassifier(n_classes=10, device=device, size=size)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Switches: {model.config.n_switches}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Patch size: {model.patch_size}x{model.patch_size}")
    
    # Optimizer - use configured LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)
    
    best_acc = 0
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            if torch.isnan(logits).any():
                continue
            
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            # Gradient clipping - important for phase stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                if torch.isnan(logits).any():
                    continue
                
                pred = logits.argmax(dim=1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)
        
        train_acc = correct / total if total > 0 else 0
        test_acc = test_correct / test_total if test_total > 0 else 0
        
        scheduler.step(1 - test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " ★"
        else:
            marker = ""
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}: Train {train_acc:.1%}, Test {test_acc:.1%}{marker}  "
              f"(lr={lr:.4f}, {time.time()-t0:.1f}s)")
    
    print("-" * 60)
    print(f"\nBest test accuracy: {best_acc:.1%}")
    print(f"\nPhase-aware training complete!")
    print(f"The network learned complex-valued connectivity (interference patterns)")
    
    return model, best_acc


if __name__ == "__main__":
    import sys
    size = sys.argv[1] if len(sys.argv) > 1 else "medium"
    
    n_epochs = 40  # Same for all sizes
    
    model, acc = train_phase_aware(size=size, n_epochs=n_epochs)