"""
╔════════════════════════════════════════════════════════════════════════════════╗
║      PHASE-AWARE PHOTONIC TRAINING (BRIDGED) — COHERENT READOUT (ROBUST)       ║
║      Homodyne / Coherent Detection with Fault-Tolerant Training                ║
╚════════════════════════════════════════════════════════════════════════════════╝

Goal:
- Preserve phase at the finish line via coherent (I/Q) detection.
- Train stably even though the photonic core is a recurrent dynamical system.

Key engineering additions vs earlier coherent versions:
1) Differentiable complex sanitization (real/imag separately).
2) Receiver-style stabilization on output field:
   RMS AGC + clamp before computing I/Q.
3) Fault-tolerant training loop:
   If a batch produces non-finite logits/loss/gradients:
     - skip the update
     - reduce LR
     - reduce injection scale
     - reset optimizer state (removes poisoned moments)
     - continue
4) Parameter clamps after each step:
   - Wrap/clamp LO phase to [-pi, pi]
   - Clamp photonic connection weights to safe range

Run:
    python train_phase.py large
    python train_phase.py medium
"""

import time
import os
import gzip
import urllib.request
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from photonic_gpu import (
    TopologyBuilder,
    SwitchType,
    TrainablePhotonicNetworkGPU,
)


# ----------------------------
# MNIST (no torchvision)
# ----------------------------

def download_mnist():
    urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs("mnist_data", exist_ok=True)

    for _, filename in files.items():
        path = os.path.join("mnist_data", filename)
        if not os.path.exists(path):
            print(f"  Downloading {filename}...", end=" ", flush=True)
            ok = False
            for base in urls:
                try:
                    urllib.request.urlretrieve(base + filename, path)
                    print("OK")
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                raise RuntimeError(f"Failed to download {filename} from all mirrors.")

    def load_images(filename):
        with gzip.open(os.path.join("mnist_data", filename), "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8, offset=16)
        return arr.reshape(-1, 28, 28).astype(np.float32) / 255.0

    def load_labels(filename):
        with gzip.open(os.path.join("mnist_data", filename), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    return (
        load_images(files["train_images"]),
        load_labels(files["train_labels"]),
        load_images(files["test_images"]),
        load_labels(files["test_labels"]),
    )


# ----------------------------
# Model
# ----------------------------

class PhotonicMNISTClassifier(nn.Module):
    """
    image -> patches -> inject into V1 -> dynamics -> coherent I/Q readout -> linear classifier
    """

    def __init__(self, n_classes: int = 10, device: torch.device = None, size: str = "medium"):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.n_classes = n_classes

        self.patch_size = 7 if size == "small" else 4

        self.builder = self._create_topology()
        config, types, conn = self.builder.build(self.device)

        # speed
        config.integration_steps = 2

        self.net = TrainablePhotonicNetworkGPU(
            config=config,
            init_connectivity=conn,
            conn_scale=0.30,
            init_phase_std=0.10,
        )
        self.net.switch_types = types
        self.net.wavelength_weights = self.net._init_wavelength_weights()

        self.regions = self.builder.regions
        out_region = self.regions["OUT"]
        self.out_count = out_region["count"]

        # Learnable LO phase per output switch (the coherent reference)
        self.lo_phase = nn.Parameter(torch.zeros(self.out_count, device=self.device))

        # Linear head sees I and Q => 2*out_count features
        self.readout = nn.Linear(2 * self.out_count, n_classes).to(self.device)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)

        self.dropout = nn.Dropout(0.15)

        # Stabilization knobs
        self.eps = 1e-6
        self.field_hard_clip = 10.0   # clamp real/imag after AGC
        self.inj_scale = 0.20         # start conservative for coherent training

        # Clamp range for photonic connection parameters (prevents runaway weights)
        self.conn_clip = 2.5

    def _create_topology(self) -> TopologyBuilder:
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

        else:  # large
            builder.add_region("V1", SwitchType.VISUAL, count=49)
            builder.add_region("V2", SwitchType.VISUAL, count=64)
            builder.add_region("V4", SwitchType.VISUAL, count=48)
            builder.add_region("IT", SwitchType.VISUAL, count=32)
            builder.add_region("OUT", SwitchType.INTEGRATOR, count=10)

            builder.connect_regions("V1", "V2", density=0.5, strength=0.35)
            builder.connect_regions("V2", "V4", density=0.5, strength=0.35)
            builder.connect_regions("V4", "IT", density=0.5, strength=0.35)
            builder.connect_regions("IT", "OUT", density=0.6, strength=0.4)

            builder.connect_regions("V1", "V4", density=0.25, strength=0.25)
            builder.connect_regions("V2", "IT", density=0.25, strength=0.25)
            builder.connect_regions("V2", "OUT", density=0.2, strength=0.2)
            builder.connect_regions("V4", "OUT", density=0.25, strength=0.25)

            builder.connect_regions("V1", "V1", density=0.15, strength=0.1)
            builder.connect_regions("V2", "V2", density=0.15, strength=0.1)

        return builder

    def _encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            images = images.squeeze(1)

        B = images.shape[0]
        ps = self.patch_size
        H, W = images.shape[1], images.shape[2]

        n_ph = H // ps
        n_pw = W // ps
        n_patches = n_ph * n_pw

        n_modes = self.net.config.n_modes
        n_l = self.net.config.n_wavelengths

        patches = images[:, : n_ph * ps, : n_pw * ps]
        patches = patches.unfold(1, ps, ps).unfold(2, ps, ps)
        patches = patches.contiguous().view(B, n_patches, ps * ps)

        if ps * ps < n_modes:
            patches = F.pad(patches, (0, n_modes - ps * ps))
        else:
            patches = patches[:, :, :n_modes]

        w_idx = torch.arange(n_l, device=self.device, dtype=torch.float32)
        w_weights = 1.0 - torch.abs(w_idx - (n_l // 2)) / float(n_l)

        encoded = patches.unsqueeze(2) * w_weights.view(1, 1, n_l, 1)
        return encoded

    def _sanitize_real(self, x: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(x)
        return torch.where(finite, x, torch.zeros_like(x))

    def _stabilize_field(self, out_slice: torch.Tensor) -> torch.Tensor:
        # Sanitize real/imag separately (autograd-safe)
        r = self._sanitize_real(out_slice.real)
        i = self._sanitize_real(out_slice.imag)

        # RMS AGC per sample
        amp2 = r * r + i * i
        rms = torch.sqrt(amp2.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps))
        r = r / rms
        i = i / rms

        # Clamp outliers
        r = r.clamp(-self.field_hard_clip, self.field_hard_clip)
        i = i.clamp(-self.field_hard_clip, self.field_hard_clip)

        return torch.complex(r, i)

    def _coherent_IQ_features(self, out_slice: torch.Tensor) -> torch.Tensor:
        out_slice = self._stabilize_field(out_slice)

        Er = out_slice.real
        Ei = out_slice.imag

        # Wrap/clamp LO phase to prevent runaway
        phi = torch.clamp(self.lo_phase, -3.14159265, 3.14159265)
        phi = phi.view(1, self.out_count, 1, 1).to(dtype=Er.dtype)

        c = torch.cos(phi)
        s = torch.sin(phi)

        # Quadrature projection (mean over λ,m)
        I = (Er * c + Ei * s).mean(dim=(2, 3))
        Q = (-Er * s + Ei * c).mean(dim=(2, 3))

        # Signed log compression
        I = torch.sign(I) * torch.log1p(torch.abs(I) + 1e-8)
        Q = torch.sign(Q) * torch.log1p(torch.abs(Q) + 1e-8)

        feats = torch.cat([I, Q], dim=1)
        feats = self._sanitize_real(feats)

        # Per-batch normalization
        mu = feats.mean(dim=0, keepdim=True)
        sig = feats.std(dim=0, keepdim=True).clamp_min(1e-3)
        feats = (feats - mu) / sig

        return feats

    def clamp_parameters_(self):
        """
        Hard safety clamps on parameters after an optimizer step.
        """
        with torch.no_grad():
            # Keep LO phase bounded (equivalent phases wrap, clamp is fine here)
            self.lo_phase.clamp_(-3.14159265, 3.14159265)

            # If photonic core exposes real/imag connectivity, clamp it
            if hasattr(self.net, "conn_real"):
                self.net.conn_real.clamp_(-self.conn_clip, self.conn_clip)
            if hasattr(self.net, "conn_imag"):
                self.net.conn_imag.clamp_(-self.conn_clip, self.conn_clip)

    def forward(self, images: torch.Tensor, n_steps: int = 6, injection_steps: int = 3) -> torch.Tensor:
        B = images.shape[0]
        if images.dim() == 4:
            images = images.squeeze(1)

        encoded = self._encode_batch(images)

        enc_max = encoded.abs().max()
        if enc_max > 1e-8:
            encoded = encoded / enc_max

        self.net.reset(B)

        v1_start = self.regions["V1"]["start_idx"]
        v1_count = self.regions["V1"]["count"]
        n_patches = min(v1_count, encoded.shape[1])

        inj = encoded[:, :n_patches, :, :] * self.inj_scale

        for t in range(n_steps):
            if t < injection_steps:
                for k in range(n_patches):
                    self.net.inject(v1_start + k, inj[:, k, :, :])
            self.net.step()

        out_start = self.regions["OUT"]["start_idx"]
        out_slice = self.net.output[:, out_start:out_start + self.out_count, :, :]

        feats = self._coherent_IQ_features(out_slice)
        feats = self.dropout(feats)

        logits = self.readout(feats)
        return logits


# ----------------------------
# Robust training utilities
# ----------------------------

def reset_optimizer_state_(optimizer: torch.optim.Optimizer):
    """
    Clear Adam moments etc. after a bad batch.
    """
    optimizer.state.clear()


def scale_optimizer_lr_(optimizer: torch.optim.Optimizer, factor: float, min_lr: float = 1e-5):
    for pg in optimizer.param_groups:
        pg["lr"] = max(min_lr, pg["lr"] * factor)


def any_nonfinite_grads(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


# ----------------------------
# Training
# ----------------------------

def train_phase_aware(size: str = "medium", n_epochs: int = 40):
    print("=" * 60)
    print("PHASE-AWARE PHOTONIC TRAINING (BRIDGED) — COHERENT READOUT (ROBUST)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Size: {size}\n")

    print("Loading MNIST...")
    train_img, train_lbl, test_img, test_lbl = download_mnist()

    train_img = torch.from_numpy(train_img)
    train_lbl = torch.from_numpy(train_lbl.astype(np.int64))
    test_img = torch.from_numpy(test_img)
    test_lbl = torch.from_numpy(test_lbl.astype(np.int64))

    if size == "small":
        n_train, n_test = 5000, 1000
        batch_size = 32
    else:
        n_train, n_test = 10000, 2000
        batch_size = 32

    print(f"  Train: {n_train}, Test: {n_test}\n")

    train_loader = DataLoader(
        TensorDataset(train_img[:n_train], train_lbl[:n_train]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        TensorDataset(test_img[:n_test], test_lbl[:n_test]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("Creating bridged coherent classifier...")
    model = PhotonicMNISTClassifier(n_classes=10, device=device, size=size)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PhotonicNetworkGPU initialized on {device}")
    print(f"  Switches: {model.net.config.n_switches}")
    print(f"  Wavelengths: {model.net.config.n_wavelengths}")
    print(f"  Modes: {model.net.config.n_modes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Patch size: {model.patch_size}x{model.patch_size}")
    print(f"  Injection scale (start): {model.inj_scale:.3f}\n")

    # Separate LR for photonic core vs readout/LO:
    # photonic updates must be smaller to avoid dynamical blow-ups
    photonic_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("net."):
            photonic_params.append(p)
        else:
            head_params.append(p)

    # Conservative base rates for coherent training
    lr_photonic = 6e-4
    lr_head = 2e-3

    optimizer = torch.optim.Adam(
        [
            {"params": photonic_params, "lr": lr_photonic},
            {"params": head_params, "lr": lr_head},
        ],
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)

    conn_l2_weight = 1e-6

    eval_n_steps = 6
    eval_injection_steps = 3

    # Slightly tighter clip for coherent stability
    grad_clip = 0.30

    # Fault-tolerance controls
    max_bad_batches_per_epoch = 8
    lr_backoff = 0.75
    inj_backoff = 0.90
    min_inj = 0.10

    print(f"Training for {n_epochs} epochs...")
    print("-" * 60)

    best_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        correct = 0
        total = 0
        bad_batches = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            if not torch.isfinite(logits).all():
                bad_batches += 1
                # Backoff + reset optimizer state
                scale_optimizer_lr_(optimizer, lr_backoff)
                model.inj_scale = max(min_inj, model.inj_scale * inj_backoff)
                reset_optimizer_state_(optimizer)
                model.clamp_parameters_()
                if bad_batches <= 3:
                    print(f"  [warn] non-finite logits -> skip batch | lr*= {lr_backoff:.2f}, inj= {model.inj_scale:.3f}")
                if bad_batches >= max_bad_batches_per_epoch:
                    raise RuntimeError(
                        f"Too many unstable batches in epoch {epoch}. "
                        f"inj_scale={model.inj_scale:.3f}. Consider stronger core limiter in photonic_gpu.py."
                    )
                continue

            loss = F.cross_entropy(logits, labels)

            if conn_l2_weight > 0.0 and hasattr(model.net, "conn_real") and hasattr(model.net, "conn_imag"):
                conn_l2 = (model.net.conn_real.pow(2).mean() + model.net.conn_imag.pow(2).mean())
                loss = loss + conn_l2_weight * conn_l2

            if not torch.isfinite(loss):
                bad_batches += 1
                scale_optimizer_lr_(optimizer, lr_backoff)
                model.inj_scale = max(min_inj, model.inj_scale * inj_backoff)
                reset_optimizer_state_(optimizer)
                model.clamp_parameters_()
                if bad_batches <= 3:
                    print(f"  [warn] non-finite loss -> skip batch | lr*= {lr_backoff:.2f}, inj= {model.inj_scale:.3f}")
                if bad_batches >= max_bad_batches_per_epoch:
                    raise RuntimeError(f"Too many unstable batches in epoch {epoch}.")
                continue

            loss.backward()

            if any_nonfinite_grads(model):
                bad_batches += 1
                optimizer.zero_grad(set_to_none=True)
                scale_optimizer_lr_(optimizer, lr_backoff)
                model.inj_scale = max(min_inj, model.inj_scale * inj_backoff)
                reset_optimizer_state_(optimizer)
                model.clamp_parameters_()
                if bad_batches <= 3:
                    print(f"  [warn] non-finite grads -> skip batch | lr*= {lr_backoff:.2f}, inj= {model.inj_scale:.3f}")
                if bad_batches >= max_bad_batches_per_epoch:
                    raise RuntimeError(f"Too many unstable batches in epoch {epoch}.")
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.clamp_parameters_()

            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0

        # ---- Eval ----
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images, n_steps=eval_n_steps, injection_steps=eval_injection_steps)
                if not torch.isfinite(logits).all():
                    # Eval should never blow up if train is stable.
                    # If it does, backoff for next epoch.
                    scale_optimizer_lr_(optimizer, lr_backoff)
                    model.inj_scale = max(min_inj, model.inj_scale * inj_backoff)
                    reset_optimizer_state_(optimizer)
                    model.clamp_parameters_()
                    print(f"  [warn] non-finite eval logits -> backing off | inj={model.inj_scale:.3f}")
                    # Treat as 0% this epoch
                    test_correct = 0
                    test_total = 1
                    break

                pred = logits.argmax(dim=1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total if test_total > 0 else 0.0
        scheduler.step(1 - test_acc)

        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " ★"

        # report current LRs
        lr0 = optimizer.param_groups[0]["lr"]
        lr1 = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:2d}: Train {train_acc:.1%}, Test {test_acc:.1%}{marker}  "
            f"(lr_core={lr0:.4f}, lr_head={lr1:.4f}, inj={model.inj_scale:.3f}, bad={bad_batches}, {elapsed:.1f}s)"
        )

    print("-" * 60)
    print(f"\nBest test accuracy: {best_acc:.1%}")
    print("\nBridged coherent (homodyne) training complete!")
    return model, best_acc


if __name__ == "__main__":
    import sys
    size = sys.argv[1] if len(sys.argv) > 1 else "medium"
    train_phase_aware(size=size, n_epochs=40)
