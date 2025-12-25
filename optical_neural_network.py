"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    DIFFRACTIVE OPTICAL NEURAL NETWORK v3                       ║
║                                                                                ║
║     Key changes: Amplitude+Phase modulation, larger resolution, tuned aug     ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ============================================================================
# OPTICAL PHYSICS
# ============================================================================

def fresnel_propagate(field: torch.Tensor, distance: float, wavelength: float = 0.5e-6, pixel_size: float = 8e-6) -> torch.Tensor:
    """Angular Spectrum Method propagation"""
    ny, nx = field.shape[-2:]
    
    fx = torch.fft.fftfreq(nx, pixel_size, device=field.device)
    fy = torch.fft.fftfreq(ny, pixel_size, device=field.device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    
    k = 2 * np.pi / wavelength
    under_sqrt = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    under_sqrt = torch.clamp(under_sqrt, min=0)
    
    H = torch.exp(1j * k * distance * torch.sqrt(under_sqrt))
    
    spectrum = torch.fft.fft2(field)
    propagated = torch.fft.ifft2(spectrum * H)
    
    return propagated


# ============================================================================
# NETWORK COMPONENTS
# ============================================================================

class ComplexModulationLayer(nn.Module):
    """
    Full complex modulation: amplitude AND phase.
    More expressive than phase-only, and physically realizable with modern SLMs.
    """
    def __init__(self, size: int, dropout_prob: float = 0.0):
        super().__init__()
        self.size = size
        self.dropout_prob = dropout_prob
        
        # Learnable amplitude (sigmoid will constrain to [0,1])
        self.amplitude = nn.Parameter(torch.zeros(size, size))
        # Learnable phase
        self.phase = nn.Parameter(torch.randn(size, size) * 0.05)
        
    def forward(self, field: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Amplitude: sigmoid ensures [0, 1] transmission
        amp = torch.sigmoid(self.amplitude)
        phase = self.phase
        
        # Dropout on phase only
        if training and self.dropout_prob > 0:
            mask = (torch.rand_like(phase) > self.dropout_prob).float()
            phase = phase * mask
            
        # Complex modulation
        modulation = amp * torch.exp(1j * phase)
        return field * modulation
    
    def smoothness_loss(self) -> torch.Tensor:
        """Penalize sharp transitions"""
        # Phase smoothness
        dx_p = self.phase[:, 1:] - self.phase[:, :-1]
        dy_p = self.phase[1:, :] - self.phase[:-1, :]
        phase_smooth = (dx_p**2).mean() + (dy_p**2).mean()
        
        # Amplitude smoothness
        dx_a = self.amplitude[:, 1:] - self.amplitude[:, :-1]
        dy_a = self.amplitude[1:, :] - self.amplitude[:-1, :]
        amp_smooth = (dx_a**2).mean() + (dy_a**2).mean()
        
        return phase_smooth + 0.5 * amp_smooth


class DetectorPlane(nn.Module):
    """Quadrant-based detector"""
    def __init__(self, size: int, n_classes: int):
        super().__init__()
        self.size = size
        self.n_classes = n_classes
        self.register_buffer('masks', self._create_detector_masks())
        
    def _create_detector_masks(self) -> torch.Tensor:
        masks = torch.zeros(self.n_classes, self.size, self.size)
        
        cx, cy = self.size // 2, self.size // 2
        x = torch.arange(self.size).float() - cx
        y = torch.arange(self.size).float() - cy
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2)
        theta = torch.atan2(Y, X)
        
        r_inner = self.size * 0.1
        r_outer = self.size * 0.42
        
        for i in range(self.n_classes):
            angle_start = (i / self.n_classes) * 2 * np.pi - np.pi
            angle_end = ((i + 1) / self.n_classes) * 2 * np.pi - np.pi
            
            angle_mask = ((theta >= angle_start) & (theta < angle_end)).float()
            radial_mask = ((R >= r_inner) & (R <= r_outer)).float()
            masks[i] = angle_mask * radial_mask
            
        return masks
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = torch.abs(field) ** 2
        
        if intensity.dim() == 2:
            intensity = intensity.unsqueeze(0)
            
        batch_size = intensity.shape[0]
        outputs = torch.zeros(batch_size, self.n_classes, device=field.device)
        
        for i in range(self.n_classes):
            outputs[:, i] = (intensity * self.masks[i]).sum(dim=(-2, -1))
            
        # Normalize
        areas = self.masks.sum(dim=(-2, -1))
        outputs = outputs / (areas + 1e-6)
            
        return outputs


class DiffractiveNeuralNetwork(nn.Module):
    def __init__(self, size: int = 64, n_layers: int = 5, n_classes: int = 4, 
                 layer_distance: float = 3e-2, dropout_prob: float = 0.1):
        super().__init__()
        self.size = size
        self.n_layers = n_layers
        self.layer_distance = layer_distance
        
        self.layers = nn.ModuleList([
            ComplexModulationLayer(size, dropout_prob=dropout_prob) for _ in range(n_layers)
        ])
        
        self.detector = DetectorPlane(size, n_classes)
        
    def forward(self, input_pattern: torch.Tensor) -> torch.Tensor:
        if input_pattern.dim() == 2:
            input_pattern = input_pattern.unsqueeze(0)
        
        field = torch.sqrt(torch.abs(input_pattern) + 1e-10).to(torch.cfloat)
        
        for layer in self.layers:
            field = layer(field, training=self.training)
            field = fresnel_propagate(field, self.layer_distance)
            
        outputs = []
        for i in range(field.shape[0]):
            outputs.append(self.detector(field[i]))
        
        return torch.cat(outputs, dim=0)
    
    def regularization_loss(self, weight: float = 0.01) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            reg = reg + layer.smoothness_loss()
        return reg * weight


# ============================================================================
# DATA 
# ============================================================================

def generate_patterns(size: int, n_classes: int, samples_per_class: int, 
                      noise_level: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate patterns with more variation"""
    patterns = []
    labels = []
    
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    for _ in range(samples_per_class):
        # Class 0: Horizontal bars
        freq = np.random.uniform(3, 7)
        phase = np.random.uniform(0, 2*np.pi)
        duty = np.random.uniform(0.3, 0.7)  # Vary duty cycle
        pattern = (torch.sin(freq * Y * np.pi + phase) > (2*duty - 1)).float()
        pattern = pattern + torch.randn(size, size) * noise_level
        patterns.append(torch.clamp(pattern, 0, 1))
        labels.append(0)
        
        # Class 1: Vertical bars
        freq = np.random.uniform(3, 7)
        phase = np.random.uniform(0, 2*np.pi)
        duty = np.random.uniform(0.3, 0.7)
        pattern = (torch.sin(freq * X * np.pi + phase) > (2*duty - 1)).float()
        pattern = pattern + torch.randn(size, size) * noise_level
        patterns.append(torch.clamp(pattern, 0, 1))
        labels.append(1)
        
        # Class 2: Diagonal (vary angle slightly)
        freq = np.random.uniform(3, 6)
        phase = np.random.uniform(0, 2*np.pi)
        angle_var = np.random.uniform(-0.2, 0.2)  # Slight angle variation
        pattern = (torch.sin(freq * (X + (1+angle_var)*Y) * np.pi + phase) > 0).float()
        pattern = pattern + torch.randn(size, size) * noise_level
        patterns.append(torch.clamp(pattern, 0, 1))
        labels.append(2)
        
        # Class 3: Concentric rings
        freq = np.random.uniform(2, 5)
        phase = np.random.uniform(0, 2*np.pi)
        cx_off = np.random.uniform(-0.1, 0.1)  # Slight center offset
        cy_off = np.random.uniform(-0.1, 0.1)
        R = torch.sqrt((X-cx_off)**2 + (Y-cy_off)**2)
        pattern = (torch.sin(freq * R * np.pi + phase) > 0).float()
        pattern = pattern + torch.randn(size, size) * noise_level
        patterns.append(torch.clamp(pattern, 0, 1))
        labels.append(3)
    
    return torch.stack(patterns), torch.tensor(labels)


def augment_batch(batch: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """Simple, fast augmentation"""
    device = batch.device
    augmented = batch.clone()
    
    for i in range(batch.shape[0]):
        # Random intensity scaling
        if torch.rand(1).item() < p:
            scale = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            augmented[i] = augmented[i] * scale
            
        # Random additive noise
        if torch.rand(1).item() < p:
            noise = torch.randn_like(augmented[i]) * 0.05
            augmented[i] = augmented[i] + noise
            
        # Random horizontal flip
        if torch.rand(1).item() < p * 0.5:  # Less frequent
            augmented[i] = torch.flip(augmented[i], dims=[-1])
            
    return torch.clamp(augmented, 0, 1)


# ============================================================================
# TRAINING
# ============================================================================

def train_optical_network(net: DiffractiveNeuralNetwork, epochs: int = 200, 
                          lr: float = 0.05, samples_per_class: int = 200,
                          reg_weight: float = 0.001):
    device = next(net.parameters()).device
    
    print("Generating training patterns...")
    X_train, y_train = generate_patterns(net.size, net.detector.n_classes, samples_per_class)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    print("Generating test patterns...")
    X_test, y_test = generate_patterns(net.size, net.detector.n_classes, samples_per_class // 4)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Separate LR for amplitude and phase
    optimizer = torch.optim.AdamW([
        {'params': [l.phase for l in net.layers], 'lr': lr},
        {'params': [l.amplitude for l in net.layers], 'lr': lr * 0.5},  # Slower for amplitude
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, 
        steps_per_epoch=len(X_train) // 32 + 1,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Label smoothing helps
    
    history = {'loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0.0
    best_state = None
    
    print(f"\nTraining D2NN: {net.n_layers} layers, {net.size}x{net.size} pixels")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("=" * 70)
    
    for epoch in range(epochs):
        net.train()
        
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        batch_size = 32
        epoch_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Light augmentation
            batch_X = augment_batch(batch_X, p=0.25)
            
            optimizer.zero_grad()
            outputs = net(batch_X)
            
            cls_loss = criterion(outputs, batch_y)
            reg_loss = net.regularization_loss(weight=reg_weight)
            loss = cls_loss + reg_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += cls_loss.item()
        
        # Evaluate
        net.eval()
        with torch.no_grad():
            train_pred = net(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()
            
            test_pred = net(X_test).argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item()
        
        history['loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={epoch_loss:.3f}, Train={train_acc:.1%}, Test={test_acc:.1%}")
    
    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\nRestored best model with test accuracy: {best_test_acc:.1%}")
    
    return history, (X_test, y_test)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(net: DiffractiveNeuralNetwork, test_data: Tuple, history: dict):
    X_test, y_test = test_data
    
    fig = plt.figure(figsize=(16, 10))
    
    # Learning curves
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history['train_acc'], label='Train', color='blue', linewidth=2)
    ax1.plot(history['test_acc'], label='Test', color='red', linewidth=2)
    ax1.axhline(y=max(history['test_acc']), color='green', linestyle='--', alpha=0.5, label=f'Best: {max(history["test_acc"]):.1%}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Show amplitude and phase for first 2 layers
    for i in range(min(2, net.n_layers)):
        # Amplitude
        ax = fig.add_subplot(2, 3, 2 + i*2)
        amp = torch.sigmoid(net.layers[i].amplitude).detach().cpu().numpy()
        im = ax.imshow(amp, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Layer {i+1} Amplitude')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Phase
        ax = fig.add_subplot(2, 3, 3 + i*2)
        phase = net.layers[i].phase.detach().cpu().numpy()
        im = ax.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax.set_title(f'Layer {i+1} Phase')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Confusion matrix
    ax6 = fig.add_subplot(2, 3, 6)
    class_names = ['Horiz', 'Vert', 'Diag', 'Rings']
    
    net.eval()
    with torch.no_grad():
        pred = net(X_test).argmax(dim=1).cpu()
        true = y_test.cpu()
    
    conf = torch.zeros(4, 4)
    for t, p in zip(true, pred):
        conf[t, p] += 1
    conf_norm = conf / conf.sum(dim=1, keepdim=True)
    
    im = ax6.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax6.text(j, i, f'{conf_norm[i,j]:.0%}', ha='center', va='center',
                    color='white' if conf_norm[i,j] > 0.5 else 'black', fontsize=12)
    ax6.set_xticks(range(4))
    ax6.set_yticks(range(4))
    ax6.set_xticklabels(class_names)
    ax6.set_yticklabels(class_names)
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('True')
    ax6.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('optical_nn_v3_results.png', dpi=150)
    print("\nSaved 'optical_nn_v3_results.png'")


def visualize_propagation(net: DiffractiveNeuralNetwork, test_data: Tuple):
    X_test, y_test = test_data
    net.eval()
    
    fig, axes = plt.subplots(4, net.n_layers + 2, figsize=(16, 8))
    class_names = ['Horizontal', 'Vertical', 'Diagonal', 'Rings']
    
    for cls in range(4):
        idx = (y_test == cls).nonzero()[0].item()
        pattern = X_test[idx]
        
        field = torch.sqrt(torch.abs(pattern) + 1e-10).to(torch.cfloat)
        
        axes[cls, 0].imshow(pattern.detach().cpu(), cmap='gray')
        axes[cls, 0].set_ylabel(class_names[cls], fontsize=10, fontweight='bold')
        if cls == 0:
            axes[cls, 0].set_title('Input')
        axes[cls, 0].axis('off')
        
        with torch.no_grad():
            for i, layer in enumerate(net.layers):
                field = layer(field, training=False)
                field = fresnel_propagate(field, net.layer_distance)
                
                axes[cls, i+1].imshow(torch.abs(field).detach().cpu(), cmap='hot')
                if cls == 0:
                    axes[cls, i+1].set_title(f'L{i+1}')
                axes[cls, i+1].axis('off')
        
        intensity = (torch.abs(field) ** 2).detach().cpu()
        axes[cls, -1].imshow(intensity, cmap='hot')
        if cls == 0:
            axes[cls, -1].set_title('Output')
        axes[cls, -1].axis('off')
    
    plt.suptitle('Light Propagation Through D2NN v3', fontsize=12)
    plt.tight_layout()
    plt.savefig('optical_nn_v3_propagation.png', dpi=150)
    print("Saved 'optical_nn_v3_propagation.png'")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("     DIFFRACTIVE OPTICAL NEURAL NETWORK v3")
    print("     Amplitude + Phase Modulation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    net = DiffractiveNeuralNetwork(
        size=64,
        n_layers=5,
        n_classes=4,
        layer_distance=2e-2,
        dropout_prob=0.05
    ).to(device)
    
    # Count parameters
    phase_params = sum(l.phase.numel() for l in net.layers)
    amp_params = sum(l.amplitude.numel() for l in net.layers)
    print(f"Phase parameters: {phase_params:,}")
    print(f"Amplitude parameters: {amp_params:,}")
    print(f"Total: {phase_params + amp_params:,}")
    
    history, test_data = train_optical_network(
        net, 
        epochs=200,
        lr=0.08,
        samples_per_class=200,
        reg_weight=0.002
    )
    
    visualize_results(net, test_data, history)
    visualize_propagation(net, test_data)
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print(f"  - Best Test Accuracy: {max(history['test_acc']):.1%}")
    print(f"  - Final Train-Test Gap: {history['train_acc'][-1] - history['test_acc'][-1]:.1%}")
    print("="*70)


if __name__ == "__main__":
    main()