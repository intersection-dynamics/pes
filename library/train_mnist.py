"""
Train photonic network on real MNIST handwritten digits.
Downloads automatically on first run. No torchvision needed.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import gzip
import urllib.request
import numpy as np

from photonic_training import PhotonicClassifier


def download_mnist():
    """Download MNIST dataset directly (no torchvision needed)."""
    # Try multiple sources
    urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",  # PyTorch mirror (faster)
        "http://yann.lecun.com/exdb/mnist/",  # Original (can be slow)
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
                except Exception as e:
                    continue
            else:
                print(f"FAILED - try manually downloading from {urls[0]}")
                raise RuntimeError(f"Could not download {filename}")
    
    # Load images
    def load_images(filename):
        with gzip.open(os.path.join("mnist_data", filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    def load_labels(filename):
        with gzip.open(os.path.join("mnist_data", filename), 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    train_images = load_images("train-images-idx3-ubyte.gz")
    train_labels = load_labels("train-labels-idx1-ubyte.gz")
    test_images = load_images("t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("t10k-labels-idx1-ubyte.gz")
    
    return train_images, train_labels, test_images, test_labels


def main():
    import sys
    
    # Get network size from command line
    size = "small"
    if len(sys.argv) > 1:
        size = sys.argv[1].lower()
        if size not in ["small", "medium", "large"]:
            print(f"Unknown size: {size}")
            print("Usage: python train_mnist.py [small|medium|large]")
            sys.exit(1)
    
    print("=" * 60)
    print("PHOTONIC NETWORK ON REAL MNIST")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Network size: {size}")
    
    # Download MNIST
    print("\nDownloading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = download_mnist()
    
    print(f"  Training images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
    
    # Convert to tensors
    train_images = torch.from_numpy(train_images)
    train_labels = torch.from_numpy(train_labels.astype(np.int64))
    test_images = torch.from_numpy(test_images)
    test_labels = torch.from_numpy(test_labels.astype(np.int64))
    
    # Use more data for larger networks
    if size == "small":
        n_train = 5000
        n_test = 1000
        n_epochs = 20
        lr = 0.01
    elif size == "medium":
        n_train = 10000
        n_test = 2000
        n_epochs = 30
        lr = 0.01
    else:  # large
        n_train = 10000
        n_test = 2000
        n_epochs = 30
        lr = 0.008
    
    train_dataset = TensorDataset(train_images[:n_train], train_labels[:n_train])
    test_dataset = TensorDataset(test_images[:n_test], test_labels[:n_test])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"\nUsing subset for training:")
    print(f"  Training: {n_train}")
    print(f"  Test: {n_test}")
    
    # Create model
    print("\nCreating photonic classifier...")
    model = PhotonicClassifier(n_classes=10, device=device, size=size)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Switches: {model.config.n_switches}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Patch size: {model.patch_size}x{model.patch_size}")
    
    n_patches = (28 // model.patch_size) ** 2
    print(f"  Input patches: {n_patches}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    best_acc = 0
    
    print(f"\nTraining for {n_epochs} epochs (lr={lr})...")
    print("-" * 60)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Images are already [batch, 28, 28]
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images, n_steps=8, injection_steps=3)
            
            if torch.isnan(logits).any():
                continue
                
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                
                logits = model(images, n_steps=8, injection_steps=3)
                if torch.isnan(logits).any():
                    continue
                    
                pred = logits.argmax(dim=1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)
        
        train_acc = correct / total if total > 0 else 0
        test_acc = test_correct / test_total if test_total > 0 else 0
        
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " ★"
        else:
            marker = ""
        
        print(f"Epoch {epoch:2d}: Train {train_acc:.1%}, Test {test_acc:.1%}{marker}  ({time.time()-t0:.1f}s)")
    
    print("-" * 60)
    print(f"\nBest test accuracy: {best_acc:.1%}")
    print(f"\nThis is on REAL handwritten digits!")
    print(f"Network: {model.config.n_switches} photonic switches, {n_params:,} parameters")
    print(f"Patch size: {model.patch_size}x{model.patch_size} ({n_patches} patches)")
    
    print("\n" + "=" * 60)
    print("USAGE:")
    print("  python train_mnist.py small   # 44 switches, 16 patches")
    print("  python train_mnist.py medium  # 131 switches, 49 patches")
    print("=" * 60)


if __name__ == "__main__":
    main()