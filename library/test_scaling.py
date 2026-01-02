"""
Test how many photonic switches fit in 8GB VRAM during training.
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def estimate_memory(n_switches, n_wavelengths=64, n_modes=32, batch_size=32):
    """Estimate memory usage in GB."""
    # Main tensors: field, output (complex64 = 8 bytes)
    tensor_size = batch_size * n_switches * n_wavelengths * n_modes * 8
    main_tensors = 2 * tensor_size  # field + output
    
    # Connectivity (float32 = 4 bytes)
    connectivity = n_switches * n_switches * 4
    
    # Gradients (roughly 2-3x for backward pass)
    grad_multiplier = 3
    
    # Other overhead (wavelength weights, gain, bias, etc.)
    overhead = n_switches * n_wavelengths * 4 * 10
    
    total = (main_tensors + connectivity + overhead) * grad_multiplier
    return total / (1024**3)  # Convert to GB

def test_training_fit(n_switches, n_wavelengths=64, n_modes=32, batch_size=32):
    """Actually try to run a training step and see if it fits."""
    clear_gpu()
    
    device = torch.device("cuda")
    
    try:
        # Allocate tensors similar to training
        field = torch.zeros(batch_size, n_switches, n_wavelengths, n_modes,
                           dtype=torch.complex64, device=device)
        output = torch.zeros_like(field)
        
        # Connectivity (learnable)
        connectivity = nn.Parameter(torch.randn(n_switches, n_switches, device=device) * 0.1)
        
        # Switch parameters
        switch_gain = nn.Parameter(torch.ones(n_switches, device=device))
        switch_bias = nn.Parameter(torch.zeros(n_switches, device=device))
        
        # Wavelength weights
        wavelength_weights = torch.ones(n_switches, n_wavelengths, device=device)
        
        # Readout
        readout = nn.Linear(10, 10).to(device)
        
        # Simulate a forward pass
        input_signal = torch.randn(batch_size, 16, n_wavelengths, n_modes, device=device)
        
        # Inject
        field[:, :16, :, :] = input_signal.to(torch.complex64)
        
        # Several network steps
        for _ in range(8):
            out_mag = torch.sqrt(output.real**2 + output.imag**2 + 1e-8)
            out_t = out_mag.permute(0, 2, 3, 1)
            gathered = torch.matmul(out_t, connectivity)
            gathered = gathered.permute(0, 3, 1, 2)
            gathered = F.relu(gathered)
            
            field_mag = torch.sqrt(field.real**2 + field.imag**2 + 1e-8)
            field_mag = field_mag * 0.8 + gathered * 0.1
            
            weighted = field_mag * wavelength_weights.unsqueeze(0).unsqueeze(-1)
            gain = F.softplus(switch_gain).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            bias = switch_bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            activated = torch.tanh(weighted * gain + bias)
            
            field = field_mag.to(torch.complex64)
            output = activated.to(torch.complex64)
        
        # Readout
        out_energy = (output[:, -10:, :, :].real ** 2).sum(dim=(2, 3))
        logits = readout(torch.log1p(out_energy + 1e-8))
        
        # Backward pass (this is where memory really gets used)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        # Get memory usage
        mem_used = torch.cuda.max_memory_allocated() / (1024**3)
        
        clear_gpu()
        return True, mem_used
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            clear_gpu()
            return False, 0
        raise
    finally:
        clear_gpu()

def find_max_switches():
    """Binary search for maximum switches that fit."""
    print("=" * 60)
    print("PHOTONIC NETWORK SCALING TEST")
    print("=" * 60)
    
    device = torch.device("cuda")
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {total_mem:.1f} GB")
    print()
    
    # Test different configurations
    configs = [
        (64, 32, 32),   # Current: 44 switches equivalent
        (64, 32, 16),   # Smaller batch
        (32, 32, 32),   # Fewer wavelengths
        (64, 16, 32),   # Fewer modes
    ]
    
    for n_wavelengths, n_modes, batch_size in configs:
        print(f"\nConfig: {n_wavelengths} wavelengths, {n_modes} modes, batch {batch_size}")
        print("-" * 50)
        
        # Quick estimates
        print("\nEstimated memory (with gradients):")
        for n in [100, 200, 500, 1000, 2000, 4000]:
            est = estimate_memory(n, n_wavelengths, n_modes, batch_size)
            print(f"  {n:4d} switches: {est:.2f} GB {'✓' if est < total_mem * 0.9 else '✗'}")
        
        # Binary search for actual max
        print("\nActual training test:")
        
        low, high = 50, 4000
        max_working = low
        
        while low <= high:
            mid = (low + high) // 2
            print(f"  Testing {mid} switches... ", end="", flush=True)
            
            success, mem = test_training_fit(mid, n_wavelengths, n_modes, batch_size)
            
            if success:
                print(f"✓ ({mem:.2f} GB)")
                max_working = mid
                low = mid + 1
            else:
                print("✗ OOM")
                high = mid - 1
        
        print(f"\n  → Maximum switches: {max_working}")
        
        # Test the max to get actual memory
        success, mem = test_training_fit(max_working, n_wavelengths, n_modes, batch_size)
        if success:
            print(f"  → Memory at max: {mem:.2f} GB")
            print(f"  → Parameters: ~{max_working * max_working + max_working * 2:,}")
    
    return max_working

if __name__ == "__main__":
    max_switches = find_max_switches()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED ARCHITECTURES")
    print("=" * 60)
    print("""
    For 8GB VRAM, suggested configurations:

    SMALL (fast training):
        V1: 32 → V2: 24 → V4: 16 → IT: 12 → OUT: 10
        Total: 94 switches, ~9K parameters

    MEDIUM (good balance):
        V1: 64 → V2: 48 → V4: 32 → IT: 24 → MEM: 16 → OUT: 10
        Total: 194 switches, ~38K parameters

    LARGE (maximum capacity):
        V1: 128 → V2: 96 → V4: 64 → IT: 48 → MEM: 32 → OUT: 10
        Total: 378 switches, ~143K parameters

    MAXIMUM (if it fits):
        Based on test results above
    """)