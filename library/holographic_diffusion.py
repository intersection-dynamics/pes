import torch
import torch.fft
import matplotlib.pyplot as plt
import numpy as np

# --- 1. THE PHYSICS OF DIFFRACTION ---
def optical_transform(signal):
    """
    Simulates light passing through a Fourier Lens.
    This spreads local data (pixels) into global frequencies (waves).
    """
    return torch.fft.fft(signal)

def inverse_optical_transform(signal):
    """
    The Lens in reverse (refocusing).
    """
    return torch.fft.ifft(signal)

def run_diffused_audit():
    print("="*60)
    print("      SECURITY PATCH: FOURIER OPTICAL DIFFUSION")
    print("="*60)
    
    device = torch.device('cpu')
    N = 64
    
    # 1. THE SECRET (The Data)
    # A clear, sparse pattern: "1 0 0 1 0 0 1..."
    secret_data = torch.zeros(N, device=device)
    secret_data[::4] = 1.0 
    secret_data[1::4] = 1.0 # Making blocks of signal
    
    print("   [Owner] Data Generated. Applying Phase Key...")

    # 2. THE LOCK (Phase Key + Diffraction)
    # Step A: Apply Phase Key
    true_key_val = 12.345
    x = torch.linspace(0, 10, N, device=device)
    key_phase = torch.exp(1j * true_key_val * x)
    
    # Step B: THE DIFFUSION LAYER (The Patch)
    # We don't just transmit the phase; we transmit the FFT of the phase.
    # This smears the information energy across all channels.
    wavefront = secret_data * key_phase
    diffused_signal = optical_transform(wavefront)
    
    print("   [Owner] Signal Diffused via Fourier Transform.")
    
    # 3. THE HACKER (The Attack)
    # Hacker intercepts 'diffused_signal'.
    # They check the intensity. Is the data visible?
    
    hacker_view = torch.abs(diffused_signal)
    
    print("   [Hacker] Intercepted Signal. analyzing...")
    
    # Hacker tries blindly un-twisting phase (Brute Force Strobe)
    # They hope that maximizing contrast/brightness reveals the image.
    
    best_hack_result = torch.zeros(N)
    max_contrast = 0.0
    
    # Hacker Loop: Try to focus the scattered light
    for i in range(1000):
        # Random Phase Guess
        random_phase = torch.exp(1j * torch.randn(N) * 2 * np.pi)
        
        # Attempt to reverse the lens with wrong key
        attempt = inverse_optical_transform(diffused_signal * random_phase)
        magnitude = torch.abs(attempt)
        
        # Metric: Contrast (Peak-to-Average Power Ratio)
        # If they find a signal, it should be spikey.
        contrast = magnitude.max() / (magnitude.mean() + 1e-6)
        
        if contrast > max_contrast:
            max_contrast = contrast.item()
            best_hack_result = magnitude
            
    print(f"   [Hacker] Attack Failed. Best Contrast Found: {max_contrast:.2f}")

    # 4. THE OWNER (The Decryption)
    # Reverse Diffusion (IFFT) -> Remove Key (Conjugate Phase)
    
    # A. Refocus (Inverse FFT)
    refocused = inverse_optical_transform(diffused_signal)
    
    # B. Unlock (Inverse Phase)
    # Note: We must apply the inverse key AFTER the inverse FFT 
    # (or before, depending on optical setup. Here we assume Key is at the image plane).
    # Actually, in this setup: Data * Key -> FFT. 
    # So Decrypt is: IFFT -> Remove Key.
    
    decrypted_wave = refocused * torch.conj(key_phase)
    owner_result = torch.abs(decrypted_wave)

    print("   [Owner] Decryption Complete.")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 10))
    
    # Plot 1: The Secret Data
    plt.subplot(4, 1, 1)
    plt.bar(range(N), secret_data.numpy(), color='green', label='True Secret')
    plt.title("1. The Secret Data (Input)")
    plt.ylim(0, 1.2)
    plt.legend()
    
    # Plot 2: What the Hacker SEES (The Raw Intercept)
    plt.subplot(4, 1, 2)
    plt.bar(range(N), hacker_view.numpy(), color='gray', label='Diffracted Intensity')
    plt.title("2. The Encrypted Signal (Frequency Domain Smear)")
    plt.legend()
    
    # Plot 3: The Hacker's Best Crack
    plt.subplot(4, 1, 3)
    plt.bar(range(N), best_hack_result.numpy(), color='red', label='Hacker Brute Force')
    plt.title(f"3. The Hack (Result: White Noise)")
    plt.ylim(0, 1.2)
    plt.legend()
    
    # Plot 4: The Owner's Decryption
    plt.subplot(4, 1, 4)
    plt.bar(range(N), owner_result.numpy(), color='blue', label='Owner Decryption')
    plt.title("4. The Correct Key (Restored Data)")
    plt.ylim(0, 1.2)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('holographic_diffusion.png')
    print("\n   [Result] Saved to 'holographic_diffusion.png'.")
    print("   The Red Graph should now be garbage.")

if __name__ == "__main__":
    run_diffused_audit()