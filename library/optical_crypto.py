"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                       THE HOLOGRAPHIC GLASS SAFE                               ║
║                                                                                ║
║           "Security through Physics: Hiding data in the phase noise."          ║
╚════════════════════════════════════════════════════════════════════════════════╝

THE MISSION:
1. Create a "Secret" visual pattern.
2. Encrypt it into the hologram using a specific 'Dispersion Key' (The Password).
3. Try to view it with the WRONG key (Hacker).
4. View it with the RIGHT key (Owner).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from holographic_core import HolographicMemory, MemoryConfig

def generate_secret_symbol(device):
    """Generates a distinct 'X' shape pattern"""
    modes = 64
    x = torch.linspace(-3, 3, modes, device=device)
    # Create two crossing Gaussians to make an X
    slash = torch.exp(-(x - 0.5)**2 / 0.1)
    back_slash = torch.exp(-(x + 0.5)**2 / 0.1)
    # Combine and make complex
    symbol = (slash + back_slash).to(torch.cfloat)
    # Add some phase complexity
    symbol = symbol * torch.exp(1j * x)
    return symbol

def crack_the_safe(brain, target_key_idx, attempt_password):
    """
    Tries to read a specific memory address using a specific dispersion password.
    Note: We bypass associative recall and read the 'Address' directly to test the encryption.
    """
    # 1. Get the Raw Hologram
    hologram = brain.hologram.hologram
    
    # 2. Get the Addressing Key for the slot we want to read
    # (In a real system, this would be the 'Reference Beam')
    address_key = brain.hologram.keys[target_key_idx]
    
    # 3. Extract the slice (Raw, twisted data)
    # This isolates the signal from the other memories, but it's still Twisted.
    raw_slice = hologram * address_key.unsqueeze(1)
    
    # 4. Apply the Decryption Lens (The Password)
    # We apply the inverse phase twist of our guessed password.
    twist = torch.linspace(0, 2*np.pi, brain.config.n_wavelengths, device=hologram.device)
    decryption_lens = torch.exp(-1j * twist * attempt_password).unsqueeze(1)
    
    decrypted_signal = raw_slice * decryption_lens
    
    return decrypted_signal

def run_crypto_demo():
    print("="*60)
    print("          OPTICAL CRYPTO: THE GLASS SAFE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    brain = HolographicMemory(MemoryConfig()).to(device)
    
    # --- STEP 1: ENCRYPT ---
    print("\n[1] Encrypting Secret Data...")
    secret = generate_secret_symbol(device)
    
    # THE PASSWORD
    REAL_PASSWORD = 7.532  # An arbitrary, specific float
    
    brain.encode(secret, dispersion_val=REAL_PASSWORD)
    print(f"    -> Secret encoded with Phase Key: {REAL_PASSWORD}")
    
    # --- STEP 2: THE HACKER (Wrong Password) ---
    print("\n[2] Intruder Attempt (Standard Lens)...")
    wrong_password = 1.0
    hacker_view = crack_the_safe(brain, 0, wrong_password)
    
    # Measure energy concentration (Focus)
    # A focused signal has high peaks; noise is spread out.
    hacker_energy = torch.abs(hacker_view.sum(dim=0))
    print("    -> Result: Static. The signal is scattered across the spectrum.")

    # --- STEP 3: THE OWNER (Right Password) ---
    print("\n[3] Owner Access (Correct Key)...")
    owner_view = crack_the_safe(brain, 0, REAL_PASSWORD)
    owner_energy = torch.abs(owner_view.sum(dim=0))
    print("    -> Result: Signal Resolved. Constructive interference achieved.")

    # --- STEP 4: VISUALIZE ---
    print("\n[4] Generating Evidence (saved to optical_safe.png)...")
    
    plt.figure(figsize=(10, 5))
    
    # Plot Hacker View
    plt.subplot(1, 2, 1)
    plt.plot(hacker_energy.cpu().numpy(), color='red')
    plt.title(f"Hacker View (Key={wrong_password})")
    plt.ylim(0, 300) # Fixed scale to show contrast
    plt.grid(True, alpha=0.3)
    plt.xlabel("Spatial Mode")
    plt.ylabel("Intensity")
    
    # Plot Owner View
    plt.subplot(1, 2, 2)
    plt.plot(owner_energy.cpu().numpy(), color='lime')
    plt.title(f"Owner View (Key={REAL_PASSWORD})")
    plt.ylim(0, 300)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Spatial Mode")
    
    plt.savefig('optical_safe.png')
    print("    -> Done.")

if __name__ == "__main__":
    run_crypto_demo()