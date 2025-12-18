"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                       THE INTERFERENCE GALLERY                                 ║
║                                                                                ║
║           "Visualizing the ghost in the machine."                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

WHAT THIS DOES:
1. Uses 'holographic_core' to create a memory bank.
2. Injects 3 orthogonal patterns (Concepts A, B, C).
3. Renders the complex memory field as an image:
   - BRIGHTNESS = Amplitude (Energy)
   - COLOR (HUE) = Phase (The Twist)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from holographic_core import HolographicMemory, MemoryConfig

def complex_to_rgb(complex_field):
    """
    Converts a complex tensor (Amplitude + Phase) into an RGB image.
    """
    # 1. Get Amplitude and Phase
    amp = torch.abs(complex_field).detach().cpu().numpy()
    phase = torch.angle(complex_field).detach().cpu().numpy()
    
    # 2. Normalize Amplitude for display
    amp = amp / (np.max(amp) + 1e-9)
    
    # 3. Convert Phase (-Pi to Pi) to Hue (0 to 1)
    hue = (phase + np.pi) / (2 * np.pi)
    
    # 4. Build Image pixel by pixel
    height, width = amp.shape
    rgb_image = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            # HSV to RGB: H = Phase, S = 1.0, V = Amplitude
            r, g, b = colorsys.hsv_to_rgb(hue[y, x], 1.0, amp[y, x])
            rgb_image[y, x] = [r, g, b]
            
    return rgb_image

def generate_concept(index, total, n_modes, device):
    """
    Generates a pattern that looks like a wave packet.
    FIX: Now accepts n_modes to match the brain's resolution.
    """
    x = torch.linspace(-3, 3, n_modes, device=device)
    
    # Create a unique center for this concept
    center = -2.0 + (index * 4.0 / total)
    
    # A "Concept" is a Gaussian wave packet
    envelope = torch.exp(-(x - center)**2 / 0.2)
    
    # We add a "Flavor" (Phase modulation) unique to this concept
    flavor = torch.cos(x * (index + 1) * 2) 
    
    return (envelope * torch.exp(1j * flavor)).to(torch.cfloat)

def run_gallery():
    print("="*60)
    print("          SCULPTING WITH LIGHT: THE GALLERY")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize the Engine
    # We use 128x128 for higher resolution art
    config = MemoryConfig(n_wavelengths=128, n_spatial_modes=128)
    brain = HolographicMemory(config).to(device)
    
    print("1. Injecting Thoughts...")
    
    # 2. Create and Store 3 "Concepts"
    # We pass config.n_spatial_modes so the pattern size matches the brain size
    
    # Concept 1: "The Calm" (Low Dispersion)
    c1 = generate_concept(0, 3, config.n_spatial_modes, device)
    brain.encode(c1, dispersion_val=0.5) 
    print("   -> Encoded 'The Calm' (Blue/Green phase)")
    
    # Concept 2: "The Chaos" (High Dispersion)
    c2 = generate_concept(1, 3, config.n_spatial_modes, device)
    brain.encode(c2, dispersion_val=1.5)
    print("   -> Encoded 'The Chaos' (Rapid rainbow phase)")
    
    # Concept 3: "The Void" (Negative Dispersion)
    c3 = generate_concept(2, 3, config.n_spatial_modes, device)
    brain.encode(c3, dispersion_val=-1.0)
    print("   -> Encoded 'The Void' (Inverted phase)")
    
    # 3. Harvest the Hologram
    # The 'hologram' tensor holds the sum of all these interferences.
    raw_memory = brain.hologram.hologram
    
    # 4. Render
    print("2. Developing the Image...")
    art = complex_to_rgb(raw_memory)
    
    # 5. Display
    plt.figure(figsize=(12, 12))
    plt.imshow(art, origin='lower', interpolation='bicubic')
    plt.title("Holographic Interference Field\n(X=Space, Y=Wavelength, Color=Phase)", fontsize=14)
    plt.xlabel("Spatial Mode (Position)")
    plt.ylabel("Wavelength (Frequency)")
    
    # Remove ticks for a cleaner "Art" look
    plt.xticks([])
    plt.yticks([])
    
    output_file = 'holographic_art.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\nSUCCESS: Masterpiece saved to {output_file}")
    print("Check the file. The colors represent the 'twist' of the light.")

if __name__ == "__main__":
    run_gallery()