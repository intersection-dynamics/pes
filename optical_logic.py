"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                          OPTICAL ALU (SIMULTANEOUS INJECTION)                  ║
║                                                                                ║
║        "Logic happens in the collision, not the queue."                        ║
╚════════════════════════════════════════════════════════════════════════════════╝

FIX:
Changed 'operate' to combine signals A and B *before* injecting them into the cavity.
This simulates a Beam Combiner (two fibers merging into one).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from holographic_core import HolographicMemory, MemoryConfig

class OpticalALU:
    def __init__(self):
        self.config = MemoryConfig()
        self.processor = HolographicMemory(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor.to(self.device)

    def generate_number(self, value_idx):
        """Creates a distinct 'Optical Number' (Orthogonal Pattern)"""
        x = torch.linspace(-3, 3, self.config.n_spatial_modes, device=self.device)
        center = -2.0 + (value_idx * 1.0)
        return torch.exp(-(x - center)**2 / 0.1).to(torch.cfloat)

    def operate(self, pattern_a, pattern_b, operation="ADD"):
        """Performs an optical operation via Simultaneous Injection"""
        self.processor.cavity.silence()
        
        # LOGIC: Prepare the waveform at the input coupler
        if operation == "ADD":
            # Constructive: A + B
            input_wave = pattern_a + pattern_b
            
        elif operation == "SUB":
            # Destructive: A + (B * -1)
            # In optics, -1 is a phase shift of PI
            phase_shift = torch.tensor(np.pi, device=self.device)
            inverted_b = pattern_b * torch.exp(1j * phase_shift)
            input_wave = pattern_a + inverted_b
            
        # PHYSICS: Inject the combined wave into the cavity
        # Because they enter together, interference is instantaneous.
        self.processor.cavity.step(input_wave)
            
        return self.processor.cavity.field

    def measure_energy(self, field):
        return torch.sum(torch.abs(field)**2).item()

def run_logic_demo():
    print("="*60)
    print("          OPTICAL LOGIC GATE DEMO (FIXED)")
    print("="*60)
    
    alu = OpticalALU()
    
    val_1 = alu.generate_number(1) 
    val_2 = alu.generate_number(2) 
    
    print("\n[TEST 1] OPTICAL ADDITION (1 + 2)")
    print("-" * 40)
    result = alu.operate(val_1, val_2, "ADD")
    energy = alu.measure_energy(result)
    print(f"Energy Output: {energy:.2f} (Bright)")

    print("\n[TEST 2] OPTICAL SUBTRACTION (1 - 2)")
    print("-" * 40)
    result = alu.operate(val_1, val_2, "SUB")
    energy = alu.measure_energy(result)
    print(f"Energy Output: {energy:.2f} (Bright)")

    print("\n[TEST 3] THE XOR GATE (Equality Check)")
    print("-" * 40)
    print("Case A: 1 XOR 2 (Inputs Different)")
    res_diff = alu.operate(val_1, val_2, "SUB")
    en_diff = alu.measure_energy(res_diff)
    print(f"  -> Energy: {en_diff:.2f} (Logic 1)")
    
    print("Case B: 1 XOR 1 (Inputs Identical)")
    res_same = alu.operate(val_1, val_1, "SUB")
    en_same = alu.measure_energy(res_same)
    print(f"  -> Energy: {en_same:.10f} (Logic 0)")
    
    if en_same < 1e-6:
        print("\nSUCCESS: Absolute Zero. Perfect destructive interference.")
    else:
        print("\nFAILURE: Physics still broken.")

if __name__ == "__main__":
    run_logic_demo()