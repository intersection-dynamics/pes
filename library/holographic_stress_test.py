"""
╔════════════════════════════════════════════════════════════════════════════════╗
║             HOLOGRAPHIC MEMORY: STRESS TEST & AUTO-FOCUS (FINAL)               ║
║                                                                                ║
║           "To validate the memory, we must define 'Truth' correctly."          ║
╚════════════════════════════════════════════════════════════════════════════════╝

FIXES APPLIED:
1. TIMING: Synchronized Encode/Recall loops (3 steps vs 3 steps).
2. TRUTH: Ground Truth is now calculated as the COHERENT SUM of the UNTWISTED state.
          This ensures we are comparing "Signal vs Signal", not "Signal vs Noise".
3. MEASUREMENT: Standardized on 'Coherent Sum' (sum -> abs) for all tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass

# ============================================================================
# 1. CORE PHYSICS ENGINE
# ============================================================================

@dataclass
class MemoryConfig:
    n_wavelengths: int = 64
    n_spatial_modes: int = 64
    cavity_roundtrip: int = 8
    cavity_loss: float = 0.2
    write_strength: float = 0.5
    retrieval_sharpness: float = 50.0

class ResonantCavity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('field', torch.zeros(config.n_wavelengths, config.n_spatial_modes, dtype=torch.cfloat))
        self.delay_buffer = [torch.zeros_like(self.field) for _ in range(config.cavity_roundtrip)]
        self.ptr = 0
        
    def silence(self):
        self.field.zero_()
        for i in range(len(self.delay_buffer)): self.delay_buffer[i].zero_()
        self.ptr = 0
        
    def step(self, input_signal, dispersion_scale=1.0):
        device = self.field.device
        feedback = self.delay_buffer[self.ptr].to(device) * (1.0 - self.config.cavity_loss)
        
        if input_signal is not None:
            if input_signal.dim() == 1: input_signal = input_signal.unsqueeze(0).expand(self.config.n_wavelengths, -1)
            feedback = feedback + (input_signal * 0.8)
            
        twist = torch.linspace(0, 2*np.pi, self.config.n_wavelengths, device=device) * dispersion_scale
        feedback = feedback * torch.exp(1j * twist).unsqueeze(1)
        
        amp = torch.abs(feedback)
        mask = amp > 5.0
        if mask.any(): feedback[mask] = feedback[mask] / amp[mask] * 5.0
            
        self.field = feedback
        self.delay_buffer[self.ptr] = feedback.detach().clone()
        self.ptr = (self.ptr + 1) % self.config.cavity_roundtrip
        return self.field

class HolographicField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('hologram', torch.zeros(config.n_wavelengths, config.n_spatial_modes, dtype=torch.cfloat))
        self.keys = []
        self.ground_truth = []

    def write(self, state, dispersion_val):
        """
        Write the cavity state to the hologram.
        CRITICAL: We also calculate what this memory *should* look like if perfectly recalled.
        """
        idx = len(self.keys)
        freq = (idx + 1) * 2 * np.pi / self.config.n_wavelengths
        key = torch.exp(1j * torch.arange(self.config.n_wavelengths, device=self.hologram.device) * freq)
        self.keys.append(key)
        
        # 1. Store: State * Key*
        self.hologram += (state * key.conj().unsqueeze(1)) * self.config.write_strength
        
        # 2. Calculate Ground Truth (Ideally Recalled)
        # The 'state' is twisted by 'dispersion_val'. 
        # A perfect recall would apply 'dispersion_val' inverse correction.
        # So GT = CoherentSum(State * Correction)
        twist = torch.linspace(0, 2*np.pi, self.config.n_wavelengths, device=state.device) * dispersion_val
        correction = torch.exp(-1j * twist).unsqueeze(1)
        ideal_memory = state * correction
        
        # Store the spatial magnitude profile of the coherent sum
        self.ground_truth.append(torch.abs(ideal_memory.sum(dim=0)).detach())

    def associative_read(self, cue):
        # 1. Hologram * Cue*
        noisy_mix = (self.hologram * cue.conj()).sum(dim=1)
        
        # 2. Match Key (Key* . Key = 1)
        correlations = torch.stack([torch.abs(torch.dot(noisy_mix, k)) for k in self.keys])
        attention = F.softmax(correlations * self.config.retrieval_sharpness, dim=0)
        
        # 3. Reconstruct
        clean_key = sum(k * w for k, w in zip(self.keys, attention))
        return self.hologram * clean_key.unsqueeze(1)

# ============================================================================
# 2. THE CONTROLLER
# ============================================================================

class MemoryController(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MemoryConfig()
        self.hologram = HolographicField(self.config)
        self.cavity = ResonantCavity(self.config)
        
    def encode(self, pattern, dispersion_val):
        self.cavity.silence()
        # 3 steps to build resonance
        self.cavity.step(pattern, dispersion_scale=dispersion_val)
        self.cavity.step(pattern, dispersion_scale=dispersion_val)
        state = self.cavity.step(pattern, dispersion_scale=dispersion_val)
        
        self.hologram.write(state, dispersion_val)
        self.cavity.silence()
        
    def recall_blind(self, cue, encoding_dispersion, compensation_val):
        self.cavity.silence()
        for _ in range(3): self.cavity.step(cue, dispersion_scale=encoding_dispersion)
        
        raw_read = self.hologram.associative_read(self.cavity.field)
        
        # Apply Fixed Compensation
        twist = torch.linspace(0, 2*np.pi, self.config.n_wavelengths, device=cue.device)
        correction = torch.exp(-1j * twist * compensation_val).unsqueeze(1)
        
        # Return compensated field
        return raw_read * correction

    def recall_autofocus(self, cue, encoding_dispersion):
        self.cavity.silence()
        for _ in range(3): self.cavity.step(cue, dispersion_scale=encoding_dispersion)
        raw_read = self.hologram.associative_read(self.cavity.field)
        
        device = cue.device
        base_twist = torch.linspace(0, 2*np.pi, self.config.n_wavelengths, device=device)
        
        best_score = -1.0
        best_image = None
        best_scale = 0.0
        
        # Sweep for the sharpest image
        # Standard range 0.5 to 1.5 covers expected drift
        sweep_range = np.linspace(0.5, 1.5, 20) 
        
        for scale in sweep_range:
            correction = torch.exp(-1j * base_twist * scale).unsqueeze(1)
            candidate = raw_read * correction
            
            # Metric: Coherent Sum Magnitude
            # If untwisted, the complex vectors align -> High Magnitude
            # If twisted, they cancel -> Low Magnitude
            score = torch.abs(candidate.sum(dim=0)).sum().item()
            
            if score > best_score:
                best_score = score
                best_image = candidate
                best_scale = scale
                
        return best_image, best_scale

# ============================================================================
# 3. TEST HARNESS
# ============================================================================

def measure_acc(recon, target_idx, ground_truths):
    # Standard Coherent Measurement (Sum -> Abs)
    # Assumes 'recon' has already been compensated
    profile = torch.abs(recon.sum(dim=0))
    
    sims = [F.cosine_similarity(profile.flatten(), gt.flatten(), dim=0, eps=1e-8).item() for gt in ground_truths]
    return np.argmax(sims) == target_idx

def run_stress_test():
    print(f"\n{'='*30}\n STRESS TEST 1: DISPERSION DRIFT\n{'='*30}")
    print(f"{'DRIFT':<10} | {'METHOD':<15} | {'ACCURACY':<10} | {'ESTIMATED'}")
    print("-" * 55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run a few drift scenarios
    for drift in [1.0, 1.1, 1.25, 0.8]:
        brain = MemoryController().to(device)
        n_patterns = 10
        
        # 1. Encode
        for i in range(n_patterns):
            x = torch.linspace(-3, 3, 64, device=device)
            np.random.seed(i) 
            center = np.random.uniform(-2.5, 2.5)
            bump = torch.exp(-(x - center)**2 / 0.1).to(torch.cfloat)
            brain.encode(bump, drift)
            
        # 2. Test Blind (Expect Failure for non-1.0)
        score_blind = 0
        for i in range(n_patterns):
            np.random.seed(i)
            center = np.random.uniform(-2.5, 2.5)
            cue = torch.exp(-(torch.linspace(-3, 3, 64, device=device) - center)**2 / 0.1).to(torch.cfloat)
            
            recon = brain.recall_blind(cue, drift, compensation_val=1.0)
            if measure_acc(recon, i, brain.hologram.ground_truth): score_blind += 1
            
        # 3. Test Auto-Focus (Expect Success)
        score_auto = 0
        est_drift_sum = 0
        for i in range(n_patterns):
            np.random.seed(i)
            center = np.random.uniform(-2.5, 2.5)
            cue = torch.exp(-(torch.linspace(-3, 3, 64, device=device) - center)**2 / 0.1).to(torch.cfloat)
            
            recon, est_drift = brain.recall_autofocus(cue, drift)
            if measure_acc(recon, i, brain.hologram.ground_truth): score_auto += 1
            est_drift_sum += est_drift
            
        avg_est = est_drift_sum / n_patterns
        
        print(f"{drift:<10.2f} | {'Blind (1.0)':<15} | {score_blind/n_patterns:.0%}       | -")
        print(f"{drift:<10.2f} | {'Auto-Focus':<15} | {score_auto/n_patterns:.0%}       | {avg_est:.2f}")
        print("-" * 55)


    print(f"\n{'='*30}\n STRESS TEST 2: THE CAPACITY WALL\n{'='*30}")
    print("Testing limits of Fourier Keying (N_Wavelengths = 64)")
    print(f"{'LOAD (K)':<10} | {'ACCURACY':<10} | {'STATUS'}")
    print("-" * 45)
    
    brain = MemoryController().to(device)
    
    checkpoints = [10, 30, 60, 64, 65, 100]
    current_idx = 0
    
    for k in checkpoints:
        # Load up to K
        while current_idx < k:
            x = torch.linspace(-3, 3, 64, device=device)
            center = np.random.uniform(-2.8, 2.8)
            bump = torch.exp(-(x - center)**2 / 0.1).to(torch.cfloat)
            brain.encode(bump, dispersion_val=1.0)
            current_idx += 1
            
        score = 0
        for i in range(k):
            key = brain.hologram.keys[i]
            # Direct Holographic Read (using 1.0 compensation assumption)
            raw_recon = brain.hologram.hologram * key.unsqueeze(1)
            
            # Apply standard compensation (untwist)
            twist = torch.linspace(0, 2*np.pi, 64, device=device)
            correction = torch.exp(-1j * twist).unsqueeze(1)
            compensated_recon = raw_recon * correction
            
            if measure_acc(compensated_recon, i, brain.hologram.ground_truth): score += 1
            
        acc = score / k
        status = "OK" if acc > 0.9 else "COLLAPSE"
        if k >= 64 and acc > 0.5: status = "IMPOSSIBLE (Aliasing)"
        
        print(f"{k:<10} | {acc:.1%}      | {status}")

if __name__ == "__main__":
    run_stress_test()