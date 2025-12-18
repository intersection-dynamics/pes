"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                      HOLOGRAPHIC MEMORY CORE ENGINE                            ║
║                                                                                ║
║      "A physics-compliant controller for dispersive optical memory."           ║
╚════════════════════════════════════════════════════════════════════════════════╝

COMPONENTS:
1. ResonantCavity: Simulates optical fiber loops with phase dispersion.
2. HolographicField: Content-addressable storage using interference patterns.
3. HolographicMemory: Main controller implementing auto-focus and recall.

PHYSICS NOTES:
- Uses discrete wavelengths as orthogonal channels (Fourier Keying).
- Implements phase-conjugate measurement to correct for cavity dispersion.
- Capacity Limit: N_patterns <= N_wavelengths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class MemoryConfig:
    n_wavelengths: int = 64
    n_spatial_modes: int = 64
    cavity_roundtrip: int = 8
    cavity_loss: float = 0.2
    write_strength: float = 0.5
    retrieval_sharpness: float = 50.0

# ============================================================================
# 2. PHYSICAL COMPONENTS
# ============================================================================

class ResonantCavity(nn.Module):
    """
    Simulates a ring resonator or fiber loop.
    Applies real physical dispersion (phase twist) to signals.
    """
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            'field', 
            torch.zeros(config.n_wavelengths, config.n_spatial_modes, dtype=torch.cfloat)
        )
        self.delay_buffer = [
            torch.zeros_like(self.field) for _ in range(config.cavity_roundtrip)
        ]
        self.ptr = 0
        
    def silence(self):
        """Clear the working memory (cavity state)"""
        self.field.zero_()
        for i in range(len(self.delay_buffer)): 
            self.delay_buffer[i].zero_()
        self.ptr = 0
        
    def step(self, input_signal: torch.Tensor = None, dispersion_scale: float = 1.0) -> torch.Tensor:
        """
        One physics timestep.
        dispersion_scale: Simulates environmental drift (1.0 = nominal).
        """
        device = self.field.device
        
        # 1. Retrieve Feedback
        feedback = self.delay_buffer[self.ptr].to(device) * (1.0 - self.config.cavity_loss)
        
        # 2. Add Input (Coupling)
        if input_signal is not None:
            if input_signal.dim() == 1: 
                input_signal = input_signal.unsqueeze(0).expand(self.config.n_wavelengths, -1)
            feedback = feedback + (input_signal * 0.8)
            
        # 3. Apply Dispersion (The "Twist")
        # Real physics: The phase twist depends on wavelength index
        twist = torch.linspace(0, 2*np.pi, self.config.n_wavelengths, device=device) * dispersion_scale
        feedback = feedback * torch.exp(1j * twist).unsqueeze(1)
        
        # 4. Clamp (Saturation/Nonlinearity)
        amp = torch.abs(feedback)
        mask = amp > 5.0
        if mask.any(): 
            feedback[mask] = feedback[mask] / amp[mask] * 5.0
            
        # 5. Store & Update
        self.field = feedback
        self.delay_buffer[self.ptr] = feedback.detach().clone()
        self.ptr = (self.ptr + 1) % self.config.cavity_roundtrip
        
        return self.field


class HolographicField(nn.Module):
    """
    The Long-Term Storage medium.
    Records interference patterns between State vectors and Orthogonal Keys.
    """
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            'hologram', 
            torch.zeros(config.n_wavelengths, config.n_spatial_modes, dtype=torch.cfloat)
        )
        self.keys: List[torch.Tensor] = []
        self.ground_truth: List[torch.Tensor] = []

    def write(self, state: torch.Tensor, dispersion_val: float):
        """
        Write state to hologram.
        Also calculates the 'Ideal Ground Truth' (untwisted) for validation.
        """
        idx = len(self.keys)
        # Generate Orthogonal Key (Fourier Basis)
        freq = (idx + 1) * 2 * np.pi / self.config.n_wavelengths
        key = torch.exp(1j * torch.arange(self.config.n_wavelengths, device=self.hologram.device) * freq)
        self.keys.append(key)
        
        # 1. Store: State * Key_Conjugate
        self.hologram += (state * key.conj().unsqueeze(1)) * self.config.write_strength
        
        # 2. Calculate Ground Truth (What this memory looks like if perfectly recalled)
        # We untwist the state using the inverse dispersion to get the 'pure' signal.
        twist = torch.linspace(0, 2*np.pi, self.config.n_wavelengths, device=state.device) * dispersion_val
        correction = torch.exp(-1j * twist).unsqueeze(1)
        ideal_memory = state * correction
        
        # Store magnitude profile of the coherent sum (for accuracy checking)
        self.ground_truth.append(torch.abs(ideal_memory.sum(dim=0)).detach())

    def associative_read(self, cue: torch.Tensor) -> torch.Tensor:
        """
        Query the hologram.
        Returns the reconstruction associated with the best matching key.
        """
        # 1. Hologram * Cue_Conjugate -> Noisy Mix
        noisy_mix = (self.hologram * cue.conj()).sum(dim=1)
        
        # 2. Match Key (Winner-Take-All)
        # We check correlation against known keys
        correlations = torch.stack([torch.abs(torch.dot(noisy_mix, k)) for k in self.keys])
        attention = F.softmax(correlations * self.config.retrieval_sharpness, dim=0)
        
        # 3. Reconstruct Key
        clean_key = sum(k * w for k, w in zip(self.keys, attention))
        
        # 4. Unlock Memory
        return self.hologram * clean_key.unsqueeze(1)

# ============================================================================
# 3. CONTROLLER
# ============================================================================

class HolographicMemory(nn.Module):
    """
    Main Interface.
    Manages the Encode/Recall cycles and Auto-Focus logic.
    """
    def __init__(self, config: MemoryConfig = None):
        super().__init__()
        self.config = config or MemoryConfig()
        self.hologram = HolographicField(self.config)
        self.cavity = ResonantCavity(self.config)
        
    def encode(self, pattern: torch.Tensor, dispersion_val: float = 1.0):
        """
        Encode a pattern into memory.
        dispersion_val: The environmental phase twist during this write.
        """
        self.cavity.silence()
        # 3 steps to build resonance (Must match recall timing!)
        self.cavity.step(pattern, dispersion_scale=dispersion_val)
        self.cavity.step(pattern, dispersion_scale=dispersion_val)
        state = self.cavity.step(pattern, dispersion_scale=dispersion_val)
        
        self.hologram.write(state, dispersion_val)
        self.cavity.silence()
        
    def recall(self, cue: torch.Tensor, encoding_dispersion: float = 1.0, auto_focus: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Recall a memory.
        If auto_focus is True, sweeps compensation to find the sharpest signal.
        Returns: (Compensated Image, Estimated Dispersion)
        """
        self.cavity.silence()
        # 3 steps resonance
        for _ in range(3): 
            self.cavity.step(cue, dispersion_scale=encoding_dispersion)
        
        raw_read = self.hologram.associative_read(self.cavity.field)
        
        if not auto_focus:
            # Blind recall (assumes nominal dispersion 1.0)
            return self._apply_correction(raw_read, 1.0), 1.0
        
        return self._auto_focus_sweep(raw_read)

    def _apply_correction(self, raw_read: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply phase compensation lens"""
        device = raw_read.device
        base_twist = torch.linspace(0, 2*np.pi, self.config.n_wavelengths, device=device)
        correction = torch.exp(-1j * base_twist * scale).unsqueeze(1)
        return raw_read * correction

    def _auto_focus_sweep(self, raw_read: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Blindly optimize dispersion compensation to maximize signal coherence.
        Metric: Max Spectral Energy (Phase Invariant).
        """
        best_score = -1.0
        best_image = None
        best_scale = 0.0
        
        # Coarse sweep covers typical drift (0.5x to 1.5x)
        sweep_range = np.linspace(0.5, 1.5, 20) 
        
        for scale in sweep_range:
            candidate = self._apply_correction(raw_read, scale)
            
            # Metric: Coherent Sum Magnitude
            # If untwisted, the complex vectors align -> High Magnitude
            # If twisted, they cancel -> Low Magnitude
            score = torch.abs(candidate.sum(dim=0)).sum().item()
            
            if score > best_score:
                best_score = score
                best_image = candidate
                best_scale = scale
                
        return best_image, best_scale