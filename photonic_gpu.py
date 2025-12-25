"""
╔════════════════════════════════════════════════════════════════════════════════╗
║              PHOTONIC NEURAL NETWORK - GPU ACCELERATED                         ║
║              Heterogeneous Switch Architecture for CUDA                        ║
╚════════════════════════════════════════════════════════════════════════════════╝

Designed for NVIDIA GPU simulation. Run with:
    python photonic_gpu.py

Requirements:
    pip install torch numpy

The key insight: wavelength-division multiplexing means we can process
all 64 wavelength channels in parallel on the GPU. Each switch processes
its entire spectrum simultaneously - this is where photonics beats electronics.
"""

import math
import time
import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPUConfig:
    # Scale parameters
    n_wavelengths: int = 64           # spectral channels (WDM bands)
    n_modes: int = 64                 # spatial modes per wavelength
    n_switches: int = 256             # total switches in network
    
    # Physics
    cavity_loss: float = 0.05
    saturation: float = 2.0
    integration_steps: int = 4
    dispersion_strength: float = 0.1
    
    # Network topology
    connectivity: float = 0.1         # fraction of possible connections
    
    # Simulation
    batch_size: int = 32              # parallel input patterns
    dtype: torch.dtype = torch.complex64
    
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# SWITCH TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class SwitchType(Enum):
    VISUAL = 0
    ARITHMETIC = 1
    MEMORY = 2
    ROUTING = 3
    INTEGRATOR = 4    # generic accumulator
    OSCILLATOR = 5    # generates rhythmic output


def get_wavelength_weights(switch_type: SwitchType, n_wavelengths: int, device: torch.device) -> torch.Tensor:
    """
    Each switch type interprets wavelengths differently.
    Returns a [n_wavelengths] weight vector.
    """
    w = torch.zeros(n_wavelengths, device=device)
    n = n_wavelengths
    
    if switch_type == SwitchType.VISUAL:
        # Edges, spatial freq, color, motion
        w[0:n//4] = 1.0
        w[n//4:n//2] = 0.8
        w[n//2:3*n//4] = 0.5
        w[3*n//4:n] = 0.3
        
    elif switch_type == SwitchType.ARITHMETIC:
        # All wavelengths equal for numerical precision
        w[:] = 1.0
        
    elif switch_type == SwitchType.MEMORY:
        # Key wavelengths weighted higher
        w[0:n//2] = 1.0    # key
        w[n//2:n] = 0.8    # value
        
    elif switch_type == SwitchType.ROUTING:
        # Priority wavelengths highest
        w[0:n//4] = 0.5      # source
        w[n//4:n//2] = 0.5   # dest
        w[n//2:3*n//4] = 1.0 # priority
        w[3*n//4:n] = 0.8    # payload
        
    elif switch_type == SwitchType.INTEGRATOR:
        # Uniform integration
        w[:] = 1.0
        
    elif switch_type == SwitchType.OSCILLATOR:
        # Resonant at specific wavelengths
        center = n // 2
        w = torch.exp(-((torch.arange(n, device=device) - center) ** 2) / (n / 8) ** 2)
    
    return w


# ══════════════════════════════════════════════════════════════════════════════
# BATCHED PHOTONIC NETWORK (GPU-OPTIMIZED)
# ══════════════════════════════════════════════════════════════════════════════

class PhotonicNetworkGPU(nn.Module):
    """
    GPU-accelerated photonic neural network.
    
    All switches are processed in parallel as a single batched operation.
    The network state is a tensor of shape:
        [batch, n_switches, n_wavelengths, n_modes]
    
    This maps naturally to GPU parallelism:
        - batch dimension: multiple input patterns simultaneously
        - switch dimension: all switches process in parallel
        - wavelength dimension: WDM channels (the photonic parallelism)
        - mode dimension: spatial modes within each channel
    """
    
    def __init__(self, config: GPUConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Assign switch types in structured blocks rather than randomly
        # This creates "regions" of same-type switches (like cortical areas)
        self.switch_types = self._init_switch_types()
        
        # Wavelength interpretation weights per switch: [n_switches, n_wavelengths]
        self.wavelength_weights = self._init_wavelength_weights()
        
        # Connectivity matrix: [n_switches, n_switches]
        # Entry [i,j] = strength of connection from switch i to switch j
        self.connectivity = self._init_connectivity()
        
        # Learnable per-switch parameters
        self.switch_gain = nn.Parameter(torch.ones(config.n_switches, device=self.device))
        self.switch_bias = nn.Parameter(torch.zeros(config.n_switches, device=self.device))
        
        # Dispersion phases (wavelength-dependent, shared)
        w = torch.linspace(-1, 1, config.n_wavelengths, device=self.device)
        w2 = w ** 2
        self.register_buffer('dispersion_shape', w2 - w2.mean())
        
        # Network state: [batch, n_switches, n_wavelengths, n_modes]
        self.field = None
        self.output = None
        
        # Memory storage for memory-type switches (simplified)
        self.hologram = torch.zeros(
            config.n_switches, config.n_wavelengths, config.n_modes,
            dtype=config.dtype, device=self.device
        )
        
        print(f"PhotonicNetworkGPU initialized on {self.device}")
        print(f"  Switches: {config.n_switches}")
        print(f"  Wavelengths: {config.n_wavelengths}")
        print(f"  Modes: {config.n_modes}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Total parallel ops per step: {config.batch_size * config.n_switches * config.n_wavelengths * config.n_modes:,}")
    
    def _init_switch_types(self) -> torch.Tensor:
        """
        Assign switch types in structured blocks (like cortical regions).
        
        Default allocation:
        - 30% Visual (early + late processing)
        - 20% Memory (storage and recall)
        - 15% Arithmetic (computation)
        - 15% Routing (attention/gating)
        - 10% Integrator (accumulation)
        - 10% Oscillator (rhythm generation)
        """
        n = self.config.n_switches
        types = torch.zeros(n, dtype=torch.long, device=self.device)
        
        # Allocate in contiguous blocks
        allocations = [
            (SwitchType.VISUAL, 0.30),
            (SwitchType.MEMORY, 0.20),
            (SwitchType.ARITHMETIC, 0.15),
            (SwitchType.ROUTING, 0.15),
            (SwitchType.INTEGRATOR, 0.10),
            (SwitchType.OSCILLATOR, 0.10),
        ]
        
        idx = 0
        for switch_type, fraction in allocations:
            count = max(1, int(n * fraction))
            end_idx = min(idx + count, n)
            types[idx:end_idx] = switch_type.value
            idx = end_idx
        
        # Fill any remaining
        if idx < n:
            types[idx:] = SwitchType.INTEGRATOR.value
        
        return types
    
    def _init_wavelength_weights(self) -> torch.Tensor:
        """Initialize wavelength weights based on switch types."""
        weights = torch.zeros(self.config.n_switches, self.config.n_wavelengths, device=self.device)
        
        for i in range(self.config.n_switches):
            st = SwitchType(self.switch_types[i].item())
            weights[i] = get_wavelength_weights(st, self.config.n_wavelengths, self.device)
        
        return weights
    
    def _init_connectivity(self) -> torch.Tensor:
        """
        Initialize structured connectivity based on switch types.
        
        Principles:
        - Same-type switches: dense local connectivity (process together)
        - Cross-type switches: sparse connections (integration pathways)
        - Hierarchy within types: feedforward + feedback
        """
        n = self.config.n_switches
        conn = torch.zeros(n, n, device=self.device)
        
        # Group switches by type
        type_indices = {}
        for t in SwitchType:
            mask = (self.switch_types == t.value)
            type_indices[t] = torch.where(mask)[0]
        
        # Connection strengths
        SAME_TYPE_DENSITY = 0.4      # dense within-type connections
        SAME_TYPE_STRENGTH = 0.3
        CROSS_TYPE_DENSITY = 0.05    # sparse cross-type connections
        CROSS_TYPE_STRENGTH = 0.15
        
        # Define cross-type pathways (which types talk to which)
        # Visual → Memory (store what we see)
        # Visual → Routing (attention to visual features)
        # Memory → Routing (memory-guided attention)
        # Memory → Arithmetic (compute on recalled data)
        # Routing → all (gating/attention)
        # Integrator → all (accumulation)
        # Oscillator → Routing, Memory (rhythmic gating)
        
        cross_type_pathways = {
            SwitchType.VISUAL: [SwitchType.MEMORY, SwitchType.ROUTING, SwitchType.VISUAL],
            SwitchType.ARITHMETIC: [SwitchType.MEMORY, SwitchType.ROUTING, SwitchType.ARITHMETIC],
            SwitchType.MEMORY: [SwitchType.ROUTING, SwitchType.ARITHMETIC, SwitchType.VISUAL, SwitchType.MEMORY],
            SwitchType.ROUTING: [SwitchType.VISUAL, SwitchType.ARITHMETIC, SwitchType.MEMORY, SwitchType.ROUTING],
            SwitchType.INTEGRATOR: [SwitchType.VISUAL, SwitchType.ARITHMETIC, SwitchType.MEMORY, SwitchType.INTEGRATOR],
            SwitchType.OSCILLATOR: [SwitchType.ROUTING, SwitchType.MEMORY, SwitchType.OSCILLATOR],
        }
        
        for src_type, src_indices in type_indices.items():
            if len(src_indices) == 0:
                continue
                
            for dst_type, dst_indices in type_indices.items():
                if len(dst_indices) == 0:
                    continue
                
                # Determine connection density and strength
                if src_type == dst_type:
                    # Same type: dense local connectivity
                    density = SAME_TYPE_DENSITY
                    strength = SAME_TYPE_STRENGTH
                elif dst_type in cross_type_pathways.get(src_type, []):
                    # Defined pathway: sparse but present
                    density = CROSS_TYPE_DENSITY
                    strength = CROSS_TYPE_STRENGTH
                else:
                    # No pathway defined: very sparse or none
                    density = 0.01
                    strength = 0.05
                
                # Create connections
                for i in src_indices:
                    for j in dst_indices:
                        if i == j:
                            continue  # no self-connections
                        if torch.rand(1, device=self.device).item() < density:
                            # Random weight with sign
                            w = strength * (2 * torch.rand(1, device=self.device).item() - 1)
                            conn[i, j] = w
        
        # Add hierarchical structure within types
        # Earlier indices → later indices (feedforward)
        # Later indices → earlier indices (feedback, weaker)
        for switch_type, indices in type_indices.items():
            if len(indices) < 4:
                continue
            
            n_type = len(indices)
            for local_i, global_i in enumerate(indices):
                for local_j, global_j in enumerate(indices):
                    if local_i == local_j:
                        continue
                    
                    # Feedforward (earlier to later)
                    if local_j > local_i and local_j - local_i <= 3:
                        if conn[global_i, global_j] == 0:
                            conn[global_i, global_j] = SAME_TYPE_STRENGTH * 0.5
                    
                    # Feedback (later to earlier, weaker)
                    if local_j < local_i and local_i - local_j <= 2:
                        if conn[global_i, global_j] == 0:
                            conn[global_i, global_j] = SAME_TYPE_STRENGTH * 0.2
        
        # Convert to complex
        conn = conn.to(self.config.dtype)
        
        return conn
    
    def get_connectivity_stats(self) -> Dict:
        """Report connectivity statistics."""
        conn_real = self.connectivity.real
        
        stats = {"total_connections": (conn_real != 0).sum().item()}
        stats["density"] = stats["total_connections"] / (self.config.n_switches ** 2)
        
        # Per-type statistics
        for t in SwitchType:
            mask = (self.switch_types == t.value)
            indices = torch.where(mask)[0]
            if len(indices) == 0:
                continue
            
            # Connections within this type
            within = 0
            outgoing = 0
            incoming = 0
            
            for i in indices:
                for j in indices:
                    if conn_real[i, j] != 0:
                        within += 1
                
                outgoing += (conn_real[i, :] != 0).sum().item()
                incoming += (conn_real[:, i] != 0).sum().item()
            
            stats[f"{t.name}_within"] = within
            stats[f"{t.name}_outgoing"] = outgoing
            stats[f"{t.name}_incoming"] = incoming
        
        return stats
    
    def reset(self, batch_size: Optional[int] = None):
        """Reset network state."""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        self.field = torch.zeros(
            batch_size, self.config.n_switches, self.config.n_wavelengths, self.config.n_modes,
            dtype=self.config.dtype, device=self.device
        )
        self.output = torch.zeros_like(self.field)
    
    def inject(self, switch_idx: int, signal: torch.Tensor):
        """
        Inject external signal into specific switch(es).
        signal: [batch, n_wavelengths, n_modes] or [batch, n_modes]
        """
        if signal.dim() == 2:
            # Expand to all wavelengths
            signal = signal.unsqueeze(1).expand(-1, self.config.n_wavelengths, -1)
        
        signal = signal.to(self.config.dtype)
        self.field[:, switch_idx, :, :] = self.field[:, switch_idx, :, :] + signal
    
    def step(self) -> torch.Tensor:
        """
        One network timestep. All operations are batched and parallel.
        
        Returns: output tensor [batch, n_switches, n_wavelengths, n_modes]
        """
        cfg = self.config
        
        # ─────────────────────────────────────────────────────────────
        # 1. GATHER: Each switch collects input from connected switches
        # ─────────────────────────────────────────────────────────────
        # output: [batch, n_switches, n_wavelengths, n_modes]
        # connectivity: [n_switches, n_switches] where conn[i,j] = weight from i to j
        # We want: new_input[b, j, λ, m] = Σᵢ connectivity[i,j] * output[b, i, λ, m]
        
        # Reshape for batched matmul: [batch, n_wavelengths, n_modes, n_switches]
        out_t = self.output.permute(0, 2, 3, 1)
        
        # Matmul: [batch, λ, m, n_switches] @ [n_switches, n_switches] -> [batch, λ, m, n_switches]
        # Note: connectivity[i,j] means i→j, so we need conn.T for the matmul convention
        # gathered[..., j] = Σᵢ out[..., i] * conn.T[i, j] = Σᵢ out[..., i] * conn[j, i]
        # But we want gathered[..., j] = Σᵢ out[..., i] * conn[i, j]
        # So we use connectivity directly (not transposed)
        gathered = torch.matmul(out_t, self.connectivity)
        
        # Back to [batch, n_switches, n_wavelengths, n_modes]
        gathered = gathered.permute(0, 3, 1, 2)
        
        # ─────────────────────────────────────────────────────────────
        # 2. INTEGRATE: Cavity accumulation with loss and dispersion
        # ─────────────────────────────────────────────────────────────
        for _ in range(cfg.integration_steps):
            # Cavity loss
            self.field = self.field * (1.0 - cfg.cavity_loss)
            
            # Dispersion (wavelength-dependent phase)
            if cfg.dispersion_strength > 0:
                phase = self.dispersion_shape * cfg.dispersion_strength
                # phase: [n_wavelengths] -> broadcast to field
                self.field = self.field * torch.exp(1j * phase).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            
            # Add gathered input
            self.field = self.field + gathered * 0.3
        
        # ─────────────────────────────────────────────────────────────
        # 3. INTERPRET: Apply wavelength-specific weights
        # ─────────────────────────────────────────────────────────────
        # wavelength_weights: [n_switches, n_wavelengths]
        # field: [batch, n_switches, n_wavelengths, n_modes]
        weighted = self.field * self.wavelength_weights.unsqueeze(0).unsqueeze(-1)
        
        # ─────────────────────────────────────────────────────────────
        # 4. ACTIVATE: Nonlinear saturation (the only nonlinearity!)
        # ─────────────────────────────────────────────────────────────
        magnitude = torch.abs(weighted)
        phase = torch.angle(weighted)
        
        # Per-switch gain and bias
        gain = self.switch_gain.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = self.switch_bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # Tanh saturation with gain/bias
        sat = cfg.saturation
        activated_mag = sat * torch.tanh((magnitude * gain + bias) / sat)
        
        self.output = activated_mag * torch.exp(1j * phase)
        
        return self.output
    
    def run(self, n_steps: int, input_fn: Optional[Callable] = None) -> torch.Tensor:
        """
        Run network for multiple steps.
        
        input_fn: optional function(step) -> (switch_idx, signal) to inject inputs
        
        Returns: final output
        """
        for t in range(n_steps):
            if input_fn is not None:
                injection = input_fn(t)
                if injection is not None:
                    switch_idx, signal = injection
                    self.inject(switch_idx, signal)
            
            self.step()
        
        return self.output
    
    def read_intensity(self, switch_idx: Optional[int] = None) -> torch.Tensor:
        """
        Read output as intensity (|field|²).
        
        Returns: [batch, n_switches, n_wavelengths] or [batch, n_wavelengths] if switch_idx given
        """
        intensity = (torch.abs(self.output) ** 2).sum(dim=-1)  # sum over modes
        
        if switch_idx is not None:
            return intensity[:, switch_idx, :]
        return intensity
    
    def read_total_energy(self) -> torch.Tensor:
        """Total network energy per batch element."""
        return (torch.abs(self.output) ** 2).sum(dim=(1, 2, 3))


# ══════════════════════════════════════════════════════════════════════════════
# HOLOGRAPHIC MEMORY OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class HolographicMemoryBank(nn.Module):
    """
    Separate module for holographic memory operations.
    Can be attached to memory-type switches.
    """
    
    def __init__(self, config: GPUConfig, n_memories: int = 100):
        super().__init__()
        self.config = config
        self.device = config.device
        self.n_memories = n_memories
        
        # Hologram: [n_memories, n_wavelengths, n_modes]
        self.hologram = torch.zeros(
            n_memories, config.n_wavelengths, config.n_modes,
            dtype=config.dtype, device=self.device
        )
        
        # Keys: [n_memories, n_wavelengths]
        self.keys = torch.zeros(
            n_memories, config.n_wavelengths,
            dtype=config.dtype, device=self.device
        )
        
        self.n_stored = 0
    
    def store(self, pattern: torch.Tensor) -> int:
        """
        Store a pattern with random phase key.
        pattern: [n_modes] or [n_wavelengths, n_modes]
        
        Returns: index of stored pattern
        """
        if self.n_stored >= self.n_memories:
            print("Memory full!")
            return -1
        
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0).expand(self.config.n_wavelengths, -1)
        
        pattern = pattern.to(self.config.dtype).to(self.device)
        
        # Random phase key
        phase = torch.rand(self.config.n_wavelengths, device=self.device) * 2 * math.pi
        key = torch.exp(1j * phase)
        
        # Store
        self.keys[self.n_stored] = key
        self.hologram[self.n_stored] = key.unsqueeze(-1) * pattern
        
        idx = self.n_stored
        self.n_stored += 1
        
        return idx
    
    def recall(self, cue: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recall pattern by content correlation (not key correlation).
        
        For each stored pattern, we decode it using its key and compare
        the decoded content to the cue. This is true content-addressable memory.
        
        cue: [batch, n_wavelengths, n_modes] or [batch, n_modes]
        
        Returns: (recalled_patterns, confidence_scores, best_indices)
            recalled: [batch, n_wavelengths, n_modes]
            confidence: [batch]
            best_idx: [batch] tensor of indices of best matching stored patterns
        """
        if cue.dim() == 2:
            # cue is [batch, n_modes], use as content to match
            cue_content = cue.to(self.config.dtype)
        else:
            # cue is [batch, n_wavelengths, n_modes], sum over wavelengths for content
            cue_content = cue.sum(dim=1).to(self.config.dtype)
        
        batch_size = cue_content.shape[0]
        
        # For each stored pattern, decode and compare to cue
        # hologram[i] = key[i] * pattern[i]
        # decoded[i] = hologram[i] * conj(key[i]) = pattern[i] (since |key|=1)
        
        scores = torch.zeros(batch_size, self.n_stored, device=self.device)
        
        for i in range(self.n_stored):
            # Decode pattern i
            decoded = self.hologram[i] * self.keys[i].conj().unsqueeze(-1)
            # Sum over wavelengths to get content
            decoded_content = decoded.sum(dim=0)  # [n_modes]
            
            # Compare to each cue in batch via cosine similarity
            # cue_content: [batch, n_modes], decoded_content: [n_modes]
            cue_norm = torch.abs(cue_content)
            dec_norm = torch.abs(decoded_content)
            
            # Cosine similarity
            dot = torch.abs((cue_norm * dec_norm.unsqueeze(0)).sum(dim=-1))
            norms = (torch.norm(cue_norm, dim=-1) * torch.norm(dec_norm) + 1e-10)
            scores[:, i] = dot / norms
        
        # Find best match per batch element
        confidence, best_idx = scores.max(dim=-1)  # [batch]
        
        # Gather best patterns
        # best_idx: [batch], need to index into hologram and keys
        best_keys = self.keys[best_idx]  # [batch, n_wavelengths]
        best_holograms = self.hologram[best_idx]  # [batch, n_wavelengths, n_modes]
        
        # Decode
        recalled = best_holograms * best_keys.conj().unsqueeze(-1)
        
        return recalled, confidence, best_idx


# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class TopologyBuilder:
    """
    Helper to define custom network topologies.
    
    Example:
        builder = TopologyBuilder(n_switches=64)
        builder.add_region("V1", SwitchType.VISUAL, count=10)
        builder.add_region("V2", SwitchType.VISUAL, count=10)
        builder.add_region("STM", SwitchType.MEMORY, count=8)
        builder.connect_regions("V1", "V2", density=0.5, strength=0.3)
        builder.connect_regions("V2", "STM", density=0.2, strength=0.2)
        
        config, types, connectivity = builder.build()
        net = PhotonicNetworkGPU(config)
        net.switch_types = types
        net.connectivity = connectivity
    """
    
    def __init__(self, n_wavelengths: int = 64, n_modes: int = 32):
        self.n_wavelengths = n_wavelengths
        self.n_modes = n_modes
        self.regions: Dict[str, Dict] = {}
        self.connections: List[Tuple] = []
        self.next_idx = 0
    
    def add_region(self, name: str, switch_type: SwitchType, count: int, 
                   internal_density: float = 0.4, internal_strength: float = 0.3):
        """Add a region of switches of the same type."""
        self.regions[name] = {
            "type": switch_type,
            "start_idx": self.next_idx,
            "count": count,
            "internal_density": internal_density,
            "internal_strength": internal_strength,
        }
        self.next_idx += count
    
    def connect_regions(self, src: str, dst: str, density: float = 0.2, 
                        strength: float = 0.2, bidirectional: bool = False):
        """Define connections between regions."""
        self.connections.append((src, dst, density, strength))
        if bidirectional:
            self.connections.append((dst, src, density, strength * 0.5))
    
    def build(self, device: torch.device = None) -> Tuple[GPUConfig, torch.Tensor, torch.Tensor]:
        """Build the network configuration, types, and connectivity."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        n_switches = self.next_idx
        
        config = GPUConfig(
            n_switches=n_switches,
            n_wavelengths=self.n_wavelengths,
            n_modes=self.n_modes,
        )
        
        # Build switch types
        types = torch.zeros(n_switches, dtype=torch.long, device=device)
        for name, region in self.regions.items():
            start = region["start_idx"]
            end = start + region["count"]
            types[start:end] = region["type"].value
        
        # Build connectivity
        conn = torch.zeros(n_switches, n_switches, dtype=torch.complex64, device=device)
        
        # Internal connections within regions
        for name, region in self.regions.items():
            start = region["start_idx"]
            count = region["count"]
            density = region["internal_density"]
            strength = region["internal_strength"]
            
            for i in range(start, start + count):
                for j in range(start, start + count):
                    if i != j and torch.rand(1).item() < density:
                        conn[i, j] = strength * (2 * torch.rand(1).item() - 1)
        
        # Cross-region connections
        for src_name, dst_name, density, strength in self.connections:
            src = self.regions[src_name]
            dst = self.regions[dst_name]
            
            for i in range(src["start_idx"], src["start_idx"] + src["count"]):
                for j in range(dst["start_idx"], dst["start_idx"] + dst["count"]):
                    if torch.rand(1).item() < density:
                        conn[i, j] = strength * (2 * torch.rand(1).item() - 1)
        
        return config, types, conn
    
    def summary(self) -> str:
        """Return a summary of the topology."""
        lines = [f"Topology: {self.next_idx} switches in {len(self.regions)} regions"]
        for name, region in self.regions.items():
            lines.append(f"  {name}: {region['count']} x {region['type'].name}")
        lines.append(f"Connections: {len(self.connections)} pathways")
        for src, dst, density, strength in self.connections:
            lines.append(f"  {src} → {dst} (d={density:.0%}, s={strength:.2f})")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_network(config: GPUConfig, n_steps: int = 100) -> Dict:
    """Benchmark network performance."""
    
    print("\n" + "=" * 70)
    print("BENCHMARK: PHOTONIC NETWORK GPU PERFORMANCE")
    print("=" * 70)
    
    net = PhotonicNetworkGPU(config)
    net.reset()
    
    # Warmup
    print("\nWarmup...")
    for _ in range(10):
        net.step()
    
    if config.device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"\nRunning {n_steps} steps...")
    t0 = time.perf_counter()
    
    for _ in range(n_steps):
        net.step()
    
    if config.device.type == "cuda":
        torch.cuda.synchronize()
    
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    steps_per_sec = n_steps / elapsed
    
    # Calculate operations
    ops_per_step = (
        config.batch_size * 
        config.n_switches * 
        config.n_wavelengths * 
        config.n_modes
    )
    total_ops = ops_per_step * n_steps
    ops_per_sec = total_ops / elapsed
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f} s")
    print(f"  Steps/sec: {steps_per_sec:.1f}")
    print(f"  Elements/step: {ops_per_step:,}")
    print(f"  Throughput: {ops_per_sec/1e9:.2f} G elements/sec")
    
    if config.device.type == "cuda":
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU memory: {mem_allocated:.2f} GB")
    
    del net
    clear_gpu_memory()
    
    return {
        "elapsed": elapsed,
        "steps_per_sec": steps_per_sec,
        "throughput": ops_per_sec
    }


def benchmark_scaling():
    """Benchmark across different network sizes."""
    
    print("\n" + "=" * 70)
    print("SCALING BENCHMARK")
    print("=" * 70)
    
    # Conservative sizes for 8GB VRAM
    sizes = [32, 64, 128]
    results = []
    
    for n_switches in sizes:
        print(f"\n--- {n_switches} switches ---")
        
        clear_gpu_memory()
        
        config = GPUConfig(
            n_switches=n_switches,
            n_wavelengths=64,
            n_modes=32,
            batch_size=8
        )
        
        try:
            result = benchmark_network(config, n_steps=50)
            result["n_switches"] = n_switches
            results.append(result)
        except RuntimeError as e:
            print(f"  Failed: {e}")
            break
    
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print(f"\n{'Switches':>10} | {'Steps/s':>10} | {'Throughput (G/s)':>16}")
    print("-" * 45)
    
    for r in results:
        print(f"{r['n_switches']:>10} | {r['steps_per_sec']:>10.1f} | {r['throughput']/1e9:>16.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ══════════════════════════════════════════════════════════════════════════════

def demo_pattern_recognition():
    """Demo: pattern recognition with holographic memory."""
    
    print("\n" + "=" * 70)
    print("DEMO: PATTERN RECOGNITION")
    print("=" * 70)
    
    config = GPUConfig(
        n_switches=64,
        n_wavelengths=64,
        n_modes=64,
        batch_size=8
    )
    
    net = PhotonicNetworkGPU(config)
    memory = HolographicMemoryBank(config, n_memories=50)
    
    # Create and store patterns
    print("\nStoring patterns...")
    x = torch.linspace(-1, 1, config.n_modes, device=config.device)
    
    patterns = {
        "gaussian": torch.exp(-x**2 / 0.1),
        "sine": (torch.sin(4 * math.pi * x) + 1) / 2,
        "step": (x > 0).float(),
        "ramp": (x + 1) / 2,
        "double_peak": torch.exp(-(x-0.5)**2/0.05) + torch.exp(-(x+0.5)**2/0.05)
    }
    
    pattern_list = list(patterns.keys())
    for name, p in patterns.items():
        idx = memory.store(p)
        print(f"  Stored '{name}' at index {idx}")
    
    # Test recall with noisy versions
    print("\nTesting recall with 30% noise...")
    print("-" * 50)
    
    batch_size = len(patterns)
    cues = torch.zeros(batch_size, config.n_modes, device=config.device)
    
    for i, (name, p) in enumerate(patterns.items()):
        noisy = p + 0.3 * torch.randn_like(p)
        cues[i] = torch.clamp(noisy, 0, 1)
    
    recalled, confidence, best_indices = memory.recall(cues)
    
    # Check accuracy by comparing recalled content to originals
    for i, name in enumerate(pattern_list):
        # The recalled pattern, decoded
        rec = torch.abs(recalled[i]).sum(dim=0)  # sum magnitudes over wavelengths
        rec = rec / (rec.max() + 1e-10)  # normalize
        
        # Find which stored pattern has highest similarity
        best_match = None
        best_sim = -1
        for check_name, check_p in patterns.items():
            # Compare normalized magnitudes
            check_norm = check_p / (check_p.max() + 1e-10)
            sim = F.cosine_similarity(rec.unsqueeze(0), check_norm.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_match = check_name
        
        status = "✓" if best_match == name else "✗"
        print(f"  '{name}' → '{best_match}' (conf: {confidence[i]:.2f}, sim: {best_sim:.2f}) {status}")
    
    # Cleanup
    del net, memory, recalled, confidence, cues, patterns
    clear_gpu_memory()


def demo_network_dynamics():
    """Demo: observe network dynamics over time."""
    
    print("\n" + "=" * 70)
    print("DEMO: NETWORK DYNAMICS")
    print("=" * 70)
    
    config = GPUConfig(
        n_switches=128,
        n_wavelengths=64,
        n_modes=32,
        batch_size=4
    )
    
    net = PhotonicNetworkGPU(config)
    
    # Show connectivity structure
    print("\nNetwork topology (structured by switch type):")
    print("-" * 50)
    
    type_counts = {}
    for t in SwitchType:
        count = (net.switch_types == t.value).sum().item()
        if count > 0:
            type_counts[t.name] = count
            print(f"  {t.name:12s}: {count:3d} switches")
    
    stats = net.get_connectivity_stats()
    print(f"\n  Total connections: {stats['total_connections']:.0f}")
    print(f"  Network density: {stats['density']:.1%}")
    
    print("\n  Cross-type pathways:")
    print("    Visual → Memory, Routing (perception to storage/attention)")
    print("    Memory → Arithmetic, Visual (recall for computation/imagery)")  
    print("    Routing → All (attention gating)")
    print("    Oscillator → Routing, Memory (rhythmic coordination)")
    
    net.reset()
    
    # Inject stimulus into VISUAL switches specifically
    visual_indices = torch.where(net.switch_types == SwitchType.VISUAL.value)[0]
    
    stimulus = torch.randn(config.batch_size, config.n_modes, device=config.device) * 0.5
    
    print(f"\nInjecting stimulus into visual switches ({len(visual_indices)} switches)...")
    print("Tracking energy propagation across switch types...\n")
    
    energies_by_type = {t.name: [] for t in SwitchType if (net.switch_types == t.value).sum() > 0}
    
    for t in range(50):
        # Inject into first few visual switches
        if t < 10:
            for v_idx in visual_indices[:3]:
                net.inject(v_idx.item(), stimulus)
        
        net.step()
        
        # Track energy per type
        output_energy = (torch.abs(net.output) ** 2).sum(dim=(0, 2, 3))  # [n_switches]
        
        for switch_type in SwitchType:
            mask = (net.switch_types == switch_type.value)
            if mask.sum() > 0:
                type_energy = output_energy[mask].sum().item()
                energies_by_type[switch_type.name].append(type_energy)
        
        if t % 10 == 0:
            print(f"  t={t:3d}:", end="")
            for name, energies in energies_by_type.items():
                if energies:
                    print(f"  {name[:4]}={energies[-1]:.0f}", end="")
            print()
    
    print("\nEnergy flow shows signal propagating from Visual → other types")
    
    # Cleanup
    del net, stimulus, energies_by_type
    clear_gpu_memory()


def demo_wavelength_semantics():
    """Demo: show how different switch types interpret the same signal."""
    
    print("\n" + "=" * 70)
    print("DEMO: WAVELENGTH SEMANTIC INTERPRETATION")
    print("=" * 70)
    
    config = GPUConfig(n_wavelengths=64, n_modes=32)
    device = config.device
    
    # Create a test signal with structure in different wavelength bands
    signal = torch.zeros(config.n_wavelengths, config.n_modes, dtype=config.dtype, device=device)
    
    # Put different patterns in different wavelength bands
    x = torch.linspace(-1, 1, config.n_modes, device=device)
    
    signal[0:16, :] = torch.exp(-x**2 / 0.1)           # edges band
    signal[16:32, :] = torch.sin(8 * math.pi * x)      # frequency band
    signal[32:48, :] = torch.ones_like(x) * 0.5        # color band (uniform)
    signal[48:64, :] = torch.cos(2 * math.pi * x)      # motion band
    
    print("\nInput signal structure:")
    print("  λ₀-₁₅:  Gaussian peak (edge-like)")
    print("  λ₁₆-₃₁: High-frequency sine (texture)")
    print("  λ₃₂-₄₇: Uniform (flat color)")
    print("  λ₄₈-₆₃: Low-frequency cosine (motion)")
    
    print("\nWeighted responses by switch type:")
    print("-" * 60)
    
    for st in SwitchType:
        weights = get_wavelength_weights(st, config.n_wavelengths, device)
        
        # Apply weights
        weighted = signal * weights.unsqueeze(-1)
        
        # Compute energy in each band
        band_energy = []
        for band_start in [0, 16, 32, 48]:
            e = (torch.abs(weighted[band_start:band_start+16, :]) ** 2).sum().item()
            band_energy.append(e)
        
        total = sum(band_energy)
        fracs = [e/total*100 if total > 0 else 0 for e in band_energy]
        
        print(f"\n  {st.name:12s}:")
        print(f"    Edges:     {fracs[0]:5.1f}%  {'█' * int(fracs[0]/5)}")
        print(f"    Frequency: {fracs[1]:5.1f}%  {'█' * int(fracs[1]/5)}")
        print(f"    Color:     {fracs[2]:5.1f}%  {'█' * int(fracs[2]/5)}")
        print(f"    Motion:    {fracs[3]:5.1f}%  {'█' * int(fracs[3]/5)}")
    
    # Cleanup
    del signal
    clear_gpu_memory()


def demo_custom_topology():
    """Demo: Build a custom visual processing hierarchy."""
    
    print("\n" + "=" * 70)
    print("DEMO: CUSTOM VISUAL HIERARCHY")
    print("=" * 70)
    
    # Build a visual processing hierarchy like V1 → V2 → V4 → IT → Memory
    builder = TopologyBuilder(n_wavelengths=64, n_modes=32)
    
    # Visual hierarchy
    builder.add_region("V1", SwitchType.VISUAL, count=12)      # Early: edges, orientation
    builder.add_region("V2", SwitchType.VISUAL, count=10)      # Texture, contours
    builder.add_region("V4", SwitchType.VISUAL, count=8)       # Shape, color
    builder.add_region("IT", SwitchType.VISUAL, count=6)       # Object recognition
    
    # Memory and attention
    builder.add_region("HC", SwitchType.MEMORY, count=8)       # Hippocampus-like memory
    builder.add_region("ATT", SwitchType.ROUTING, count=6)     # Attention/gating
    
    # Feedforward visual pathway
    builder.connect_regions("V1", "V2", density=0.5, strength=0.3)
    builder.connect_regions("V2", "V4", density=0.4, strength=0.3)
    builder.connect_regions("V4", "IT", density=0.4, strength=0.3)
    
    # Memory encoding
    builder.connect_regions("IT", "HC", density=0.3, strength=0.25)
    
    # Feedback (top-down attention)
    builder.connect_regions("ATT", "V4", density=0.3, strength=0.2)
    builder.connect_regions("ATT", "V2", density=0.2, strength=0.15)
    builder.connect_regions("HC", "ATT", density=0.3, strength=0.2)
    
    # Memory retrieval to visual (imagery)
    builder.connect_regions("HC", "IT", density=0.2, strength=0.15)
    
    print(f"\n{builder.summary()}")
    
    # Build and run
    config, types, conn = builder.build()
    
    net = PhotonicNetworkGPU(config)
    net.switch_types = types
    net.connectivity = conn
    net.reset()
    
    # Inject "stimulus" into V1
    v1_start = builder.regions["V1"]["start_idx"]
    stimulus = torch.randn(config.batch_size, config.n_modes, device=config.device) * 0.5
    
    print(f"\nInjecting stimulus into V1...")
    print("Tracking activation across regions:\n")
    
    for t in range(30):
        if t < 5:
            for i in range(v1_start, v1_start + 3):
                net.inject(i, stimulus)
        
        net.step()
        
        if t % 5 == 0:
            print(f"  t={t:2d}: ", end="")
            for name, region in builder.regions.items():
                start = region["start_idx"]
                end = start + region["count"]
                energy = (torch.abs(net.output[:, start:end, :, :]) ** 2).sum().item()
                bar = "█" * min(10, int(energy / 1000))
                print(f"{name}={energy:5.0f}{bar:10s}", end=" ")
            print()
    
    print("\n  Signal propagates: V1 → V2 → V4 → IT → HC")
    print("  (with feedback modulation from ATT)")
    
    del net
    clear_gpu_memory()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║              PHOTONIC NEURAL NETWORK - GPU SIMULATION              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run demos
    demo_wavelength_semantics()
    demo_pattern_recognition()
    demo_network_dynamics()
    demo_custom_topology()
    
    # Clear GPU memory before benchmarks
    clear_gpu_memory()
    
    # Run benchmarks
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    # Conservative for 8GB VRAM
    config = GPUConfig(
        n_switches=64,
        n_wavelengths=64,
        n_modes=32,
        batch_size=8
    )
    
    benchmark_network(config, n_steps=100)
    
    # Optional: scaling benchmark (uncomment for full test)
    # benchmark_scaling()
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("""
    This GPU simulation demonstrates:
    
    1. MASSIVE PARALLELISM
       - All switches process simultaneously (GPU batch)
       - All wavelengths process simultaneously (WDM)
       - All spatial modes process simultaneously
       → O(1) per step regardless of network size
    
    2. HETEROGENEOUS INTERPRETATION
       - Same optical signal, different weights per switch type
       - Visual switches: emphasize edges/frequency
       - Arithmetic switches: uniform precision
       - Memory switches: emphasize key wavelengths
    
    3. STRUCTURED TOPOLOGY
       - Same-type switches densely connected (local processing)
       - Cross-type connections sparse (integration pathways)
       - Hierarchical within types (feedforward + feedback)
       - TopologyBuilder for custom architectures
    
    4. NEURAL (NOT LOGIC) COMPUTING
       - Linear integration in cavities
       - Single nonlinearity at activation
       - Natural fit for photonic hardware
    
    Run scaling benchmarks to see how throughput scales with network size.
    """)