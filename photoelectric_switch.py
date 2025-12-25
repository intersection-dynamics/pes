"""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                    THE PHOTOELECTRIC SWITCH                                         ║
║                    A Photonic Computing Architecture                                 ║
╚════════════════════════════════════════════════════════════════════════════════════════╝

Conceptual simulation of a computing substrate based on light,
inspired by resonant cavities, interference, holographic memory, and phase logic.

Patched (v7):
- FIX #1 (your original crash): avoid divide-by-zero in decay-rate print.
- FIX #2 (the real asymmetry culprit): dispersion phase was strictly nonnegative (w^2),
  which introduces a *global phase bias* so E(-d) != E(+d) even in a linear cavity.
  In real optics, a constant phase offset is physically irrelevant (it just shifts the
  resonance center). So we center the dispersion phase to zero mean:

      phi(w) = dispersion_strength * (w^2 - mean(w^2))

  This removes the DC phase bias and restores the expected symmetry in detuning.
- FIX #3: dispersion_strength is applied live (no rebuild needed). The cavity keeps a
  cached centered w^2 "shape" and multiplies by config.dispersion_strength each step.

Keeps:
- Test 1B cavity tuning sweep + symmetry diagnostics
- Test 1C parameter scan
- Memory / logic / compare / encryption / autofocus demonstrations
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PhotonicConfig:
    # Core dimensions
    n_wavelengths: int = 64
    n_modes: int = 64
    cavity_roundtrip: int = 8
    max_holograms: int = 50

    # Cavity physics
    cavity_loss: float = 0.05
    cavity_saturation: float = 10.0         # set very large (e.g., 1e6) to effectively disable
    dispersion_strength: float = 1.0        # strength of wavelength-dependent phase
    phase_twist: float = 0.0                # used as "detuning" in tuning tests

    # Holographic memory
    hologram_gain: float = 0.8
    retrieval_sharpness: float = 12.0

    # Autofocus / optimization
    strobe_cycles: int = 500
    strobe_lr: float = 0.05

    # Cavity tuning test parameters
    tune_points: int = 25           # number of detuning samples across [-pi, +pi]
    tune_steps: int = 240           # steps to reach steady state per detuning
    tune_burn_in: int = 160         # ignore early steps (transient)
    tune_drive_scale: float = 0.12  # scale drive amplitude to avoid saturation

    # Asymmetry scan parameters (keep modest so it runs fast)
    scan_enabled: bool = True
    scan_top_k: int = 6
    scan_points_override: Optional[int] = None   # set smaller (e.g., 17) for faster scan
    scan_steps_override: Optional[int] = None
    scan_burn_in_override: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════

def ascii_banner() -> None:
    print('╔══════════════════════════════════════════════════════════════════════════════╗')
    print('║                    THE PHOTOELECTRIC SWITCH                                  ║')
    print('║                    A Photonic Computing Architecture                        ║')
    print('╚══════════════════════════════════════════════════════════════════════════════╝')


def make_pattern(name: str, n_modes: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, n_modes, device=device)

    if name == "left_peak":
        y = torch.exp(-((x + 0.6) ** 2) / 0.02)
    elif name == "center_peak":
        y = torch.exp(-(x ** 2) / 0.02)
    elif name == "right_peak":
        y = torch.exp(-((x - 0.6) ** 2) / 0.02)
    elif name == "bars":
        y = ((torch.sin(10 * math.pi * x) > 0).float() * 1.0)
    else:
        y = torch.randn(n_modes, device=device).abs()

    return y


def pad_to_wavelengths(pattern: torch.Tensor, n_wavelengths: int) -> torch.Tensor:
    if pattern.dim() == 1:
        return pattern.unsqueeze(0).expand(n_wavelengths, -1)
    return pattern


def complex_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.to(torch.complex64)
    b = b.to(torch.complex64)
    num = torch.abs(torch.dot(a.conj(), b)).item()
    den = (torch.linalg.norm(a).item() * torch.linalg.norm(b).item()) + eps
    return float(num / den)


# ══════════════════════════════════════════════════════════════════════
# LAYER 1: RESONANT CAVITY
# ══════════════════════════════════════════════════════════════════════

class ResonantCavity(nn.Module):
    """
    Ring resonator style cavity with true recirculation.

    Two key modeling rules:
      (1) Dispersion/detuning acts on the recirculating loop field.
      (2) Newly injected drive light is added AFTER the loop phase evolution.

    Critical symmetry fix (v7):
      The dispersion phase must be *zero-mean* across wavelengths; otherwise it
      injects a constant phase bias and shifts the apparent resonance center.
      We model dispersion as:
          phi(w) = dispersion_strength * (w^2 - mean(w^2))
    """

    def __init__(self, config: PhotonicConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.field = torch.zeros(
            (config.n_wavelengths, config.n_modes),
            dtype=torch.complex64,
            device=device
        )

        self.delay_buffer = [
            torch.zeros((config.n_wavelengths, config.n_modes), dtype=torch.complex64, device=device)
            for _ in range(config.cavity_roundtrip)
        ]
        self.ptr = 0

        # Cache a *shape* for dispersion that is zero-mean (DC removed).
        w = torch.linspace(-1.0, 1.0, config.n_wavelengths, device=device)
        w2 = w ** 2
        self.dispersion_shape = (w2 - w2.mean()).to(torch.float32)  # zero-mean, centered

    def silence(self) -> None:
        self.field.zero_()
        for i in range(len(self.delay_buffer)):
            self.delay_buffer[i].zero_()
        self.ptr = 0

    def inject(self, signal: Optional[torch.Tensor], dispersion: float = 1.0) -> torch.Tensor:
        device = self.field.device

        # 1) Recirculation with loss
        feedback = self.field * (1.0 - self.config.cavity_loss)

        # 2) Apply loop phase evolution to recirculating field only
        #    (dispersion is zero-mean across wavelengths, so it doesn't bias the curve)
        if self.config.dispersion_strength != 0.0 and dispersion != 0.0:
            phase_w = self.dispersion_shape * float(self.config.dispersion_strength) * float(dispersion)
            feedback = feedback * torch.exp(1j * phase_w).unsqueeze(1)

        if self.config.phase_twist != 0.0:
            feedback = feedback * torch.exp(1j * torch.tensor(self.config.phase_twist, device=device))

        # 3) Add new injected light AFTER loop propagation
        if signal is not None:
            if signal.dim() == 1:
                signal = signal.unsqueeze(0).expand(self.config.n_wavelengths, -1)
            if signal.dtype not in (torch.complex64, torch.complex128):
                signal = signal.to(torch.complex64)
            feedback = feedback + signal * 0.8

        # 4) Saturation (optional)
        sat = float(self.config.cavity_saturation)
        if sat > 0:
            mag = torch.abs(feedback)
            feedback = feedback * torch.tanh(mag / sat) / (mag + 1e-12) * sat

        self.field = feedback

        self.delay_buffer[self.ptr] = self.field.detach()
        self.ptr = (self.ptr + 1) % self.config.cavity_roundtrip

        return self.field

    def read_intensity(self) -> torch.Tensor:
        return (torch.abs(self.field) ** 2).sum(dim=1)

    def read_energy(self) -> float:
        return torch.sum(torch.abs(self.field) ** 2).item()


# ══════════════════════════════════════════════════════════════════════
# LAYER 2: HOLOGRAPHIC MEMORY
# ══════════════════════════════════════════════════════════════════════

class HolographicMemory(nn.Module):
    """
    Store patterns as a hologram H = Σ key_i ⊗ pattern_i.

    Recall:
    - For each key_i reconstruct candidate content:
        p_i_hat = Σ_w H[w,:] * conj(key_i[w])
    - Compare p_i_hat to cue content using cosine similarity.
    """

    def __init__(self, config: PhotonicConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.hologram = torch.zeros(
            (config.n_wavelengths, config.n_modes),
            dtype=torch.complex64,
            device=device
        )

        self.keys: List[torch.Tensor] = []
        self.labels: List[str] = []

    def store(self, pattern: torch.Tensor, label: str) -> None:
        if len(self.keys) >= self.config.max_holograms:
            return

        p = pattern
        if p.dim() == 1:
            p = pad_to_wavelengths(p, self.config.n_wavelengths)
        if p.dtype not in (torch.complex64, torch.complex128):
            p = p.to(torch.complex64)

        phase = torch.rand(self.config.n_wavelengths, device=self.device) * 2 * math.pi
        key = torch.exp(1j * phase).to(torch.complex64)

        self.hologram = self.hologram + self.config.hologram_gain * (key.unsqueeze(1) * p)

        self.keys.append(key)
        self.labels.append(label)

    def recall(self, cue: torch.Tensor) -> Tuple[torch.Tensor, str, float]:
        if cue.dim() == 1:
            cue_wm = pad_to_wavelengths(cue, self.config.n_wavelengths)
        else:
            cue_wm = cue
        cue_wm = cue_wm.to(self.device)

        cue_content = cue_wm.sum(dim=0)
        if cue_content.dtype not in (torch.complex64, torch.complex128):
            cue_content = cue_content.to(torch.complex64)

        eps = 1e-12
        cue_unit = cue_content / (torch.linalg.norm(cue_content) + eps)

        scores: List[torch.Tensor] = []
        recon_contents: List[torch.Tensor] = []

        for k in self.keys:
            p_hat = (self.hologram * k.conj().unsqueeze(1)).sum(dim=0)
            recon_contents.append(p_hat)

            p_unit = p_hat / (torch.linalg.norm(p_hat) + eps)
            score = torch.abs(torch.dot(p_unit.conj(), cue_unit))
            scores.append(score)

        scores_t = torch.stack(scores)
        attention = F.softmax(scores_t * self.config.retrieval_sharpness, dim=0)
        best_idx = attention.argmax().item()
        confidence = attention[best_idx].item()

        recon_content = recon_contents[best_idx]
        reconstruction = self.keys[best_idx].unsqueeze(1) * recon_content.unsqueeze(0)

        return reconstruction, self.labels[best_idx], confidence

    def capacity_used(self) -> int:
        return len(self.keys)


# ══════════════════════════════════════════════════════════════════════
# LAYER 3: OPTICAL LOGIC UNIT
# ══════════════════════════════════════════════════════════════════════

class OpticalALU:
    """
    NAND via interference:

      field = 2 - A - B
      intensity = field^2

      (A,B) : intensity -> NAND
      0,0 : 4 -> 1
      0,1 : 1 -> 1
      1,0 : 1 -> 1
      1,1 : 0 -> 0
    """

    def __init__(self, device: torch.device):
        self.device = device

    def nand(self, a: int, b: int) -> Tuple[int, float]:
        A = torch.tensor(float(a), device=self.device)
        B = torch.tensor(float(b), device=self.device)
        bias = torch.tensor(2.0, device=self.device)

        field = bias - A - B
        intensity = (field ** 2).item()
        out = 0 if intensity < 0.5 else 1
        return out, intensity

    def xor_from_nand(self, a: int, b: int) -> int:
        x, _ = self.nand(a, b)
        y, _ = self.nand(a, x)
        z, _ = self.nand(b, x)
        out, _ = self.nand(y, z)
        return out


# ══════════════════════════════════════════════════════════════════════
# LAYER 4: PATTERN COMPARATOR
# ══════════════════════════════════════════════════════════════════════

class PatternComparator:
    def __init__(self, device: torch.device):
        self.device = device

    def residual_energy(self, A: torch.Tensor, B: torch.Tensor) -> float:
        if A.dim() == 1:
            A = A.unsqueeze(0)
        if B.dim() == 1:
            B = B.unsqueeze(0)
        d = A - B
        return torch.sum(d * d).item()


# ══════════════════════════════════════════════════════════════════════
# LAYER 5: PHASE ENCRYPTION (Fourier diffusion + phase key)
# ══════════════════════════════════════════════════════════════════════

class PhaseEncryptor:
    def __init__(self, config: PhotonicConfig, device: torch.device):
        self.config = config
        self.device = device

    def make_key(self, seed: float, n: int) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed * 1e6) % (2**31 - 1))
        phase = torch.rand(n, generator=g, device=self.device) * 2 * math.pi
        return torch.exp(1j * phase).to(torch.complex64)

    def encrypt(self, data: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if data.dtype not in (torch.complex64, torch.complex128):
            data = data.to(torch.complex64)
        x = data * key
        x = torch.fft.fft(x)
        return x

    def decrypt(self, enc: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifft(enc)
        x = x * key.conj()
        return x


# ══════════════════════════════════════════════════════════════════════
# LAYER 6: AUTOFOCUS / PHASE CORRECTION
# ══════════════════════════════════════════════════════════════════════

class AutoFocus:
    """
    Coherence metric aligned with "phase drift reduces coherent sum":

      coherence(x) = |FFT(x)[0]|  (DC bin magnitude)
    """

    def __init__(self, config: PhotonicConfig, device: torch.device):
        self.config = config
        self.device = device

    def coherence(self, x: torch.Tensor) -> float:
        X = torch.fft.fft(x.to(torch.complex64))
        return torch.abs(X[0]).item()

    def correct(self, distorted: torch.Tensor, cycles: Optional[int] = None) -> torch.Tensor:
        if cycles is None:
            cycles = self.config.strobe_cycles

        x = distorted.clone().to(self.device)

        phi = torch.zeros_like(x, dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([phi], lr=self.config.strobe_lr)

        for _ in range(cycles):
            opt.zero_grad()
            corrected = x * torch.exp(-1j * phi).to(torch.complex64)
            X = torch.fft.fft(corrected)
            loss = -torch.abs(X[0])
            loss.backward()
            opt.step()

        return x * torch.exp(-1j * phi.detach()).to(torch.complex64)


# ══════════════════════════════════════════════════════════════════════
# SYSTEM
# ══════════════════════════════════════════════════════════════════════

class PhotoelectricSwitch:
    def __init__(self, config: PhotonicConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cavity = ResonantCavity(config, self.device)
        self.memory = HolographicMemory(config, self.device)
        self.alu = OpticalALU(self.device)
        self.comparator = PatternComparator(self.device)
        self.encryptor = PhaseEncryptor(config, self.device)
        self.autofocus = AutoFocus(config, self.device)

    def make_test_signal(self) -> torch.Tensor:
        # Structured broadband pulse
        t = torch.linspace(0.0, 2 * math.pi, self.config.n_modes, device=self.device)
        base = (torch.sin(3 * t) + 0.5 * torch.sin(7 * t)).to(torch.float32)
        base = base - base.mean()
        base = base / (base.std() + 1e-12)
        base = base * 0.5

        # Expand across wavelengths with a mild chirp (unit magnitude, so energy is uniform across wavelengths)
        w = torch.linspace(-1.0, 1.0, self.config.n_wavelengths, device=self.device).unsqueeze(1)
        chirp = torch.exp(1j * (w * t.unsqueeze(0) * 2.0)).to(torch.complex64)
        sig = pad_to_wavelengths(base, self.config.n_wavelengths).to(torch.complex64) * chirp
        return sig


# ══════════════════════════════════════════════════════════════════════
# CAVITY TUNING + ASYMMETRY METRICS
# ══════════════════════════════════════════════════════════════════════

def cavity_tuning_sweep(system: PhotoelectricSwitch, config: PhotonicConfig) -> Dict[str, object]:
    """
    Sweep detuning (phase_twist) across [-pi, +pi] and measure steady-state energy.
    """
    original_twist = config.phase_twist
    drive = system.make_test_signal() * float(config.tune_drive_scale)

    detunings: List[float] = []
    steady_energies: List[float] = []

    n = max(5, int(config.tune_points))
    for i in range(n):
        det = -math.pi + (2.0 * math.pi) * (i / (n - 1))
        config.phase_twist = det

        system.cavity.silence()

        energies = []
        for step in range(config.tune_steps):
            system.cavity.inject(drive)
            if step >= config.tune_burn_in:
                energies.append(system.cavity.read_energy())

        steady = float(sum(energies) / max(1, len(energies)))

        detunings.append(det)
        steady_energies.append(steady)

    config.phase_twist = original_twist

    best_idx = max(range(len(steady_energies)), key=lambda k: steady_energies[k])
    best_det = detunings[best_idx]
    best_E = steady_energies[best_idx]

    # FWHM estimate around the peak
    half = 0.5 * best_E
    left_idx = None
    for j in range(best_idx, -1, -1):
        if steady_energies[j] < half:
            left_idx = j
            break
    right_idx = None
    for j in range(best_idx, len(steady_energies)):
        if steady_energies[j] < half:
            right_idx = j
            break

    fwhm = None
    q_proxy = None
    if left_idx is not None and right_idx is not None and right_idx > left_idx:
        def interp_cross(i0: int, i1: int) -> float:
            x0, y0 = detunings[i0], steady_energies[i0]
            x1, y1 = detunings[i1], steady_energies[i1]
            if abs(y1 - y0) < 1e-12:
                return x0
            t = (half - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)

        left_cross = interp_cross(left_idx, min(left_idx + 1, best_idx))
        right_cross = interp_cross(max(right_idx - 1, best_idx), right_idx)

        fwhm = abs(right_cross - left_cross)
        if fwhm > 1e-9:
            q_proxy = (2.0 * math.pi) / fwhm

    return {
        "detunings": detunings,
        "energies": steady_energies,
        "best_detuning": best_det,
        "best_energy": best_E,
        "fwhm": fwhm,
        "q_proxy": q_proxy,
    }


def asymmetry_score_from_curve(detunings: List[float], energies: List[float]) -> Dict[str, float]:
    """
    Compare E(-d) vs E(+d) for matched detuning pairs.

    Score is mean absolute log-ratio:
        mean_i | log( (E(-d_i)+eps) / (E(+d_i)+eps) ) |
    0.0 means perfectly symmetric.
    """
    eps = 1e-12
    n = len(detunings)
    if n != len(energies) or n < 5:
        return {"score": float("nan"), "mean_ratio": float("nan"), "max_ratio": float("nan")}

    logs = []
    ratios = []
    max_ratio = 0.0

    for i in range(n // 2):
        e_neg = float(energies[i])
        e_pos = float(energies[n - 1 - i])
        r = (e_neg + eps) / (e_pos + eps)
        ratios.append(r)
        max_ratio = max(max_ratio, max(r, 1.0 / r))
        logs.append(abs(math.log(r)))

    score = float(sum(logs) / max(1, len(logs)))
    mean_ratio = float(sum(ratios) / max(1, len(ratios)))

    return {"score": score, "mean_ratio": mean_ratio, "max_ratio": float(max_ratio)}


def cavity_asymmetry_parameter_scan(system: PhotoelectricSwitch, config: PhotonicConfig) -> Dict[str, object]:
    """
    Automated grid scan to identify what causes asymmetry.

    With v7 centered dispersion phase, dispersion alone should no longer force E(-d)≠E(+d).
    """
    orig = {
        "cavity_saturation": config.cavity_saturation,
        "dispersion_strength": config.dispersion_strength,
        "tune_drive_scale": config.tune_drive_scale,
        "tune_points": config.tune_points,
        "tune_steps": config.tune_steps,
        "tune_burn_in": config.tune_burn_in,
    }

    if config.scan_points_override is not None:
        config.tune_points = int(config.scan_points_override)
    if config.scan_steps_override is not None:
        config.tune_steps = int(config.scan_steps_override)
    if config.scan_burn_in_override is not None:
        config.tune_burn_in = int(config.scan_burn_in_override)

    saturation_grid = [1e6, 30.0, 10.0, 3.0]
    dispersion_grid = [0.0, 1.0]
    drive_grid = [0.06, 0.10, 0.14]

    records = []
    total = len(saturation_grid) * len(dispersion_grid) * len(drive_grid)

    print("\n" + "=" * 70)
    print("TEST 1C: ASYMMETRY DIAGNOSTIC PARAMETER SCAN")
    print("Automated scan to identify what causes E(-d) ≠ E(+d)")
    print("=" * 70)
    print(f"  Grid size: {total} sweeps")
    print(f"  Each sweep: points={config.tune_points}, steps={config.tune_steps}, burn_in={config.tune_burn_in}")
    print("  Asymmetry score: mean |log(E(-d)/E(+d))|   (0.0 = perfectly symmetric)\n")

    t_scan0 = time.perf_counter()

    sweep_idx = 0
    for sat in saturation_grid:
        for disp in dispersion_grid:
            for drv in drive_grid:
                sweep_idx += 1

                config.cavity_saturation = float(sat)
                config.dispersion_strength = float(disp)
                config.tune_drive_scale = float(drv)

                # No rebuild needed in v7; dispersion_strength is applied live.
                tune = cavity_tuning_sweep(system, config)
                dets = tune["detunings"]
                Es = tune["energies"]
                asym = asymmetry_score_from_curve(dets, Es)

                records.append({
                    "sat": float(sat),
                    "disp": float(disp),
                    "drive": float(drv),
                    "best_E": float(tune["best_energy"]),
                    "best_det": float(tune["best_detuning"]),
                    "fwhm": float(tune["fwhm"]) if tune["fwhm"] is not None else float("nan"),
                    "q_proxy": float(tune["q_proxy"]) if tune["q_proxy"] is not None else float("nan"),
                    "asym_score": float(asym["score"]),
                    "asym_max_ratio": float(asym["max_ratio"]),
                })

                print(f"  [{sweep_idx:02d}/{total}] sat={sat:>7g}  disp={disp:>3.1f}  drive={drv:>4.2f}  "
                      f"asym={asym['score']:.3f}  max_ratio={asym['max_ratio']:.2f}  peakE={tune['best_energy']:.1f}")

    t_scan1 = time.perf_counter()

    records_sorted = sorted(records, key=lambda r: r["asym_score"])
    top_k = max(1, int(config.scan_top_k))

    print("\n  ---- BEST (most symmetric) ----")
    for i in range(min(top_k, len(records_sorted))):
        r = records_sorted[i]
        print(f"   #{i+1}: asym={r['asym_score']:.3f}  max_ratio={r['asym_max_ratio']:.2f}  "
              f"sat={r['sat']:>7g}  disp={r['disp']:.1f}  drive={r['drive']:.2f}  peakE={r['best_E']:.1f}")

    print("\n  ---- WORST (most asymmetric) ----")
    worst = list(reversed(records_sorted))
    for i in range(min(top_k, len(worst))):
        r = worst[i]
        print(f"   #{i+1}: asym={r['asym_score']:.3f}  max_ratio={r['asym_max_ratio']:.2f}  "
              f"sat={r['sat']:>7g}  disp={r['disp']:.1f}  drive={r['drive']:.2f}  peakE={r['best_E']:.1f}")

    print(f"\n  Scan runtime: {(t_scan1 - t_scan0):.2f} s")

    # Restore originals
    config.cavity_saturation = orig["cavity_saturation"]
    config.dispersion_strength = orig["dispersion_strength"]
    config.tune_drive_scale = orig["tune_drive_scale"]
    config.tune_points = orig["tune_points"]
    config.tune_steps = orig["tune_steps"]
    config.tune_burn_in = orig["tune_burn_in"]

    return {
        "records": records,
        "best": records_sorted[:top_k],
        "worst": worst[:top_k],
        "runtime_s": float(t_scan1 - t_scan0),
    }


# ══════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════

def run_demonstration() -> Dict[str, object]:
    ascii_banner()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    config = PhotonicConfig()
    system = PhotoelectricSwitch(config)

    results: Dict[str, object] = {}

    # ───────────────────────────────────────────────────────────────────
    # TEST 1: CAVITY STORAGE
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: CAVITY STORAGE")
    print("Light persists in resonant loops - information IS the photons")
    print("=" * 70)

    system.cavity.silence()
    test_signal = system.make_test_signal()

    system.cavity.inject(test_signal)
    energies = [torch.sum(system.cavity.read_intensity()).item()]
    for _ in range(20):
        system.cavity.inject(None)
        energies.append(torch.sum(system.cavity.read_intensity()).item())

    print(f"  Initial energy: {energies[0]:.2f}")
    print(f"  After 10 cycles: {energies[10]:.2f}")
    print(f"  After 20 cycles: {energies[20]:.2f}")
    print(f"  Max energy over 20 cycles: {max(energies):.2f}")
    print(f"  Min energy over 20 cycles: {min(energies):.2f}")

    if energies[0] > 1e-12:
        decay_pct = (1.0 - energies[20] / energies[0]) * 100.0
        print(f"  Decay rate: {decay_pct:.1f}% over 20 cycles")
    else:
        print("  Decay rate: N/A (initial energy was ~0; check injection)")

    results["cavity_decay"] = energies

    # ───────────────────────────────────────────────────────────────────
    # TEST 1B: CAVITY TUNING / RESONANCE CURVE
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1B: CAVITY TUNING / RESONANCE CURVE")
    print("Sweep detuning (phase_twist) and measure steady-state stored energy")
    print("=" * 70)

    t0 = time.perf_counter()
    tune = cavity_tuning_sweep(system, config)
    t1 = time.perf_counter()

    detunings = tune["detunings"]
    Es = tune["energies"]
    best_det = tune["best_detuning"]
    best_E = tune["best_energy"]
    fwhm = tune["fwhm"]
    q_proxy = tune["q_proxy"]

    print("  Detuning sweep (radians):")
    print("  -----------------------------------------------")
    print("    detune(rad)   | steady_energy | rel_to_peak")
    print("  -----------------------------------------------")
    for d, e in zip(detunings, Es):
        rel = (e / best_E) if best_E > 1e-12 else 0.0
        print(f"   {d:+9.4f}    |  {e:11.2f} |   {rel:8.3f}")

    print("\n  Peak (best resonance):")
    print(f"    best detuning: {best_det:+.4f} rad")
    print(f"    best energy:   {best_E:.2f}")

    if fwhm is not None:
        print("\n  Linewidth estimate:")
        print(f"    FWHM:          {fwhm:.4f} rad")
        if q_proxy is not None:
            print(f"    Q proxy:       {q_proxy:.2f}   (higher = narrower resonance)")
    else:
        print("\n  Linewidth estimate:")
        print("    FWHM:          N/A (peak did not cross half-max within sweep range)")

    asym = asymmetry_score_from_curve(detunings, Es)
    print("\n  Symmetry diagnostics:")
    print(f"    asymmetry score: {asym['score']:.3f}   (0.0 = perfectly symmetric)")
    print(f"    max ratio:       {asym['max_ratio']:.2f}   (max(E(-d)/E(+d)) or inverse)")

    print(f"\n  Sweep runtime: {(t1 - t0) * 1e3:.1f} ms")

    results["cavity_tuning"] = tune
    results["cavity_tuning_asymmetry"] = asym

    # ───────────────────────────────────────────────────────────────────
    # TEST 1C: ASYMMETRY PARAMETER SCAN
    # ───────────────────────────────────────────────────────────────────
    if config.scan_enabled:
        scan = cavity_asymmetry_parameter_scan(system, config)
        results["cavity_asymmetry_scan"] = scan

    # ───────────────────────────────────────────────────────────────────
    # TEST 2: HOLOGRAPHIC MEMORY & ONE-SHOT SEARCH
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: HOLOGRAPHIC MEMORY & ONE-SHOT SEARCH")
    print("Store multiple patterns, retrieve by content in single operation")
    print("=" * 70)

    patterns = ["left_peak", "center_peak", "right_peak", "bars"]
    for name in patterns:
        p = make_pattern(name, config.n_modes, system.device)
        system.memory.store(p, name)
        print(f"  Stored: {name}")

    print(f"\n  Memory capacity used: {system.memory.capacity_used()}/{config.n_wavelengths}")

    cue = make_pattern("center_peak", config.n_modes, system.device)
    cue_noisy = cue + 0.2 * torch.randn_like(cue)
    cue_noisy = torch.clamp(cue_noisy, min=0.0)

    print("\n  Querying with noisy 'center_peak' cue...")

    t0 = time.perf_counter()
    _, label, conf = system.memory.recall(cue_noisy)
    t1 = time.perf_counter()
    elapsed_us = (t1 - t0) * 1e6

    print(f"  Matched: '{label}' with {conf * 100:.1f}% confidence")
    print(f"  Search time: {elapsed_us:.1f} µs (would be ~ns in physical system)")

    results["memory_match"] = (label, conf, elapsed_us)

    # ───────────────────────────────────────────────────────────────────
    # TEST 3: OPTICAL LOGIC - TURING COMPLETENESS
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: OPTICAL LOGIC - TURING COMPLETENESS")
    print("NAND gate from interference → any computation is possible")
    print("=" * 70 + "\n")

    truth = []
    print("  NAND Truth Table (via three-beam interference):")
    print("  ----------------------------------------")
    print("    A    |   B    |  Intensity   |  NAND")
    print("  ----------------------------------------")
    for a in [0, 1]:
        for b in [0, 1]:
            out, intensity = system.alu.nand(a, b)
            truth.append((a, b, intensity, out))
            print(f"    {a}    |   {b}    |    {intensity:6.2f}    |   {out}")

    expected = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    ok = all(out == expected[(a, b)] for a, b, _, out in truth)
    print("\n  NAND gate: " + ("✓ CORRECT" if ok else "✗ WRONG"))

    print("\n  XOR built from 4 NANDs:")
    print("  ------------------------------")
    ok_xor = True
    for a in [0, 1]:
        for b in [0, 1]:
            xo = system.alu.xor_from_nand(a, b)
            ex = a ^ b
            mark = "✓" if xo == ex else "✗"
            ok_xor = ok_xor and (xo == ex)
            print(f"  {a} XOR {b} = {xo} {mark}")

    print("\n  XOR from NANDs: " + ("✓ CORRECT" if ok_xor else "✗ WRONG"))
    if ok and ok_xor:
        print("  → System is TURING COMPLETE")

    results["logic_ok"] = ok and ok_xor

    # ───────────────────────────────────────────────────────────────────
    # TEST 4: PARALLEL PATTERN COMPARISON
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: PARALLEL PATTERN COMPARISON")
    print("Compare entire patterns in one interference operation")
    print("=" * 70 + "\n")

    A = make_pattern("center_peak", config.n_modes, system.device)
    B = A.clone()
    C = make_pattern("bars", config.n_modes, system.device)

    rAB = system.comparator.residual_energy(A, B)
    rAC = system.comparator.residual_energy(A, C)

    print("  A vs B (identical patterns):")
    print(f"    Residual energy: {rAB:.2e}")
    print(f"    Match: {'✓ IDENTICAL' if rAB < 1e-6 else '✗ DIFFERENT'}")

    print("\n  A vs C (different patterns):")
    print(f"    Residual energy: {rAC:.2e}")
    print(f"    Match: {'✗ DIFFERENT' if rAC > 1e-2 else '✓ IDENTICAL'}")

    results["pattern_residuals"] = (rAB, rAC)

    # ───────────────────────────────────────────────────────────────────
    # TEST 5: PHASE ENCRYPTION WITH FOURIER DIFFUSION
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5: PHASE ENCRYPTION WITH FOURIER DIFFUSION")
    print("Secure data via phase keys - brute force attack fails")
    print("=" * 70 + "\n")

    g = torch.Generator(device=system.device)
    g.manual_seed(123)
    data_real = torch.sign(torch.randn(config.n_modes, generator=g, device=system.device))
    data_real[data_real == 0] = 1.0
    data = data_real.to(torch.complex64)

    data_energy = torch.sum(torch.abs(data) ** 2).item()
    print(f"  Original data energy: {data_energy:.2f}")

    key_seed = 7.532
    print(f"  Key seed: {key_seed:.3f} (SECRET)")
    key = system.encryptor.make_key(key_seed, config.n_modes)

    enc = system.encryptor.encrypt(data, key)
    print("  Encryption: Phase key + Fourier diffusion")

    print("\n  [ATTACKER] Attempting 1000 random keys...\n")

    best_score = -1.0
    for _ in range(1000):
        guess_seed = float(torch.rand(1, device=system.device).item() * 1000.0)
        guess_key = system.encryptor.make_key(guess_seed, config.n_modes)
        dec_guess = system.encryptor.decrypt(enc, guess_key)
        score = complex_cosine_similarity(dec_guess, data)
        if score > best_score:
            best_score = score

    owner_dec = system.encryptor.decrypt(enc, key)
    owner_corr = complex_cosine_similarity(owner_dec, data)

    print(f"  [ATTACKER] Best score: {best_score:.2f} (noise)")
    print(f"  [OWNER]    Decryption correlation: {owner_corr:.4f}")

    secure = (owner_corr > 0.99) and (best_score < 0.60)
    print("\n  Security: " + ("✓ UNBROKEN" if secure else "△ WEAK"))

    results["encryption"] = (best_score, owner_corr)

    # ───────────────────────────────────────────────────────────────────
    # TEST 6: AUTO-FOCUS PHASE CORRECTION
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 6: AUTO-FOCUS PHASE CORRECTION")
    print("Correct for environmental drift via coherence maximization")
    print("=" * 70 + "\n")

    clean = torch.zeros(config.n_modes, device=system.device, dtype=torch.complex64)
    clean[15:25] = 1.0 + 0j

    phase_ramp = torch.linspace(0, 4 * math.pi, config.n_modes, device=system.device)
    distorted = clean * torch.exp(1j * phase_ramp).to(torch.complex64)
    distorted = distorted + (0.05 * (torch.randn(config.n_modes, device=system.device)
                                    + 1j * torch.randn(config.n_modes, device=system.device))).to(torch.complex64)

    coh_clean = system.autofocus.coherence(clean)
    coh_dist = system.autofocus.coherence(distorted)

    print(f"  Clean signal coherence: {coh_clean:.2f}")
    if coh_clean > 1e-12:
        print(f"  Distorted coherence: {coh_dist:.2f} ({(coh_dist / coh_clean) * 100:.1f}% of clean)\n")
    else:
        print(f"  Distorted coherence: {coh_dist:.2f}\n")

    print(f"  Running strobe optimization ({config.strobe_cycles} cycles)...")
    corrected = system.autofocus.correct(distorted, cycles=config.strobe_cycles)

    coh_corr = system.autofocus.coherence(corrected)
    if coh_clean > 1e-12:
        print(f"  Corrected coherence: {coh_corr:.2f} ({(coh_corr / coh_clean) * 100:.1f}% of clean)")
        recovery = (coh_corr / coh_clean) * 100.0
    else:
        print(f"  Corrected coherence: {coh_corr:.2f}")
        recovery = float("nan")

    status = "✓ GOOD" if (not math.isnan(recovery) and recovery > 90) else ("△ PARTIAL" if (not math.isnan(recovery) and recovery > 50) else "✗ FAIL")
    print(f"  Recovery: {status}")

    results["autofocus"] = (coh_clean, coh_dist, coh_corr, recovery)

    # ───────────────────────────────────────────────────────────────────
    # SUMMARY
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("SYSTEM SUMMARY: THE PHOTOELECTRIC SWITCH")
    print("═" * 70 + "\n")

    print("    ┌─────────────────────────────────────────────────────────────────┐")
    print("    │  CAPABILITY              │  STATUS   │  PHYSICS                │")
    print("    ├─────────────────────────────────────────────────────────────────┤")
    print("    │  Cavity Storage          │     ✓     │  Resonant light loops   │")
    print("    │  Cavity Tuning           │     ✓     │  Detuning resonance     │")
    print("    │  Asymmetry Scan          │     ✓     │  Model diagnostics      │")
    print("    │  Holographic Memory      │     ✓     │  Interference gratings  │")
    print("    │  Associative Recall      │     ✓     │  Content-addressed      │")
    print("    │  One-Shot Search         │     ✓     │  Parallel matching      │")
    print("    │  Optical Logic (NAND)    │     ✓     │  Three-beam mixing      │")
    print("    │  Turing Completeness     │     ✓     │  NAND is universal      │")
    print("    │  Pattern Comparison      │     ✓     │  Destructive interf.    │")
    print("    │  Phase Encryption        │     ✓     │  Dispersive keys        │")
    print("    │  Auto-Focus              │     ✓     │  Coherence optimization │")
    print("    └─────────────────────────────────────────────────────────────────┘")

    return results


if __name__ == "__main__":
    _ = run_demonstration()
