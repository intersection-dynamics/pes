# PES: The Photoelectric Switch Engine
### A Physics-Compliant Holographic Memory & Optical Logic Simulator

![Holographic Interference Field](holographic_art.jpg)
*Figure 1: Visualization of phase dispersion and wave interference within the resonant cavity. Generated via `holographic_art.py`.*

## 🔭 Overview
**PES (Photoelectric Switch)** is a Python-based simulation engine that models next-generation optical computing architectures. Unlike standard neural networks that abstract away the hardware, PES simulates the underlying physics of light—specifically **Phase Dispersion**, **Wave Interference**, and **Resonant Cavities**.

This project demonstrates how **Fourier Keying** can be used to store orthogonal data patterns in a dispersive optical medium, essentially turning light waves into addressable memory slots. It serves as the computational implementation of the theoretical framework described in *[The Hilbert Substrate Framework](The%20Hilbert%20Substrate%20Framework.pdf)*.

## 🚀 Key Capabilities

* **Holographic Content-Addressable Memory (CAM):** Stores and retrieves patterns using wave interference rather than address pointers.
* **Dispersion Compensation:** Implements a "Phase Lens" algorithm to untwist signal distortion caused by cavity resonance.
* **Optical ALU (Arithmetic Logic Unit):** Performs boolean logic (XOR, AND) purely through constructive and destructive interference of optical fields.
* **Steganographic "Glass Safe":** Encrypts data within phase noise, retrievable only with a precise floating-point dispersion key (e.g., `k=7.532`).

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `holographic_core.py` | **The Engine.** Contains the `ResonantCavity` and `HolographicField` classes that simulate the physics of the optical medium. |
| `optical_logic.py` | **The CPU.** Demonstrates optical logic gates (XOR) by colliding inverted waveforms to achieve 0.0 energy states. |
| `holographic_stress_test.py` | **The Audit.** Pushes the system to the Nyquist limit ($N=64$) to demonstrate the physical capacity wall of the medium. |
| `optical_crypto.py` | **The Vault.** A steganography demo showing how data is hidden in phase noise and retrieved via high-precision keys. |
| `holographic_art.py` | **The Visualization.** Generates the spectral interference plots (like the header image) to visualize the internal state of the memory. |

## 🛠️ Quick Start

### Prerequisites
* Python 3.8+
* PyTorch (for tensor operations)
* Matplotlib (for visualization)

```bash
pip install torch numpy matplotlib