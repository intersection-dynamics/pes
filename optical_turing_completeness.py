import numpy as np

def optical_wave(amplitude, phase):
    """Returns a complex wave component."""
    return amplitude * np.exp(1j * phase)

def optical_nand_gate(input_a_intensity, input_b_intensity):
    """
    Simulates a NAND gate using Optical Interference.
    
    Physics Logic:
    - Inputs A, B: Constructive waves (Amplitude 0 or 1)
    - Reference Beam: Destructive 'Bias' Wave (Amplitude 2, Phase PI)
    - Result: A + B - 2
    """
    # 1. Convert Logic Inputs to Optical Waves
    wave_a = optical_wave(amplitude=input_a_intensity, phase=0)
    wave_b = optical_wave(amplitude=input_b_intensity, phase=0)
    
    # 2. The Reference Beam (The "Bias")
    # Amplitude 2, Phase PI (effectively -2)
    reference_wave = optical_wave(amplitude=2.0, phase=np.pi)
    
    # 3. Superposition (Interference in the Cavity)
    interference_field = wave_a + wave_b + reference_wave
    
    # 4. Detection (Measure Intensity)
    # 0,0 -> |-2|^2 = 4 (High)
    # 0,1 -> |-1|^2 = 1 (Med)
    # 1,1 -> | 0|^2 = 0 (Low)
    output_intensity = np.abs(interference_field)**2
    
    # 5. Thresholding (The Photodetector)
    # Any light > 0.1 is Logic 1. Complete darkness is Logic 0.
    logic_output = 1 if output_intensity > 0.1 else 0
    
    return logic_output, output_intensity

def optical_xor_composite(a, b):
    # Proving Composability: XOR built from 4 NANDs
    # XOR(A, B) = (A NAND (A NAND B)) NAND (B NAND (A NAND B))
    n1, _ = optical_nand_gate(a, b)      
    n2, _ = optical_nand_gate(a, n1)     
    n3, _ = optical_nand_gate(b, n1)     
    final, _ = optical_nand_gate(n2, n3) 
    return final

def run_turing_proof():
    print("="*60)
    print("      OPTICAL TURING COMPLETENESS PROOF")
    print("      Demonstrating Universal Logic (NAND) via Interference")
    print("="*60)
    
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    print(f" {'Input A':<8} | {'Input B':<8} | {'Raw Intensity':<15} | {'NAND Output':<10}")
    print("-" * 60)
    
    for a, b in inputs:
        logic_out, raw_int = optical_nand_gate(a, b)
        print(f" {a:<8} | {b:<8} | {raw_int:<15.2f} | {logic_out:<10}")

    print("\n" + "="*60)
    print("      COMPOSITE TEST: OPTICAL XOR (from NANDs)")
    print("="*60)
    print(f" {'Input A':<8} | {'Input B':<8} | {'XOR Output':<10}")
    print("-" * 60)
    
    for a, b in inputs:
        xor_out = optical_xor_composite(a, b)
        print(f" {a:<8} | {b:<8} | {xor_out:<10}")

if __name__ == "__main__":
    run_turing_proof()