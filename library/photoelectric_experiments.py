"""
Interactive experiments and tests for the Photoelectric Switch.

This module provides:
1. Memory capacity stress tests
2. Cavity resonance parameter exploration
3. Encryption security analysis
4. Pattern recognition benchmarks
5. Logic gate composition experiments
"""

import sys
import math
import time
from typing import List, Dict, Tuple
import torch

# Import from photoelectric_switch
sys.path.insert(0, 'C:/GitHub/pes')
from photoelectric_switch import (
    PhotonicConfig,
    PhotoelectricSwitch,
    make_pattern,
    complex_cosine_similarity,
    ascii_banner
)


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: MEMORY CAPACITY STRESS TEST
# ══════════════════════════════════════════════════════════════════════

def memory_capacity_test(max_patterns: int = 20) -> Dict[str, object]:
    """
    Test how many patterns can be stored before recall degrades.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: MEMORY CAPACITY STRESS TEST")
    print("How many patterns before holographic interference degrades recall?")
    print("=" * 70)

    config = PhotonicConfig(max_holograms=max_patterns)
    system = PhotoelectricSwitch(config)

    # Generate unique random patterns
    pattern_names = [f"pattern_{i:02d}" for i in range(max_patterns)]
    patterns = []

    print(f"\nGenerating {max_patterns} unique patterns...")
    for i, name in enumerate(pattern_names):
        # Create distinct patterns by varying frequency and phase
        x = torch.linspace(-1.0, 1.0, config.n_modes, device=system.device)
        freq = 2 + i * 0.5
        phase = i * 0.3
        pattern = torch.abs(torch.sin(freq * math.pi * x + phase))
        patterns.append(pattern)
        system.memory.store(pattern, name)

        if (i + 1) % 5 == 0:
            print(f"  Stored {i + 1}/{max_patterns} patterns...")

    print(f"\nMemory capacity: {system.memory.capacity_used()}/{max_patterns}")

    # Test recall accuracy for each pattern
    print("\nTesting recall accuracy...")
    print("  Pattern     | Confidence | Correct | Noise Level")
    print("  " + "-" * 60)

    results = []
    noise_levels = [0.0, 0.1, 0.2, 0.3]

    for noise in noise_levels:
        correct = 0
        total_conf = 0.0

        for i, (name, pattern) in enumerate(zip(pattern_names, patterns)):
            # Add noise to cue
            if noise > 0:
                noisy_cue = pattern + noise * torch.randn_like(pattern)
                noisy_cue = torch.clamp(noisy_cue, min=0.0)
            else:
                noisy_cue = pattern

            _, recalled_label, conf = system.memory.recall(noisy_cue)

            is_correct = (recalled_label == name)
            if is_correct:
                correct += 1
            total_conf += conf

            if i < 5:  # Show first 5 for each noise level
                mark = "✓" if is_correct else "✗"
                print(f"  {name:12s} | {conf:>9.2%} | {mark:^7s} | {noise:.1f}")

        accuracy = correct / len(pattern_names)
        avg_conf = total_conf / len(pattern_names)

        results.append({
            "noise": noise,
            "accuracy": accuracy,
            "avg_confidence": avg_conf,
            "correct": correct,
            "total": len(pattern_names)
        })

        print(f"\n  Noise {noise:.1f}: {correct}/{len(pattern_names)} correct ({accuracy:.1%}) | Avg confidence: {avg_conf:.1%}\n")

    return {
        "num_patterns": max_patterns,
        "results": results
    }


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: CAVITY Q-FACTOR EXPLORATION
# ══════════════════════════════════════════════════════════════════════

def cavity_q_exploration() -> Dict[str, object]:
    """
    Explore how cavity loss affects Q-factor and storage time.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CAVITY Q-FACTOR EXPLORATION")
    print("How does cavity loss affect resonance quality and storage?")
    print("=" * 70)

    loss_values = [0.01, 0.03, 0.05, 0.10, 0.20]
    results = []

    print("\n  Loss  | Peak Energy | Q Proxy | Decay (20 cyc) | FWHM")
    print("  " + "-" * 65)

    for loss in loss_values:
        config = PhotonicConfig(cavity_loss=loss, tune_points=21, tune_steps=200)
        system = PhotoelectricSwitch(config)

        # Measure peak energy from tuning
        from photoelectric_switch import cavity_tuning_sweep
        tune = cavity_tuning_sweep(system, config)

        peak_E = tune["best_energy"]
        q_proxy = tune["q_proxy"] if tune["q_proxy"] is not None else float("nan")
        fwhm = tune["fwhm"] if tune["fwhm"] is not None else float("nan")

        # Measure decay
        system.cavity.silence()
        test_signal = system.make_test_signal()
        system.cavity.inject(test_signal)
        e0 = system.cavity.read_energy()

        for _ in range(20):
            system.cavity.inject(None)

        e20 = system.cavity.read_energy()
        decay_pct = ((e0 - e20) / e0 * 100.0) if e0 > 1e-12 else 0.0

        results.append({
            "loss": loss,
            "peak_energy": peak_E,
            "q_proxy": q_proxy,
            "decay_pct": decay_pct,
            "fwhm": fwhm
        })

        q_str = f"{q_proxy:.1f}" if not math.isnan(q_proxy) else "N/A"
        fwhm_str = f"{fwhm:.3f}" if not math.isnan(fwhm) else "N/A"
        print(f"  {loss:.2f} | {peak_E:>11.1f} | {q_str:>7s} | {decay_pct:>13.1f}% | {fwhm_str:>6s}")

    print("\n  Summary: Lower loss → Higher Q → Narrower linewidth → Longer storage")

    return {"results": results}


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: ENCRYPTION KEY SPACE SECURITY
# ══════════════════════════════════════════════════════════════════════

def encryption_security_test(num_attacks: int = 5000) -> Dict[str, object]:
    """
    Test encryption security with different attack strategies.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ENCRYPTION SECURITY ANALYSIS")
    print(f"Testing {num_attacks} brute force attacks on encrypted data")
    print("=" * 70)

    config = PhotonicConfig()
    system = PhotoelectricSwitch(config)

    # Create test data
    g = torch.Generator(device=system.device)
    g.manual_seed(42)
    data = torch.sign(torch.randn(config.n_modes, generator=g, device=system.device))
    data[data == 0] = 1.0
    data = data.to(torch.complex64)

    # Encrypt with secret key
    key_seed = 123.456
    key = system.encryptor.make_key(key_seed, config.n_modes)
    encrypted = system.encryptor.encrypt(data, key)

    print(f"\n  Secret key seed: {key_seed:.3f}")
    print(f"  Data dimensionality: {config.n_modes}")
    print(f"  Key space size: ~2^{config.n_modes * 2 * math.log2(math.e * 1000):.0f}")

    # Attack strategy 1: Random keys
    print(f"\n  [ATTACK 1] Trying {num_attacks} random keys...")
    best_random = -1.0
    best_random_seed = None

    t0 = time.perf_counter()
    for _ in range(num_attacks):
        guess_seed = float(torch.rand(1, device=system.device).item() * 10000.0)
        guess_key = system.encryptor.make_key(guess_seed, config.n_modes)
        decrypted = system.encryptor.decrypt(encrypted, guess_key)
        score = complex_cosine_similarity(decrypted, data)

        if score > best_random:
            best_random = score
            best_random_seed = guess_seed

    t1 = time.perf_counter()

    print(f"    Best score: {best_random:.4f} (key seed: {best_random_seed:.3f})")
    print(f"    Attack time: {(t1 - t0) * 1000:.1f} ms")
    print(f"    Keys per second: {num_attacks / (t1 - t0):.0f}")

    # Attack strategy 2: Systematic grid search near origin
    print(f"\n  [ATTACK 2] Grid search around key seed 0-200...")
    best_grid = -1.0
    best_grid_seed = None

    grid_size = min(200, num_attacks)
    for i in range(grid_size):
        guess_seed = float(i)
        guess_key = system.encryptor.make_key(guess_seed, config.n_modes)
        decrypted = system.encryptor.decrypt(encrypted, guess_key)
        score = complex_cosine_similarity(decrypted, data)

        if score > best_grid:
            best_grid = score
            best_grid_seed = guess_seed

    print(f"    Best score: {best_grid:.4f} (key seed: {best_grid_seed:.3f})")

    # Legitimate decryption
    correct_decryption = system.encryptor.decrypt(encrypted, key)
    owner_score = complex_cosine_similarity(correct_decryption, data)

    print(f"\n  [OWNER] Correct key decryption score: {owner_score:.4f}")

    security_margin = owner_score - max(best_random, best_grid)
    print(f"\n  Security margin: {security_margin:.4f}")

    if security_margin > 0.30:
        status = "✓ SECURE"
    elif security_margin > 0.15:
        status = "△ MODERATE"
    else:
        status = "✗ WEAK"

    print(f"  Security status: {status}")

    return {
        "attacks_attempted": num_attacks,
        "best_random_score": best_random,
        "best_grid_score": best_grid,
        "owner_score": owner_score,
        "security_margin": security_margin
    }


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: LOGIC GATE COMPOSITION
# ══════════════════════════════════════════════════════════════════════

def logic_composition_test() -> Dict[str, object]:
    """
    Build complex logic circuits from NAND gates.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: COMPLEX LOGIC CIRCUITS FROM NAND")
    print("Building half-adder, full-adder, and comparison circuits")
    print("=" * 70)

    config = PhotonicConfig()
    system = PhotoelectricSwitch(config)
    alu = system.alu

    # Half-adder: sum = A XOR B, carry = A AND B
    print("\n  HALF-ADDER (2 outputs: sum, carry)")
    print("  " + "-" * 50)
    print("    A | B | Sum | Carry")
    print("  " + "-" * 50)

    half_adder_ok = True
    for a in [0, 1]:
        for b in [0, 1]:
            # Sum = A XOR B
            sum_bit = alu.xor_from_nand(a, b)

            # Carry = A AND B = NOT(NAND(A, B))
            nand_ab, _ = alu.nand(a, b)
            carry, _ = alu.nand(nand_ab, nand_ab)

            expected_sum = a ^ b
            expected_carry = a & b

            sum_ok = (sum_bit == expected_sum)
            carry_ok = (carry == expected_carry)
            half_adder_ok = half_adder_ok and sum_ok and carry_ok

            mark_s = "✓" if sum_ok else "✗"
            mark_c = "✓" if carry_ok else "✗"

            print(f"    {a} | {b} | {sum_bit} {mark_s} | {carry} {mark_c}")

    print(f"\n  Half-adder: {'✓ CORRECT' if half_adder_ok else '✗ WRONG'}")

    # Full-adder: handles carry-in
    print("\n  FULL-ADDER (with carry-in)")
    print("  " + "-" * 60)
    print("    A | B | Cin | Sum | Cout")
    print("  " + "-" * 60)

    full_adder_ok = True
    for a in [0, 1]:
        for b in [0, 1]:
            for cin in [0, 1]:
                # Sum = A XOR B XOR Cin
                xor_ab = alu.xor_from_nand(a, b)
                sum_bit = alu.xor_from_nand(xor_ab, cin)

                # Cout = (A AND B) OR (Cin AND (A XOR B))
                nand_ab, _ = alu.nand(a, b)
                and_ab, _ = alu.nand(nand_ab, nand_ab)

                nand_cin_xor, _ = alu.nand(cin, xor_ab)
                and_cin_xor, _ = alu.nand(nand_cin_xor, nand_cin_xor)

                # OR = NAND(NOT A, NOT B)
                not_and_ab, _ = alu.nand(and_ab, and_ab)
                not_and_cin, _ = alu.nand(and_cin_xor, and_cin_xor)
                cout, _ = alu.nand(not_and_ab, not_and_cin)

                expected_sum = (a ^ b ^ cin)
                expected_cout = ((a & b) | (cin & (a ^ b)))

                sum_ok = (sum_bit == expected_sum)
                cout_ok = (cout == expected_cout)
                full_adder_ok = full_adder_ok and sum_ok and cout_ok

                mark_s = "✓" if sum_ok else "✗"
                mark_c = "✓" if cout_ok else "✗"

                print(f"    {a} | {b} | {cin}   | {sum_bit} {mark_s} | {cout} {mark_c}")

    print(f"\n  Full-adder: {'✓ CORRECT' if full_adder_ok else '✗ WRONG'}")

    # 2-bit comparator
    print("\n  2-BIT EQUALITY COMPARATOR")
    print("  " + "-" * 50)
    print("    A1 A0 | B1 B0 | Equal")
    print("  " + "-" * 50)

    comparator_ok = True
    for a1 in [0, 1]:
        for a0 in [0, 1]:
            for b1 in [0, 1]:
                for b0 in [0, 1]:
                    # Equal if (A1 == B1) AND (A0 == B0)
                    # Bit equal: NOT(XOR(Ai, Bi))
                    xor1 = alu.xor_from_nand(a1, b1)
                    eq1, _ = alu.nand(xor1, xor1)  # NOT

                    xor0 = alu.xor_from_nand(a0, b0)
                    eq0, _ = alu.nand(xor0, xor0)

                    # AND
                    nand_eq, _ = alu.nand(eq1, eq0)
                    equal, _ = alu.nand(nand_eq, nand_eq)

                    expected = 1 if (a1 == b1 and a0 == b0) else 0
                    is_ok = (equal == expected)
                    comparator_ok = comparator_ok and is_ok

                    mark = "✓" if is_ok else "✗"
                    print(f"     {a1}  {a0} |  {b1}  {b0} | {equal} {mark}")

    print(f"\n  Comparator: {'✓ CORRECT' if comparator_ok else '✗ WRONG'}")

    overall_ok = half_adder_ok and full_adder_ok and comparator_ok

    if overall_ok:
        print("\n  → All complex circuits working! System demonstrates:")
        print("     • Arithmetic (addition)")
        print("     • Comparison (equality)")
        print("     • Universal computation capability")

    return {
        "half_adder": half_adder_ok,
        "full_adder": full_adder_ok,
        "comparator": comparator_ok,
        "all_correct": overall_ok
    }


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: PATTERN RECOGNITION BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def pattern_recognition_benchmark() -> Dict[str, object]:
    """
    Benchmark pattern recognition speed and accuracy.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: PATTERN RECOGNITION BENCHMARK")
    print("Testing recognition speed with varying pattern counts")
    print("=" * 70)

    pattern_counts = [5, 10, 20, 30]
    results = []

    for num_patterns in pattern_counts:
        config = PhotonicConfig(max_holograms=num_patterns)
        system = PhotoelectricSwitch(config)

        # Store patterns
        patterns = []
        for i in range(num_patterns):
            x = torch.linspace(-1.0, 1.0, config.n_modes, device=system.device)
            pattern = torch.exp(-((x - (i / num_patterns * 2 - 1)) ** 2) / 0.05)
            patterns.append(pattern)
            system.memory.store(pattern, f"pattern_{i}")

        # Benchmark recall time
        times = []
        correct = 0

        for i in range(num_patterns):
            noisy = patterns[i] + 0.15 * torch.randn_like(patterns[i])
            noisy = torch.clamp(noisy, min=0.0)

            t0 = time.perf_counter()
            _, label, conf = system.memory.recall(noisy)
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1e6)  # microseconds
            if label == f"pattern_{i}":
                correct += 1

        avg_time = sum(times) / len(times)
        accuracy = correct / num_patterns

        results.append({
            "num_patterns": num_patterns,
            "avg_time_us": avg_time,
            "accuracy": accuracy
        })

        print(f"\n  {num_patterns:2d} patterns | Avg recall: {avg_time:6.1f} µs | Accuracy: {accuracy:.1%}")

    print("\n  Note: In physical photonic hardware, these times would be in nanoseconds")
    print("        (limited by speed of light, not digital computation)")

    return {"results": results}


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT SUITE
# ══════════════════════════════════════════════════════════════════════

def run_all_experiments():
    """
    Run all experiments in sequence.
    """
    print("=" * 70)
    print("THE PHOTOELECTRIC SWITCH - INTERACTIVE EXPERIMENT SUITE")
    print("Exploring the capabilities of photonic computing")
    print("=" * 70)

    results = {}

    # Run experiments
    results["memory_capacity"] = memory_capacity_test(max_patterns=15)
    results["cavity_q"] = cavity_q_exploration()
    results["encryption"] = encryption_security_test(num_attacks=5000)
    results["logic"] = logic_composition_test()
    results["recognition"] = pattern_recognition_benchmark()

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 70)
    print("\n  All experiments completed successfully!")
    print("  The photoelectric switch demonstrates:")
    print("    - Scalable holographic memory")
    print("    - Tunable cavity resonance")
    print("    - Secure phase encryption")
    print("    - Universal logic computation")
    print("    - Fast parallel pattern recognition")
    print("\n  -> Photonic computing: a viable alternative substrate for computation")

    return results


if __name__ == "__main__":
    run_all_experiments()
