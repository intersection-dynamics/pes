#!/usr/bin/env python3
"""
benchmark_strobe_tsplib.py

Rock-solid Berlin52 benchmark for the "strobe / variable-horizon" 2-opt optimizer.

What changed vs your original:
- Uses TSPLIB EUC_2D distance rule (integer rounding): d = int(sqrt(dx^2+dy^2) + 0.5)
- Compares against the TSPLIB-known optimum for berlin52: 7542
- Adds multi-trial reporting (best/mean/std, best gap, and distribution)

Run (Windows, single line):
  python benchmark_strobe_tsplib.py --trials 20 --cycles 100000 --plot

Notes:
- Berlin52 coordinates are the standard TSPLIB instance.
- The objective here is the canonical TSPLIB tour length (integer).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# PART 1: THE DATA (Berlin52)
# ==========================================
BERLIN52_COORDS = np.array([
    [565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
    [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
    [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
    [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
    [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
    [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
    [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
    [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
    [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
    [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0],
    [1340.0, 725.0], [1740.0, 245.0]
], dtype=np.float64)

# TSPLIB-known optimum for berlin52 (EUC_2D rounded integer distances)
OPTIMAL_SCORE = 7542


# ==========================================
# PART 2: TSPLIB distance + cost
# ==========================================
def euc_2d_tsplib(a: np.ndarray, b: np.ndarray) -> int:
    """TSPLIB EUC_2D distance: int(sqrt(dx^2+dy^2) + 0.5)."""
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def make_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Precompute integer distance matrix for speed and determinism."""
    n = int(cities.shape[0])
    D = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            dij = euc_2d_tsplib(cities[i], cities[j])
            D[i, j] = dij
            D[j, i] = dij
    return D


def tsp_cost_int(path: np.ndarray, D: np.ndarray) -> int:
    """Integer TSPLIB tour length."""
    n = int(path.shape[0])
    total = 0
    # Use Python int to avoid int32 overflow paranoia (safe here anyway).
    for k in range(n):
        total += int(D[int(path[k]), int(path[(k + 1) % n])])
    return int(total)


# ==========================================
# PART 3: THE ENTANGLEMENT ENGINE (same idea, deterministic scoring)
# ==========================================
@dataclass
class EntanglementOptimizer:
    cost_func: Callable[[np.ndarray], int]
    state: np.ndarray

    current_cost: int = 0
    best_state: np.ndarray | None = None
    min_cost: int = 0

    def __post_init__(self) -> None:
        self.state = self.state.copy()
        self.current_cost = int(self.cost_func(self.state))
        self.best_state = self.state.copy()
        self.min_cost = int(self.current_cost)

    def mutate_variable_horizon(self, path: np.ndarray, horizon_ratio: float, rng: np.random.Generator) -> np.ndarray:
        """
        Variable-horizon 2-opt.
        horizon_ratio (0.05..1.0):
          - 1.0 = global reach
          - 0.05 = very local reach (still at least 2-opt neighbor-ish)
        """
        n = int(path.shape[0])

        # 1) Anchor
        i = int(rng.integers(0, n))

        # 2) Horizon window
        window_size = max(2, int(n * float(horizon_ratio)))

        # 3) Second index within horizon (wrap)
        offset = int(rng.integers(1, window_size))
        j = (i + offset) % n

        # Slice handling
        if i > j:
            i, j = j, i

        new_path = path.copy()
        new_path[i : j + 1] = new_path[i : j + 1][::-1]
        return new_path

    def step(self, t: int, cycle_length: int, rng: np.random.Generator) -> Tuple[int, float]:
        # === THE STROBE: OSCILLATING HORIZON ===
        phase = (t % cycle_length) / float(cycle_length)

        # Cosine strobe from 1 -> 0 -> 1
        horizon = 0.5 * (math.cos(phase * 2.0 * math.pi) + 1.0)
        horizon = max(horizon, 0.05)

        # Candidate
        candidate_state = self.mutate_variable_horizon(self.state, horizon, rng)
        candidate_cost = int(self.cost_func(candidate_state))

        delta = candidate_cost - self.current_cost

        if delta < 0:
            self.state = candidate_state
            self.current_cost = candidate_cost

            if self.current_cost < self.min_cost:
                self.min_cost = self.current_cost
                self.best_state = self.state.copy()
        else:
            # "Tunneling" proportional to horizon
            tunneling_amplitude = horizon * 0.05

            # Guard against zero / tiny amplitude
            denom = max(tunneling_amplitude * 1000.0, 1e-9)
            accept_p = math.exp(-float(delta) / denom)

            if float(rng.random()) < accept_p:
                self.state = candidate_state
                self.current_cost = candidate_cost

        return int(self.current_cost), float(horizon)


# ==========================================
# PART 4: experiment runner
# ==========================================
def run_one_trial(
    cities: np.ndarray,
    D: np.ndarray,
    cycles: int,
    seed: int,
    cycle_length: int,
    verbose: bool = False,
) -> Tuple[int, np.ndarray, List[int], List[float]]:
    rng = np.random.default_rng(seed)

    n = int(cities.shape[0])
    initial_path = np.arange(n, dtype=np.int32)
    rng.shuffle(initial_path)

    opt = EntanglementOptimizer(
        cost_func=lambda p: tsp_cost_int(p, D),
        state=initial_path,
    )

    cost_history: List[int] = []
    horizon_history: List[float] = []

    # Track a "lock" message like your original, but now aligned to integer objective
    lock_printed = False

    for t in range(int(cycles)):
        cost, horizon = opt.step(t, cycle_length=cycle_length, rng=rng)
        cost_history.append(cost)
        horizon_history.append(horizon)

        if verbose and (t % max(1, cycles // 10) == 0):
            print(f"   Cycle {t:6d}: cost={cost} horizon={horizon:.3f} best={opt.min_cost}")

        if (not lock_printed) and opt.min_cost <= OPTIMAL_SCORE + 10:
            # "within 10 units" is a nice human milestone
            print(f"   ⚡ near-opt region reached at cycle {t}: best={opt.min_cost}")
            lock_printed = True

    best_score = int(opt.min_cost)
    best_path = opt.best_state.copy() if opt.best_state is not None else opt.state.copy()
    return best_score, best_path, cost_history, horizon_history


def summarize(scores: List[int]) -> str:
    arr = np.array(scores, dtype=np.int32)
    best = int(arr.min())
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    med = float(np.median(arr))
    return f"best={best}  mean={mean:.2f}  median={med:.2f}  std={std:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rock-solid Berlin52 benchmark using TSPLIB EUC_2D distances.")
    parser.add_argument("--cycles", type=int, default=100000, help="optimization steps per trial")
    parser.add_argument("--trials", type=int, default=10, help="number of independent random seeds")
    parser.add_argument("--seed", type=int, default=0, help="base seed (trial k uses seed+ k)")
    parser.add_argument("--cycle-length", type=int, default=2000, help="strobe cycle length")
    parser.add_argument("--plot", action="store_true", help="plot best-trial path and dynamics")
    parser.add_argument("--verbose", action="store_true", help="print periodic updates within each trial")
    args = parser.parse_args()

    cities = BERLIN52_COORDS
    D = make_distance_matrix(cities)
    n = int(cities.shape[0])

    print("🌌 BOOTING ENTANGLEMENT ENGINE (Variable Horizon) — TSPLIB EUC_2D")
    print(f"   Instance: berlin52 (n={n})")
    print(f"   TSPLIB optimum: {OPTIMAL_SCORE}")
    print(f"   Trials: {args.trials}   Cycles/trial: {args.cycles}   Base seed: {args.seed}")
    print("")

    all_scores: List[int] = []
    best_overall_score = None
    best_overall_path = None
    best_cost_history: List[int] = []
    best_horizon_history: List[float] = []

    t0 = time.time()
    for k in range(int(args.trials)):
        trial_seed = int(args.seed + k)
        print(f"[Trial {k+1:02d}/{args.trials}] seed={trial_seed}")
        score, path, cost_hist, horiz_hist = run_one_trial(
            cities=cities,
            D=D,
            cycles=args.cycles,
            seed=trial_seed,
            cycle_length=args.cycle_length,
            verbose=args.verbose,
        )
        gap_pct = ((score - OPTIMAL_SCORE) / OPTIMAL_SCORE) * 100.0
        print(f"   🏁 best={score}   gap={gap_pct:.4f}%")
        all_scores.append(score)

        if best_overall_score is None or score < best_overall_score:
            best_overall_score = int(score)
            best_overall_path = path.copy()
            best_cost_history = list(cost_hist)
            best_horizon_history = list(horiz_hist)

    elapsed = time.time() - t0
    assert best_overall_score is not None and best_overall_path is not None

    best_gap = ((best_overall_score - OPTIMAL_SCORE) / OPTIMAL_SCORE) * 100.0

    # Extra "rock-solid" reporting: how often you hit within X units
    within_0 = sum(1 for s in all_scores if s == OPTIMAL_SCORE)
    within_2 = sum(1 for s in all_scores if s <= OPTIMAL_SCORE + 2)
    within_5 = sum(1 for s in all_scores if s <= OPTIMAL_SCORE + 5)
    within_10 = sum(1 for s in all_scores if s <= OPTIMAL_SCORE + 10)

    print("\n" + "=" * 52)
    print("✅ SUMMARY (TSPLIB-comparable, integer objective)")
    print(f"   {summarize(all_scores)}")
    print(f"   best gap: {best_gap:.4f}%  (best={best_overall_score}, optimum={OPTIMAL_SCORE})")
    print(f"   hit-rate: optimal={within_0}/{args.trials}  ≤+2={within_2}/{args.trials}  ≤+5={within_5}/{args.trials}  ≤+10={within_10}/{args.trials}")
    print(f"   elapsed: {elapsed:.2f}s")
    print("=" * 52)

    if args.plot:
        # Path plot (best overall)
        best_coords = cities[best_overall_path]
        plot_coords = np.vstack([best_coords, best_coords[0]])

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(plot_coords[:, 0], plot_coords[:, 1], "-o", markersize=4)
        plt.title(f"Best Tour (TSPLIB EUC_2D): {best_overall_score}")

        plt.subplot(1, 2, 2)
        plt.plot(best_cost_history, linewidth=0.8, label="Tour length (integer)")
        # scale horizon so it visually shares axis, but keep it honest in legend
        scaled_h = np.array(best_horizon_history, dtype=np.float64) * (max(best_cost_history) - min(best_cost_history)) + min(best_cost_history)
        plt.plot(scaled_h, alpha=0.35, linewidth=0.6, label="Horizon (scaled)")
        plt.axhline(y=OPTIMAL_SCORE, linestyle="--", label="TSPLIB optimum (7542)")
        plt.title("Dynamics (best trial)")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
