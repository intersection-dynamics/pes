#!/usr/bin/env python3
"""
benchmark_strobe_tsplib_patched.py

Patched "rock-solid" Berlin52 benchmark for the strobe / variable-horizon 2-opt optimizer.

What's new in this patched version:
1) TSPLIB EUC_2D integer distances (same as before) -> comparable to optimum 7542.
2) O(1) delta-cost evaluation for each 2-opt move (much faster / more moves per second).
3) Optional greedy 2-opt "polish" passes at strobe-cycle boundaries (reduces variance hard).
4) Summary includes restart math: probability of hitting optimum with best-of-N restarts.

Run (Windows, single line):
  python benchmark_strobe_tsplib_patched.py --trials 20 --cycles 100000 --plot --polish

Notes:
- This script is designed to make the claim defensible: you are optimizing the *canonical*
  TSPLIB berlin52 objective (integer-rounded EUC_2D) and measuring hit rates over seeds.
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
    for k in range(n):
        total += int(D[int(path[k]), int(path[(k + 1) % n])])
    return int(total)


# ==========================================
# PART 3: Fast 2-opt delta + greedy polish
# ==========================================
def two_opt_delta(path: np.ndarray, i: int, j: int, D: np.ndarray) -> int:
    """
    O(1) delta cost for reversing path[i:j+1] in a cyclic tour.

    Removes edges:
      (i-1 -> i) and (j -> j+1)
    Adds edges:
      (i-1 -> j) and (i -> j+1)

    delta = new - old  (negative is improvement)
    """
    n = int(path.shape[0])
    if i == j:
        return 0

    a = int(path[(i - 1) % n])
    b = int(path[i])
    c = int(path[j])
    d = int(path[(j + 1) % n])

    old_cost = int(D[a, b]) + int(D[c, d])
    new_cost = int(D[a, c]) + int(D[b, d])
    return int(new_cost - old_cost)


def apply_two_opt(path: np.ndarray, i: int, j: int) -> None:
    """In-place reverse of path segment [i, j] (inclusive). Assumes 0 <= i <= j < n."""
    path[i : j + 1] = path[i : j + 1][::-1]


def greedy_two_opt_polish(
    path: np.ndarray,
    D: np.ndarray,
    current_cost: int,
    max_passes: int = 20,
    first_improvement: bool = True,
) -> Tuple[int, int]:
    """
    Greedy 2-opt local search on a cyclic tour.

    Returns: (new_cost, num_moves_applied)

    Strategy:
      - Repeatedly scan all i<j for improving moves (delta < 0).
      - Apply either the first improving move (first_improvement=True) or the best in the pass.
      - Stop when no improvement found, or max_passes reached.

    For berlin52 (n=52), this is very cheap and tends to crush variance.
    """
    n = int(path.shape[0])
    moves = 0
    cost = int(current_cost)

    for _pass in range(int(max_passes)):
        best_delta = 0
        best_i = -1
        best_j = -1

        improved = False

        for i in range(n - 1):
            # j must be > i
            for j in range(i + 1, n):
                # Skip trivial full reversal (i=0, j=n-1) since it doesn't change the tour
                if i == 0 and j == n - 1:
                    continue

                delta = two_opt_delta(path, i, j, D)
                if delta < 0:
                    if first_improvement:
                        apply_two_opt(path, i, j)
                        cost += delta
                        moves += 1
                        improved = True
                        break
                    else:
                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_j = i, j
            if first_improvement and improved:
                break

        if not first_improvement:
            if best_delta < 0:
                apply_two_opt(path, best_i, best_j)
                cost += int(best_delta)
                moves += 1
                improved = True

        if not improved:
            break

    return int(cost), int(moves)


# ==========================================
# PART 4: THE ENTANGLEMENT ENGINE (patched)
# ==========================================
@dataclass
class EntanglementOptimizer:
    D: np.ndarray
    state: np.ndarray

    current_cost: int = 0
    best_state: np.ndarray | None = None
    min_cost: int = 0

    def __post_init__(self) -> None:
        self.state = self.state.copy()
        self.current_cost = int(tsp_cost_int(self.state, self.D))
        self.best_state = self.state.copy()
        self.min_cost = int(self.current_cost)

    def mutate_variable_horizon_indices(self, n: int, horizon_ratio: float, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Pick (i, j) for 2-opt using variable horizon.

        IMPORTANT:
        - Avoid the degenerate full reversal (i=0, j=n-1). It leaves the cyclic tour unchanged
          but breaks the simple 2-opt delta formula (it would reference the same edge twice).
        - Also avoid i == j.
        """
        for _ in range(16):
            i = int(rng.integers(0, n))
            window_size = max(2, int(n * float(horizon_ratio)))
            offset = int(rng.integers(1, window_size))
            j = (i + offset) % n
            if i > j:
                i, j = j, i
            if i == j:
                continue
            if i == 0 and j == n - 1:
                continue
            return i, j
        return 1, n - 2

    def step(self, t: int, cycle_length: int, rng: np.random.Generator) -> Tuple[int, float]:
        # === THE STROBE: OSCILLATING HORIZON ===
        phase = (t % cycle_length) / float(cycle_length)
        horizon = 0.5 * (math.cos(phase * 2.0 * math.pi) + 1.0)
        horizon = max(horizon, 0.05)

        n = int(self.state.shape[0])
        i, j = self.mutate_variable_horizon_indices(n, horizon, rng)

        # Fast candidate delta
        delta = two_opt_delta(self.state, i, j, self.D)

        if delta < 0:
            apply_two_opt(self.state, i, j)
            self.current_cost += int(delta)

            if self.current_cost < self.min_cost:
                self.min_cost = int(self.current_cost)
                self.best_state = self.state.copy()
        else:
            tunneling_amplitude = horizon * 0.05
            denom = max(tunneling_amplitude * 1000.0, 1e-9)
            accept_p = math.exp(-float(delta) / denom)

            if float(rng.random()) < accept_p:
                apply_two_opt(self.state, i, j)
                self.current_cost += int(delta)

        return int(self.current_cost), float(horizon)

    def maybe_polish(self, max_passes: int, first_improvement: bool = True) -> Tuple[int, int]:
        """
        Greedy 2-opt polish the *current* state. Updates best if improved.
        Returns (new_cost, moves_applied).
        """
        new_cost, moves = greedy_two_opt_polish(
            self.state, self.D, self.current_cost, max_passes=max_passes, first_improvement=first_improvement
        )
        self.current_cost = int(new_cost)
        if self.current_cost < self.min_cost:
            self.min_cost = int(self.current_cost)
            self.best_state = self.state.copy()
        return int(new_cost), int(moves)

    def polish_best(self, max_passes: int, first_improvement: bool = True) -> Tuple[int, int]:
        """
        Greedy 2-opt polish the *best_state* (post-run cleanup).
        """
        if self.best_state is None:
            self.best_state = self.state.copy()
        best_cost = int(tsp_cost_int(self.best_state, self.D))
        new_cost, moves = greedy_two_opt_polish(
            self.best_state, self.D, best_cost, max_passes=max_passes, first_improvement=first_improvement
        )
        if new_cost < self.min_cost:
            self.min_cost = int(new_cost)
        return int(new_cost), int(moves)


# ==========================================
# PART 5: experiment runner
# ==========================================
def run_one_trial(
    cities: np.ndarray,
    D: np.ndarray,
    cycles: int,
    seed: int,
    cycle_length: int,
    verbose: bool,
    do_polish: bool,
    polish_every_cycles: int,
    polish_max_passes: int,
    polish_first_improvement: bool,
    sanity_every: int,
) -> Tuple[int, np.ndarray, List[int], List[float]]:
    rng = np.random.default_rng(seed)

    n = int(cities.shape[0])
    initial_path = np.arange(n, dtype=np.int32)
    rng.shuffle(initial_path)

    opt = EntanglementOptimizer(D=D, state=initial_path)

    cost_history: List[int] = []
    horizon_history: List[float] = []

    lock_printed = False

    for t in range(int(cycles)):
        cost, horizon = opt.step(t, cycle_length=cycle_length, rng=rng)
        cost_history.append(cost)
        horizon_history.append(horizon)

        # Optional polish at strobe boundaries (every N strobe cycles)
        if do_polish and (t > 0) and ((t + 1) % cycle_length == 0):
            strobe_idx = (t + 1) // cycle_length
            if polish_every_cycles <= 1 or (strobe_idx % polish_every_cycles == 0):
                polished_cost, moves = opt.maybe_polish(max_passes=polish_max_passes, first_improvement=polish_first_improvement)
                if verbose:
                    print(f"     🔧 polish @t={t+1}: cost={polished_cost} moves={moves}")

        # Optional sanity check: recompute true cost to detect drift
        if sanity_every and (t % sanity_every == 0):
            true_cost = tsp_cost_int(opt.state, opt.D)
            if true_cost != opt.current_cost:
                raise RuntimeError(f"Sanity check failed at t={t}: current_cost={opt.current_cost} true_cost={true_cost}")

        if verbose and (t % max(1, cycles // 10) == 0):
            print(f"   Cycle {t:6d}: cost={cost} horizon={horizon:.3f} best={opt.min_cost}")

        if (not lock_printed) and opt.min_cost <= OPTIMAL_SCORE + 10:
            print(f"   ⚡ near-opt region reached at cycle {t}: best={opt.min_cost}")
            lock_printed = True

    # Post-run polish of best_state for "rock-solid" reporting (optional but recommended)
    if do_polish:
        opt.polish_best(max_passes=polish_max_passes, first_improvement=polish_first_improvement)

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


def restart_success_table(p_hit: float, ks: List[int]) -> str:
    """
    Probability of at least one success with k independent restarts:
      p_success(k) = 1 - (1 - p_hit)^k
    """
    lines = []
    for k in ks:
        p = 1.0 - (1.0 - p_hit) ** float(k)
        lines.append(f"best-of-{k}: {p*100:5.1f}%")
    return "  ".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patched Berlin52 benchmark (TSPLIB EUC_2D) with delta-cost and polish.")
    parser.add_argument("--cycles", type=int, default=100000, help="optimization steps per trial")
    parser.add_argument("--trials", type=int, default=10, help="number of independent random seeds")
    parser.add_argument("--seed", type=int, default=0, help="base seed (trial k uses seed+ k)")
    parser.add_argument("--cycle-length", type=int, default=2000, help="strobe cycle length")
    parser.add_argument("--plot", action="store_true", help="plot best-trial path and dynamics")
    parser.add_argument("--verbose", action="store_true", help="print periodic updates within each trial")

    # Patch knobs
    parser.add_argument("--polish", action="store_true", help="enable greedy 2-opt polish at strobe boundaries + end")
    parser.add_argument("--polish-every-cycles", type=int, default=1, help="polish every N strobe cycles (1 = every cycle)")
    parser.add_argument("--polish-max-passes", type=int, default=20, help="max polish passes when triggered")
    parser.add_argument("--polish-best-improvement", action="store_true", help="use best-improvement per pass (slower, sometimes better). Default is first-improvement.")
    parser.add_argument("--sanity-every", type=int, default=0, help="if >0, recompute exact tour cost every N steps to sanity-check delta updates")
    args = parser.parse_args()

    cities = BERLIN52_COORDS
    D = make_distance_matrix(cities)
    n = int(cities.shape[0])

    print("🌌 BOOTING ENTANGLEMENT ENGINE (Variable Horizon) — TSPLIB EUC_2D (PATCHED)")
    print(f"   Instance: berlin52 (n={n})")
    print(f"   TSPLIB optimum: {OPTIMAL_SCORE}")
    print(f"   Trials: {args.trials}   Cycles/trial: {args.cycles}   Base seed: {args.seed}")
    print(f"   Delta-cost: ON")
    print(f"   Polish: {'ON' if args.polish else 'OFF'}", end="")
    if args.polish:
        mode = "best-improvement" if args.polish_best_improvement else "first-improvement"
        print(f"  (every {args.polish_every_cycles} strobe cycles, max_passes={args.polish_max_passes}, mode={mode})")
    else:
        print("")
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
            do_polish=args.polish,
            polish_every_cycles=max(1, int(args.polish_every_cycles)),
            polish_max_passes=max(1, int(args.polish_max_passes)),
            polish_first_improvement=(not args.polish_best_improvement),
            sanity_every=max(0, int(args.sanity_every)),
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

    within_0 = sum(1 for s in all_scores if s == OPTIMAL_SCORE)
    within_2 = sum(1 for s in all_scores if s <= OPTIMAL_SCORE + 2)
    within_5 = sum(1 for s in all_scores if s <= OPTIMAL_SCORE + 5)
    within_10 = sum(1 for s in all_scores if s <= OPTIMAL_SCORE + 10)

    p_hit = within_0 / float(args.trials) if args.trials > 0 else 0.0

    print("\n" + "=" * 60)
    print("✅ SUMMARY (TSPLIB-comparable, integer objective)")
    print(f"   {summarize(all_scores)}")
    print(f"   best gap: {best_gap:.4f}%  (best={best_overall_score}, optimum={OPTIMAL_SCORE})")
    print(f"   hit-rate: optimal={within_0}/{args.trials}  ≤+2={within_2}/{args.trials}  ≤+5={within_5}/{args.trials}  ≤+10={within_10}/{args.trials}")
    if args.trials > 0:
        print(f"   p(hit optimal in one run): {p_hit*100:.1f}%")
        print(f"   restart math: {restart_success_table(p_hit, [2, 3, 5, 10])}")
    print(f"   elapsed: {elapsed:.2f}s")
    print("=" * 60)

    if args.plot:
        best_coords = cities[best_overall_path]
        plot_coords = np.vstack([best_coords, best_coords[0]])

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(plot_coords[:, 0], plot_coords[:, 1], "-o", markersize=4)
        plt.title(f"Best Tour (TSPLIB EUC_2D): {best_overall_score}")

        plt.subplot(1, 2, 2)
        plt.plot(best_cost_history, linewidth=0.8, label="Tour length (integer)")
        scaled_h = np.array(best_horizon_history, dtype=np.float64) * (max(best_cost_history) - min(best_cost_history)) + min(best_cost_history)
        plt.plot(scaled_h, alpha=0.35, linewidth=0.6, label="Horizon (scaled)")
        plt.axhline(y=OPTIMAL_SCORE, linestyle="--", label="TSPLIB optimum (7542)")
        plt.title("Dynamics (best trial)")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
