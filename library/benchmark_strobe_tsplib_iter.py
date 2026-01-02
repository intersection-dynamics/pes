#!/usr/bin/env python3
"""
benchmark_strobe_tsplib_iter.py

Single-file, iterate-here benchmark for Berlin52 using TSPLIB EUC_2D (integer) metric.

Core features (all in this one script):
- TSPLIB EUC_2D integer distances (comparable to optimum 7542)
- Variable-horizon (strobe) 2-opt with O(1) delta-cost updates
- Optional greedy 2-opt polish (cycle-boundary + post-run)
- Optional sanity check (recompute exact cost every N steps)
- Optional stagnation "kick" (force horizon high for a short burst when stuck)
- Optional in-run perturbation (big 2-opt) after hopeless polish plateaus

Windows (single-line) examples:
  python benchmark_strobe_tsplib_iter.py --trials 20 --cycles 100000 --plot
  python benchmark_strobe_tsplib_iter.py --trials 20 --cycles 100000 --polish --plot
  python benchmark_strobe_tsplib_iter.py --trials 20 --cycles 100000 --polish --kick --plot
  python benchmark_strobe_tsplib_iter.py --trials 20 --cycles 100000 --polish --kick --perturb --plot
  python benchmark_strobe_tsplib_iter.py --trials 10 --cycles 100000 --sanity-every 1000

Notes:
- Berlin52 optimum (TSPLIB EUC_2D) is 7542.
- This is a benchmark harness + optimizer in one file for iteration.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# DATA: Berlin52 (TSPLIB)
# ==========================================
BERLIN52_COORDS = np.array(
    [
        [565.0, 575.0],
        [25.0, 185.0],
        [345.0, 750.0],
        [945.0, 685.0],
        [845.0, 655.0],
        [880.0, 660.0],
        [25.0, 230.0],
        [525.0, 1000.0],
        [580.0, 1175.0],
        [650.0, 1130.0],
        [1605.0, 620.0],
        [1220.0, 580.0],
        [1465.0, 200.0],
        [1530.0, 5.0],
        [845.0, 680.0],
        [725.0, 370.0],
        [145.0, 665.0],
        [415.0, 635.0],
        [510.0, 875.0],
        [560.0, 365.0],
        [300.0, 465.0],
        [520.0, 585.0],
        [480.0, 415.0],
        [835.0, 625.0],
        [975.0, 580.0],
        [1215.0, 245.0],
        [1320.0, 315.0],
        [1250.0, 400.0],
        [660.0, 180.0],
        [410.0, 250.0],
        [420.0, 555.0],
        [575.0, 665.0],
        [1150.0, 1160.0],
        [700.0, 580.0],
        [685.0, 595.0],
        [685.0, 610.0],
        [770.0, 610.0],
        [795.0, 645.0],
        [720.0, 635.0],
        [760.0, 650.0],
        [475.0, 960.0],
        [95.0, 260.0],
        [875.0, 920.0],
        [700.0, 500.0],
        [555.0, 815.0],
        [830.0, 485.0],
        [1170.0, 65.0],
        [830.0, 610.0],
        [605.0, 625.0],
        [595.0, 360.0],
        [1340.0, 725.0],
        [1740.0, 245.0],
    ],
    dtype=np.float64,
)

OPTIMAL_SCORE = 7542  # berlin52 TSPLIB EUC_2D integer optimum


# ==========================================
# TSPLIB EUC_2D distances
# ==========================================
def euc_2d_tsplib(a: np.ndarray, b: np.ndarray) -> int:
    """TSPLIB EUC_2D: int(sqrt(dx^2+dy^2) + 0.5)"""
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def make_distance_matrix(cities: np.ndarray) -> np.ndarray:
    n = int(cities.shape[0])
    D = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            dij = euc_2d_tsplib(cities[i], cities[j])
            D[i, j] = dij
            D[j, i] = dij
    return D


def tsp_cost_int(path: np.ndarray, D: np.ndarray) -> int:
    n = int(path.shape[0])
    total = 0
    for k in range(n):
        total += int(D[int(path[k]), int(path[(k + 1) % n])])
    return int(total)


# ==========================================
# 2-opt delta + apply
# ==========================================
def two_opt_delta(path: np.ndarray, i: int, j: int, D: np.ndarray) -> int:
    """
    O(1) delta for reversing segment [i..j] in a cyclic tour.

    Remove edges: (i-1 -> i) and (j -> j+1)
    Add edges:    (i-1 -> j) and (i -> j+1)
    """
    n = int(path.shape[0])
    a = int(path[(i - 1) % n])
    b = int(path[i])
    c = int(path[j])
    d = int(path[(j + 1) % n])
    old_cost = int(D[a, b]) + int(D[c, d])
    new_cost = int(D[a, c]) + int(D[b, d])
    return int(new_cost - old_cost)


def apply_two_opt(path: np.ndarray, i: int, j: int) -> None:
    path[i : j + 1] = path[i : j + 1][::-1]


# ==========================================
# Greedy 2-opt polish
# ==========================================
def greedy_two_opt_polish(
    path: np.ndarray,
    D: np.ndarray,
    current_cost: int,
    max_passes: int = 20,
    first_improvement: bool = True,
) -> Tuple[int, int]:
    """
    Greedy 2-opt local search on the cyclic tour in-place.

    Returns: (new_cost, moves_applied)
    """
    n = int(path.shape[0])
    cost = int(current_cost)
    moves = 0

    for _ in range(int(max_passes)):
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1

        for i in range(n - 1):
            for j in range(i + 1, n):
                # skip degenerate full reversal
                if i == 0 and j == n - 1:
                    continue

                delta = two_opt_delta(path, i, j, D)
                if delta < 0:
                    if first_improvement:
                        apply_two_opt(path, i, j)
                        cost += int(delta)
                        moves += 1
                        improved = True
                        break
                    else:
                        if delta < best_delta:
                            best_delta = int(delta)
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
# Optimizer
# ==========================================
@dataclass
class EntanglementOptimizer:
    D: np.ndarray
    state: np.ndarray

    current_cost: int = 0
    best_state: np.ndarray | None = None
    min_cost: int = 0

    # Stagnation tracking
    best_improve_step: int = 0

    def __post_init__(self) -> None:
        self.state = self.state.copy()
        self.current_cost = int(tsp_cost_int(self.state, self.D))
        self.best_state = self.state.copy()
        self.min_cost = int(self.current_cost)
        self.best_improve_step = 0

    def mutate_variable_horizon_indices(
        self, n: int, horizon_ratio: float, rng: np.random.Generator
    ) -> Tuple[int, int]:
        """
        Pick (i, j) for 2-opt using variable horizon.

        IMPORTANT: avoid degenerate (i=0, j=n-1) and i==j.
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

    def step(
        self,
        t: int,
        cycle_length: int,
        rng: np.random.Generator,
        kick_active: bool,
        kick_horizon: float,
    ) -> Tuple[int, float]:
        # Base strobe horizon
        phase = (t % cycle_length) / float(cycle_length)
        horizon = 0.5 * (math.cos(phase * 2.0 * math.pi) + 1.0)
        horizon = max(horizon, 0.05)

        # Optional kick overrides horizon upwards
        if kick_active:
            horizon = max(horizon, float(kick_horizon))
            horizon = min(horizon, 1.0)

        n = int(self.state.shape[0])
        i, j = self.mutate_variable_horizon_indices(n, horizon, rng)

        delta = two_opt_delta(self.state, i, j, self.D)

        if delta < 0:
            apply_two_opt(self.state, i, j)
            self.current_cost += int(delta)

            if self.current_cost < self.min_cost:
                self.min_cost = int(self.current_cost)
                self.best_state = self.state.copy()
                self.best_improve_step = int(t)

        else:
            tunneling_amplitude = horizon * 0.05
            denom = max(tunneling_amplitude * 1000.0, 1e-9)
            accept_p = math.exp(-float(delta) / denom)

            if float(rng.random()) < accept_p:
                apply_two_opt(self.state, i, j)
                self.current_cost += int(delta)

        return int(self.current_cost), float(horizon)

    def maybe_polish(self, max_passes: int, first_improvement: bool) -> Tuple[int, int]:
        new_cost, moves = greedy_two_opt_polish(
            self.state,
            self.D,
            self.current_cost,
            max_passes=max_passes,
            first_improvement=first_improvement,
        )
        self.current_cost = int(new_cost)
        if self.current_cost < self.min_cost:
            self.min_cost = int(self.current_cost)
            self.best_state = self.state.copy()
            # polishing improvement counts as improvement "now"
            # (caller can pass t if they want; we leave as-is)
        return int(new_cost), int(moves)

    def polish_best(self, max_passes: int, first_improvement: bool) -> Tuple[int, int]:
        if self.best_state is None:
            self.best_state = self.state.copy()
        best_cost = int(tsp_cost_int(self.best_state, self.D))
        new_cost, moves = greedy_two_opt_polish(
            self.best_state,
            self.D,
            best_cost,
            max_passes=max_passes,
            first_improvement=first_improvement,
        )
        if new_cost < self.min_cost:
            self.min_cost = int(new_cost)
        return int(new_cost), int(moves)

    def perturb_big_2opt(self, rng: np.random.Generator, min_span: int = 10) -> None:
        """
        One deliberate big reversal to escape a basin.
        """
        n = int(self.state.shape[0])
        # pick i < j with a minimum span (avoid tiny local moves)
        for _ in range(64):
            i = int(rng.integers(0, n - 1))
            j = int(rng.integers(i + 1, n))
            if i == 0 and j == n - 1:
                continue
            if (j - i) >= min_span:
                delta = two_opt_delta(self.state, i, j, self.D)
                apply_two_opt(self.state, i, j)
                self.current_cost += int(delta)
                return
        # fallback
        i, j = 1, n - 2
        delta = two_opt_delta(self.state, i, j, self.D)
        apply_two_opt(self.state, i, j)
        self.current_cost += int(delta)


# ==========================================
# Runner + reporting
# ==========================================
def summarize(scores: List[int]) -> str:
    arr = np.array(scores, dtype=np.int32)
    return (
        f"best={int(arr.min())}  mean={float(arr.mean()):.2f}  "
        f"median={float(np.median(arr)):.2f}  std={float(arr.std(ddof=0)):.2f}"
    )


def restart_success_table(p_hit: float, ks: List[int]) -> str:
    parts = []
    for k in ks:
        p = 1.0 - (1.0 - p_hit) ** float(k)
        parts.append(f"best-of-{k}: {p*100:5.1f}%")
    return "  ".join(parts)


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
    # kick
    do_kick: bool,
    kick_stagnation_steps: int,
    kick_steps: int,
    kick_horizon: float,
    # perturb
    do_perturb: bool,
    perturb_after_polish_no_moves: int,
    perturb_if_best_above: int,
) -> Tuple[int, np.ndarray, List[int], List[float]]:
    rng = np.random.default_rng(seed)

    n = int(cities.shape[0])
    initial_path = np.arange(n, dtype=np.int32)
    rng.shuffle(initial_path)

    opt = EntanglementOptimizer(D=D, state=initial_path)

    cost_history: List[int] = []
    horizon_history: List[float] = []

    lock_printed = False

    kick_remaining = 0
    no_move_polish_streak = 0

    for t in range(int(cycles)):
        # Kick logic: if best hasn't improved for a while, temporarily raise horizon
        if do_kick and kick_remaining <= 0:
            if (t - opt.best_improve_step) >= kick_stagnation_steps:
                kick_remaining = int(kick_steps)

        kick_active = kick_remaining > 0
        cost, horizon = opt.step(
            t=t,
            cycle_length=cycle_length,
            rng=rng,
            kick_active=kick_active,
            kick_horizon=kick_horizon,
        )
        if kick_remaining > 0:
            kick_remaining -= 1

        cost_history.append(cost)
        horizon_history.append(horizon)

        # Optional sanity check (detect delta drift)
        if sanity_every and (t % sanity_every == 0):
            true_cost = tsp_cost_int(opt.state, opt.D)
            if true_cost != opt.current_cost:
                raise RuntimeError(
                    f"Sanity check failed at t={t}: current_cost={opt.current_cost} true_cost={true_cost}"
                )

        # Optional polish at strobe boundaries
        if do_polish and (t > 0) and ((t + 1) % cycle_length == 0):
            strobe_idx = (t + 1) // cycle_length
            if polish_every_cycles <= 1 or (strobe_idx % polish_every_cycles == 0):
                polished_cost, moves = opt.maybe_polish(
                    max_passes=polish_max_passes,
                    first_improvement=polish_first_improvement,
                )
                if moves == 0:
                    no_move_polish_streak += 1
                else:
                    no_move_polish_streak = 0

                if verbose:
                    print(
                        f"     🔧 polish @t={t+1}: cost={polished_cost} moves={moves} no-move-streak={no_move_polish_streak}"
                    )

                # Optional perturbation: if polish keeps doing nothing AND we're still far from optimum
                if do_perturb and no_move_polish_streak >= perturb_after_polish_no_moves:
                    if opt.min_cost >= (OPTIMAL_SCORE + perturb_if_best_above):
                        opt.perturb_big_2opt(rng=rng, min_span=10)
                        # reset streak after perturb
                        no_move_polish_streak = 0
                        # also reset stagnation counter so kick doesn't spam instantly
                        opt.best_improve_step = t
                        if verbose:
                            print(f"     💥 perturb triggered @t={t+1}: current_cost={opt.current_cost} best={opt.min_cost}")

        if verbose and (t % max(1, cycles // 10) == 0):
            print(f"   Cycle {t:6d}: cost={cost} horizon={horizon:.3f} best={opt.min_cost}")

        if (not lock_printed) and opt.min_cost <= OPTIMAL_SCORE + 10:
            print(f"   ⚡ near-opt region reached at cycle {t}: best={opt.min_cost}")
            lock_printed = True

    # Post-run polish of best tour for reporting
    if do_polish:
        opt.polish_best(max_passes=polish_max_passes, first_improvement=polish_first_improvement)

    best_score = int(opt.min_cost)
    best_path = opt.best_state.copy() if opt.best_state is not None else opt.state.copy()
    return best_score, best_path, cost_history, horizon_history


def main() -> None:
    p = argparse.ArgumentParser(description="Berlin52 TSPLIB EUC_2D strobe optimizer (single-file iteration script).")
    p.add_argument("--cycles", type=int, default=100000, help="steps per trial")
    p.add_argument("--trials", type=int, default=10, help="number of trials")
    p.add_argument("--seed", type=int, default=0, help="base seed (trial k uses seed+k)")
    p.add_argument("--cycle-length", type=int, default=2000, help="strobe cycle length")
    p.add_argument("--plot", action="store_true", help="plot best-trial path + dynamics")
    p.add_argument("--verbose", action="store_true", help="verbose progress printing")
    p.add_argument("--sanity-every", type=int, default=0, help="if >0, recompute exact cost every N steps")

    # polish knobs
    p.add_argument("--polish", action="store_true", help="enable greedy 2-opt polish at strobe boundaries + end")
    p.add_argument("--polish-every-cycles", type=int, default=1, help="polish every N strobe cycles (1=every)")
    p.add_argument("--polish-max-passes", type=int, default=20, help="max polish passes per trigger")
    p.add_argument(
        "--polish-best-improvement",
        action="store_true",
        help="use best-improvement per pass (slower). default=first-improvement",
    )

    # kick knobs
    p.add_argument("--kick", action="store_true", help="enable stagnation kick (force horizon high briefly)")
    p.add_argument("--kick-stagnation-steps", type=int, default=8000, help="steps without best-improve before kicking")
    p.add_argument("--kick-steps", type=int, default=800, help="kick duration (steps)")
    p.add_argument("--kick-horizon", type=float, default=1.0, help="horizon during kick (0..1)")

    # perturb knobs
    p.add_argument("--perturb", action="store_true", help="enable perturbation after repeated no-move polish plateaus")
    p.add_argument("--perturb-after-no-move-polish", type=int, default=3, help="trigger after N consecutive no-move polishes")
    p.add_argument("--perturb-if-best-above", type=int, default=50, help="only perturb if best is still >= optimum + this")

    args = p.parse_args()

    cities = BERLIN52_COORDS
    D = make_distance_matrix(cities)
    n = int(cities.shape[0])

    print("🌌 BOOTING ENTANGLEMENT ENGINE (Variable Horizon) — TSPLIB EUC_2D (ITER)")
    print(f"   Instance: berlin52 (n={n})")
    print(f"   TSPLIB optimum: {OPTIMAL_SCORE}")
    print(f"   Trials: {args.trials}   Cycles/trial: {args.cycles}   Base seed: {args.seed}")
    print("   Delta-cost: ON")
    print(f"   Polish: {'ON' if args.polish else 'OFF'}")
    print(f"   Kick:   {'ON' if args.kick else 'OFF'}")
    print(f"   Perturb:{'ON' if args.perturb else 'OFF'}")
    if args.sanity_every:
        print(f"   Sanity: every {args.sanity_every} steps")
    print("")

    scores: List[int] = []
    best_overall_score = None
    best_overall_path = None
    best_cost_hist: List[int] = []
    best_horiz_hist: List[float] = []

    t0 = time.time()
    for k in range(int(args.trials)):
        seed = int(args.seed + k)
        print(f"[Trial {k+1:02d}/{args.trials}] seed={seed}")

        score, path, cost_hist, horiz_hist = run_one_trial(
            cities=cities,
            D=D,
            cycles=int(args.cycles),
            seed=seed,
            cycle_length=int(args.cycle_length),
            verbose=bool(args.verbose),
            do_polish=bool(args.polish),
            polish_every_cycles=max(1, int(args.polish_every_cycles)),
            polish_max_passes=max(1, int(args.polish_max_passes)),
            polish_first_improvement=(not bool(args.polish_best_improvement)),
            sanity_every=max(0, int(args.sanity_every)),
            do_kick=bool(args.kick),
            kick_stagnation_steps=max(1, int(args.kick_stagnation_steps)),
            kick_steps=max(1, int(args.kick_steps)),
            kick_horizon=float(args.kick_horizon),
            do_perturb=bool(args.perturb),
            perturb_after_polish_no_moves=max(1, int(args.perturb_after_no_move_polish)),
            perturb_if_best_above=max(0, int(args.perturb_if_best_above)),
        )

        gap_pct = ((score - OPTIMAL_SCORE) / OPTIMAL_SCORE) * 100.0
        print(f"   🏁 best={score}   gap={gap_pct:.4f}%")

        scores.append(int(score))
        if best_overall_score is None or score < best_overall_score:
            best_overall_score = int(score)
            best_overall_path = path.copy()
            best_cost_hist = list(cost_hist)
            best_horiz_hist = list(horiz_hist)

    elapsed = time.time() - t0
    assert best_overall_score is not None and best_overall_path is not None

    within_0 = sum(1 for s in scores if s == OPTIMAL_SCORE)
    within_2 = sum(1 for s in scores if s <= OPTIMAL_SCORE + 2)
    within_5 = sum(1 for s in scores if s <= OPTIMAL_SCORE + 5)
    within_10 = sum(1 for s in scores if s <= OPTIMAL_SCORE + 10)
    p_hit = within_0 / float(len(scores)) if scores else 0.0
    best_gap = ((best_overall_score - OPTIMAL_SCORE) / OPTIMAL_SCORE) * 100.0

    print("\n" + "=" * 60)
    print("✅ SUMMARY (TSPLIB-comparable, integer objective)")
    print(f"   {summarize(scores)}")
    print(f"   best gap: {best_gap:.4f}%  (best={best_overall_score}, optimum={OPTIMAL_SCORE})")
    print(
        f"   hit-rate: optimal={within_0}/{len(scores)}  ≤+2={within_2}/{len(scores)}  ≤+5={within_5}/{len(scores)}  ≤+10={within_10}/{len(scores)}"
    )
    if len(scores) > 0:
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
        plt.plot(best_cost_hist, linewidth=0.8, label="Tour length (integer)")
        scaled_h = np.array(best_horiz_hist, dtype=np.float64) * (max(best_cost_hist) - min(best_cost_hist)) + min(
            best_cost_hist
        )
        plt.plot(scaled_h, alpha=0.35, linewidth=0.6, label="Horizon (scaled)")
        plt.axhline(y=OPTIMAL_SCORE, linestyle="--", label="TSPLIB optimum (7542)")
        plt.title("Dynamics (best trial)")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
