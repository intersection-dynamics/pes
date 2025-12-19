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
])

# ==========================================
# PART 2: THE ENTANGLEMENT ENGINE
# ==========================================

class EntanglementOptimizer:
    def __init__(self, cost_function, initial_state):
        self.cost_func = cost_function
        self.state = initial_state.copy()
        
        self.current_cost = self.cost_func(self.state)
        self.best_state = self.state.copy()
        self.min_cost = self.current_cost

    def mutate_variable_horizon(self, path, horizon_ratio):
        """
        The Quantum Mutation.
        Instead of random selection, the 'Horizon' dictates the reach.
        horizon_ratio (0.0 to 1.0): 
            - 1.0 = Global Entanglement (Connect any two nodes)
            - 0.1 = Local Decoherence (Connect only neighbors)
        """
        n = len(path)
        
        # 1. Pick first node (Anchor)
        i = np.random.randint(0, n)
        
        # 2. Determine the Horizon Window
        # How far away can the second node be?
        window_size = max(2, int(n * horizon_ratio))
        
        # 3. Pick second node WITHIN the Horizon
        # We look 'window_size' steps ahead or behind (wrapping around)
        offset = np.random.randint(1, window_size)
        j = (i + offset) % n
        
        # Ensure i < j for slicing simplicity, handle wrap
        if i > j: i, j = j, i
            
        # 4. Perform Topological Flip (2-Opt)
        new_path = path.copy()
        new_path[i:j+1] = new_path[i:j+1][::-1]
        
        return new_path

    def step(self, t, max_cycles):
        # === THE STROBE: OSCILLATING HORIZON ===
        # We don't use temperature. We use Interaction Range.
        # We breathe: Expansion (Global Search) -> Contraction (Local Polish)
        
        cycle_length = 2000
        phase = (t % cycle_length) / cycle_length
        
        # Horizon Logic:
        # High Strobe: Horizon = 1.0 (Full Entanglement)
        # Low Strobe:  Horizon = 0.05 (Local Stability)
        # We use a Cosine wave to smooth the transition
        horizon = 0.5 * (np.cos(phase * 2 * np.pi) + 1)
        # Clip it so it never goes to 0 (must allow neighbor swaps)
        horizon = max(horizon, 0.05) 
        
        # === DYNAMICS ===
        # Create a candidate based on current Horizon
        candidate_state = self.mutate_variable_horizon(self.state, horizon)
        candidate_cost = self.cost_func(candidate_state)
        
        delta_E = candidate_cost - self.current_cost
        
        # === STABILITY CHECK (Tunneling) ===
        # If Simplicity increases (Lower Cost), we ALWAYS collapse.
        if delta_E < 0:
            self.state = candidate_state
            self.current_cost = candidate_cost
            
            # Record Record
            if self.current_cost < self.min_cost:
                self.min_cost = self.current_cost
                self.best_state = self.state.copy()
                if self.min_cost < 7600:
                    print(f"   ⚡ GEOMETRY LOCKED at cycle {t}: {self.min_cost:.2f}")

        # If Simplicity decreases (Higher Cost), we check Stability.
        else:
            # We allow tunneling ONLY when Horizon is High (Quantum Phase).
            # When Horizon is Low (Classical Phase), the system resists noise.
            
            # This replaces "Temperature" with "Entanglement Strength"
            tunneling_amplitude = horizon * 0.05 # Tunable constant
            
            # Standard acceptance logic using Horizon as the driver
            if np.random.rand() < np.exp(-delta_E / (tunneling_amplitude * 1000)):
                self.state = candidate_state
                self.current_cost = candidate_cost

        return self.current_cost, horizon

# ==========================================
# EXECUTION
# ==========================================

def tsp_cost(path, cities):
    dist = 0.0
    for i in range(len(path)):
        c1 = cities[path[i]]
        c2 = cities[path[(i+1) % len(path)]]
        dist += np.linalg.norm(c1 - c2)
    return dist

if __name__ == "__main__":
    cities = BERLIN52_COORDS
    num_cities = len(cities)
    OPTIMAL_SCORE = 7542.0
    
    print(f"🌌 BOOTING ENTANGLEMENT ENGINE (Variable Horizon)...")
    print(f"   Target: {OPTIMAL_SCORE}")
    
    initial_path = np.arange(num_cities)
    np.random.shuffle(initial_path)
    
    opt = EntanglementOptimizer(
        cost_function=lambda p: tsp_cost(p, cities),
        initial_state=initial_path
    )
    
    CYCLES = 100000
    print(f"   Running {CYCLES} cycles...")
    
    cost_history = []
    horizon_history = []
    
    for t in range(CYCLES):
        cost, horizon = opt.step(t, CYCLES)
        cost_history.append(cost)
        horizon_history.append(horizon * 10000) # Scale for plotting
        
        if t % 10000 == 0:
            print(f"   Cycle {t}: Cost={cost:.1f} Horizon={horizon:.2f}")

    final_score = tsp_cost(opt.best_state, cities)
    gap = ((final_score - OPTIMAL_SCORE) / OPTIMAL_SCORE) * 100
    
    print("\n" + "="*40)
    print(f"🏁 RESULTS")
    print(f"   Strobe Score: {final_score:.2f}")
    print(f"   World Record: {OPTIMAL_SCORE}")
    print(f"   Gap: {gap:.4f}%")
    print("="*40)

    # Visualization
    plt.figure(figsize=(14, 6))
    
    # Path
    plt.subplot(1, 2, 1)
    best_coords = cities[opt.best_state]
    plot_coords = np.vstack([best_coords, best_coords[0]])
    plt.plot(plot_coords[:,0], plot_coords[:,1], 'c-o', markersize=4)
    plt.title(f"Factorized Geometry (Score: {final_score:.0f})")
    
    # Dynamics
    plt.subplot(1, 2, 2)
    plt.plot(cost_history, 'r-', linewidth=0.5, label='Locality Cost')
    plt.plot(horizon_history, 'b-', linewidth=0.1, alpha=0.3, label='Entanglement Horizon')
    plt.axhline(y=OPTIMAL_SCORE, color='g', linestyle='--')
    plt.title("Tension: Simplicity vs Stability")
    plt.legend()
    
    plt.show()