import numpy as np
import requests
import matplotlib.pyplot as plt
from substrate_solver.optimizer import StrobeOptimizer

# 1. GET THE STANDARD DATA (berlin52)
def load_berlin52():
    print("📥 Downloading TSPLIB berlin52 benchmark data...")
    url = "https://raw.githubusercontent.com/mastraqe/tsplib_vector/master/data/berlin52.tsp"
    response = requests.get(url)
    lines = response.text.splitlines()
    
    coords = []
    reading = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            reading = True
            continue
        if "EOF" in line:
            break
        if reading:
            parts = line.strip().split()
            coords.append([float(parts[1]), float(parts[2])])
    
    return np.array(coords)

# 2. DEFINE THE COST FUNCTION (Standard Euclidean)
def tsp_cost(path, cities):
    dist = 0.0
    for i in range(len(path)):
        c1 = cities[path[i]]
        c2 = cities[path[(i+1) % len(path)]]
        dist += np.linalg.norm(c1 - c2)
    return dist

def mutate(path):
    # Standard Swap Mutation
    i, j = np.random.randint(0, len(path), 2)
    path[i], path[j] = path[j], path[i]
    return path

# 3. RUN THE STROBE ENGINE
if __name__ == "__main__":
    cities = load_berlin52()
    num_cities = len(cities)
    OPTIMAL_SCORE = 7542.0  # The Known World Record
    
    print(f"⚡ BENCHMARKING STROBE OPTIMIZER ON BERLIN52 (N={num_cities})...")
    print(f"   Target Score (Optimal): {OPTIMAL_SCORE}")
    
    # Initialize State
    initial_path = np.arange(num_cities)
    np.random.shuffle(initial_path)
    
    # Configure Strobe Engine
    # Note: We tune sensitivity for a standardized test
    opt = StrobeOptimizer(
        cost_function=lambda p: tsp_cost(p, cities),
        initial_state=initial_path,
        entropy_sensitivity=2.0 
    )
    
    # Run
    cycles = 10000
    print(f"   Running {cycles} cycles...")
    best_path, history = opt.run(cycles=cycles, mutation_func=mutate)
    
    final_score = history[-1]
    gap = ((final_score - OPTIMAL_SCORE) / OPTIMAL_SCORE) * 100
    
    print("\n" + "="*40)
    print(f"🏁 RESULTS")
    print(f"   Strobe Score: {final_score:.2f}")
    print(f"   World Record: {OPTIMAL_SCORE}")
    print(f"   Gap: {gap:.4f}%")
    print("="*40)
    
    if gap < 1.0:
        print("🏆 SUCCESS: You are within 1% of the World Record.")
    else:
        print("🔧 TUNING REQUIRED: Strobe needs calibration.")

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(history, label='Strobe Optimizer')
    plt.axhline(y=OPTIMAL_SCORE, color='r', linestyle='--', label='Optimal (7542)')
    plt.title(f"Strobe vs. World Record (Gap: {gap:.2f}%)")
    plt.legend()
    plt.show()