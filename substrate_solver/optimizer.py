import numpy as np

class StrobeOptimizer:
    """
    The Strobe Optimization Engine.
    
    Based on the 'Emergent Physics' framework.
    Principle: The optimal solution to a combinatorial problem corresponds to the
    state that minimizes the 'Locality Cost' (Signaling Capacity) of the system.
    
    Mechanic:
    1. Entropy Build-up: Introduce non-local fluctuations (noise).
    2. Strobe: Check if fluctuations reveal a lower-energy geometry.
    3. Pop: Collapse the system state when a new stable manifold is found.
    """
    
    def __init__(self, cost_function, initial_state, entropy_sensitivity=0.1):
        """
        :param cost_function: A function f(state) -> float (The 'Energy' to minimize)
        :param initial_state: The starting configuration (e.g., a random list of cities)
        :param entropy_sensitivity: How likely the system is to accept bad moves (Temperature)
        """
        self.cost_func = cost_function
        self.state = initial_state
        self.sensitivity = entropy_sensitivity
        
        # Initial Physics Check
        self.current_cost = self.cost_func(self.state)
        self.best_state = self.state.copy()
        self.min_cost = self.current_cost

    def step(self, t, mutation_func):
        """
        Executes one 'Strobe Cycle'.
        :param t: Current time step (used for cooling/decay)
        :param mutation_func: Function that returns a slightly modified state
        """
        # 1. ENTROPY BUILD (The Fluctuation)
        # Create a 'virtual particle' pair (a candidate state)
        candidate_state = mutation_func(self.state.copy())
        candidate_cost = self.cost_func(candidate_state)
        
        delta_E = candidate_cost - self.current_cost
        
        # 2. THE STROBE (The Selection Rule)
        # If delta_E < 0, the geometry is 'more local'. We fall into it.
        if delta_E < 0:
            self.state = candidate_state
            self.current_cost = candidate_cost
            
            # 3. THE POP (Global Minimum Check)
            if self.current_cost < self.min_cost:
                self.min_cost = self.current_cost
                self.best_state = self.state.copy()
                return True, self.current_cost # "POP" Event occurred
        
        else:
            # If delta_E > 0, we might accept it to tunnel out of local minima.
            # "Cooling Schedule": The sensitivity drops over time (t).
            probability = np.exp(-delta_E / (self.sensitivity * (1.0 + t/1000.0)))
            if np.random.rand() < probability:
                self.state = candidate_state
                self.current_cost = candidate_cost
                
        return False, self.current_cost

    def run(self, cycles, mutation_func):
        """
        Runs the full simulation loop.
        """
        history = []
        for t in range(cycles):
            popped, cost = self.step(t, mutation_func)
            history.append(cost)
        
        return self.best_state, history