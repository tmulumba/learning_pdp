# Learning On-Demand Pickup and Delivery Routing

A comprehensive framework for solving Pickup and Delivery Problems (PDP) in food delivery contexts, combining classical optimization heuristics with machine learning.

## Overview

This package implements a **cluster-first route-second** approach enhanced with:

1. **Spatio-Temporal Clustering**: Groups requests by location and time for efficient decomposition
2. **Cheapest Insertion Heuristic**: Constructs routes within clusters while respecting constraints
3. **Adaptive Large Neighborhood Search (ALNS)**: Iteratively improves solutions using destroy/repair operators
4. **Learning Components**:
   - ML-based cluster count prediction
   - RL-guided ALNS operator selection
   - GNN-enhanced clustering

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from pdp import PDPSolver, solve_from_csv

# Solve directly from CSV
solution = solve_from_csv(
    input_file='input.csv',
    output_file='solution.csv',
    n_clusters=50,
    verbose=True
)

# Or use the solver class for more control
from pdp import PDPInstance, PDPSolver

instance = PDPInstance.from_csv('input.csv')
solver = PDPSolver(n_clusters=50, avg_duration_limit=45.0)
solution = solver.solve(instance, verbose=True)

print(f"Efficiency: {solution.calculate_efficiency():.2f} deliveries/hour")
```

### With ALNS Improvement

```python
from pdp import PDPInstance, PDPSolver, create_default_alns, ALNSConfig

# Load instance
instance = PDPInstance.from_csv('input.csv')

# Initial solution
solver = PDPSolver(n_clusters=50)
solution = solver.solve(instance)

# Improve with ALNS
config = ALNSConfig(max_iterations=1000, max_time=300.0)
alns = create_default_alns(avg_duration_limit=45.0, config=config)
improved_solution, stats = alns.run(solution, verbose=True)

print(f"Improvement: {stats.initial_objective - stats.final_objective:.1f} minutes saved")
```

### Running Experiments

```python
from pdp import ExperimentRunner, get_default_configs, PDPInstance

instance = PDPInstance.from_csv('input.csv')
runner = ExperimentRunner(output_dir='results')

# Run with default configurations
configs = get_default_configs()
results_df = runner.run_comparison(instance, configs, instance_name='test')

# View summary
print(runner.generate_summary_table(results_df))
```

### Generating Synthetic Instances

```python
from pdp import SyntheticGenerator, GeneratorConfig

# Generate single instance
config = GeneratorConfig(num_requests=150, seed=42)
generator = SyntheticGenerator(config)
df = generator.generate()
df.to_csv('synthetic_instance.csv', index=False)

# Generate benchmark suite
from pdp import generate_benchmark_suite
files = generate_benchmark_suite(
    output_dir='benchmark_instances',
    num_instances=50,
    sizes=[100, 150, 200]
)
```

## Package Structure

```
pdp/
├── __init__.py          # Main exports
├── core.py              # Data structures (Request, Route, Solution)
├── clustering.py        # Clustering algorithms
├── insertion.py         # Insertion heuristics
├── solver.py            # Main solver
├── alns.py              # ALNS implementation
├── evaluator.py         # Solution evaluation
├── synthetic_generator.py  # Instance generation
├── experiments.py       # Benchmarking
└── learning/
    ├── cluster_predictor.py  # ML cluster count prediction
    ├── rl_operator.py        # RL operator selection
    └── gnn_clustering.py     # GNN embeddings
```

## Key Features

### Clustering Methods
- **Spatio-Temporal** (default): Uses pickup/dropoff coordinates and food ready time
- **Geographic**: Coordinates only
- **GNN-Enhanced**: Learned embeddings capturing routing relevance

### Insertion Heuristics
- **Greedy**: Insert at cheapest feasible position, create new route if needed
- **Cheapest**: Pure cheapest insertion without fallback
- **Regret-k**: Prioritize high-regret requests

### ALNS Operators

**Destroy:**
- Random Removal
- Worst Removal (remove costly requests)
- Related Removal (remove geographically similar)
- Shaw Removal (classic related removal variant)

**Repair:**
- Greedy Insertion
- Regret-2 Insertion
- Regret-3 Insertion

### Constraints
- Pickup must precede dropoff for each request
- Average delivery duration ≤ 45 minutes (configurable)
- Food ready time respected


## License

MIT License

