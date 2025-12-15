"""
PDP Solver Package
==================

A cluster-first route-second framework for Pickup and Delivery Problems
with Adaptive Large Neighborhood Search and learning components.

Main Components:
- Core: Data structures (Request, Route, Solution, PDPInstance)
- Clustering: Spatio-temporal clustering for problem decomposition
- Insertion: Greedy and regret-based insertion heuristics
- ALNS: Adaptive Large Neighborhood Search for improvement
- Learning: ML cluster prediction, RL operator selection, GNN embeddings
- Experiments: Benchmarking harness
"""

from .core import Request, Route, Solution, PDPInstance
from .clustering import SpatioTemporalClusterer, GeographicClusterer
from .insertion import CheapestInsertion, GreedyInsertion, RegretInsertion
from .solver import PDPSolver, solve_from_csv
from .evaluator import SolutionEvaluator, evaluate_from_files
from .alns import (
    ALNS, 
    ALNSConfig, 
    ALNSStats,
    create_default_alns,
    # Destroy operators
    RandomRemoval,
    WorstRemoval,
    RelatedRemoval,
    ShawRemoval,
    # Repair operators
    GreedyRepair,
    RegretRepair,
)
from .synthetic_generator import SyntheticGenerator, GeneratorConfig, generate_benchmark_suite
from .experiments import ExperimentRunner, ExperimentConfig, ExperimentResult, get_default_configs
from .milp_baseline import PDPMILPSolver, OptimalityGapAnalyzer, MILPResult, OptimalityGapResult

__version__ = "0.1.0"
__all__ = [
    # Core
    "Request",
    "Route", 
    "Solution",
    "PDPInstance",
    # Clustering
    "SpatioTemporalClusterer",
    "GeographicClusterer",
    # Insertion
    "CheapestInsertion",
    "GreedyInsertion",
    "RegretInsertion",
    # Solver
    "PDPSolver",
    "solve_from_csv",
    # Evaluator
    "SolutionEvaluator",
    "evaluate_from_files",
    # ALNS
    "ALNS",
    "ALNSConfig",
    "ALNSStats",
    "create_default_alns",
    "RandomRemoval",
    "WorstRemoval",
    "RelatedRemoval",
    "ShawRemoval",
    "GreedyRepair",
    "RegretRepair",
    # Synthetic generation
    "SyntheticGenerator",
    "GeneratorConfig",
    "generate_benchmark_suite",
    # Experiments
    "ExperimentRunner",
    "ExperimentConfig", 
    "ExperimentResult",
    "get_default_configs",
    # MILP Baseline
    "PDPMILPSolver",
    "OptimalityGapAnalyzer",
    "MILPResult",
    "OptimalityGapResult",
]

