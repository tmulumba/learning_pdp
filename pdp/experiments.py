"""
Experiment harness for PDP solver evaluation.

Provides utilities for:
- Running experiments with different configurations
- Comparing against baselines
- Generating result tables and plots
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .core import PDPInstance, Solution
from .solver import PDPSolver
from .clustering import SpatioTemporalClusterer, GeographicClusterer, SingleCluster
from .insertion import CheapestInsertion, GreedyInsertion, RegretInsertion
from .alns import ALNS, ALNSConfig, create_default_alns
from .evaluator import SolutionEvaluator, EvaluationResult


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    name: str
    
    # Clustering
    clustering_method: str = 'spatio_temporal'  # 'spatio_temporal', 'geographic', 'single', 'gnn'
    n_clusters: int = 50
    predict_clusters: bool = False  # Use ML to predict n_clusters
    
    # Insertion
    insertion_method: str = 'greedy'  # 'cheapest', 'greedy', 'regret'
    
    # ALNS
    use_alns: bool = False
    alns_iterations: int = 1000
    alns_time_limit: float = 8000.0 # 300.0 #
    use_rl_operator: bool = False  # Use RL for operator selection
    
    # Constraints
    avg_duration_limit: float = 45.0
    
    # Random seed
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    
    config_name: str
    instance_name: str
    
    # Metrics
    efficiency: float
    avg_delivery_time: float
    total_travel_time: float
    num_routes: int
    feasible: bool
    
    # Timing
    clustering_time: float
    insertion_time: float
    alns_time: float
    total_time: float
    
    # Additional info
    num_requests: int
    actual_clusters: int
    alns_iterations: int = 0
    alns_improvements: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ExperimentRunner:
    """Runs experiments with different configurations."""
    
    def __init__(
        self,
        output_dir: str = 'experiment_results',
        verbose: bool = True
    ):
        self.output_dir = output_dir
        self.verbose = verbose
        self.results: List[ExperimentResult] = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single(
        self,
        instance: PDPInstance,
        config: ExperimentConfig,
        instance_name: str = 'instance'
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            instance: PDP instance.
            config: Experiment configuration.
            instance_name: Name for reporting.
            
        Returns:
            ExperimentResult.
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Running: {config.name} on {instance_name}")
            print(f"{'='*50}")
        
        # Determine number of clusters
        n_clusters = config.n_clusters
        if config.predict_clusters:
            from .learning.cluster_predictor import ClusterCountPredictor
            predictor = ClusterCountPredictor()
            n_clusters = predictor.predict(instance)
            if self.verbose:
                print(f"  Predicted clusters: {n_clusters}")
        
        # Clustering phase
        cluster_start = time.time()
        
        if config.clustering_method == 'gnn':
            from .learning.gnn_clustering import GNNClusterer
            clusterer = GNNClusterer(n_clusters=n_clusters, random_state=config.seed)
        elif config.clustering_method == 'geographic':
            clusterer = GeographicClusterer(n_clusters=n_clusters, random_state=config.seed)
        elif config.clustering_method == 'single':
            clusterer = SingleCluster()
        else:
            clusterer = SpatioTemporalClusterer(
                n_clusters=n_clusters, 
                random_state=config.seed
            )
        
        cluster_map = clusterer.fit_predict(instance)
        actual_clusters = clusterer.get_num_clusters()
        clustering_time = time.time() - cluster_start
        
        if self.verbose:
            print(f"  Clustering: {actual_clusters} clusters in {clustering_time:.2f}s")
        
        # Route construction
        insertion_start = time.time()
        solver = PDPSolver(
            n_clusters=actual_clusters,
            avg_duration_limit=config.avg_duration_limit,
            insertion_method=config.insertion_method,
            random_state=config.seed
        )
        
        # Use pre-clustered data
        from .clustering import get_requests_by_cluster
        from .insertion import build_solution_with_insertion
        
        cluster_requests = get_requests_by_cluster(instance)
        
        if config.insertion_method == 'regret':
            insertion = RegretInsertion(k=2, avg_duration_limit=config.avg_duration_limit)
        elif config.insertion_method == 'cheapest':
            insertion = CheapestInsertion(avg_duration_limit=config.avg_duration_limit)
        else:
            insertion = GreedyInsertion(avg_duration_limit=config.avg_duration_limit)
        
        solution = build_solution_with_insertion(
            instance=instance,
            cluster_requests=cluster_requests,
            insertion_heuristic=insertion,
            order_by='food_ready_time'
        )
        insertion_time = time.time() - insertion_start
        
        if self.verbose:
            print(f"  Insertion: {solution.num_routes} routes in {insertion_time:.2f}s")
        
        # ALNS improvement
        alns_time = 0.0
        alns_iterations = 0
        alns_improvements = 0
        
        if config.use_alns:
            alns_start = time.time()
            
            alns_config = ALNSConfig(
                max_iterations=config.alns_iterations,
                max_time=config.alns_time_limit
            )
            
            if config.use_rl_operator:
                from .learning.rl_operator import create_default_rl_alns, RLGuidedALNS
                destroy_ops, repair_ops, agent = create_default_rl_alns(
                    avg_duration_limit=config.avg_duration_limit
                )
                alns = RLGuidedALNS(destroy_ops, repair_ops, agent, alns_config)
            else:
                alns = create_default_alns(
                    avg_duration_limit=config.avg_duration_limit,
                    config=alns_config
                )
            
            solution, alns_stats = alns.run(solution, verbose=self.verbose)
            
            alns_time = time.time() - alns_start
            alns_iterations = alns_stats.iterations
            alns_improvements = alns_stats.improvements
            
            if self.verbose:
                print(f"  ALNS: {alns_iterations} iters, {alns_improvements} improvements in {alns_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Evaluate
        evaluator = SolutionEvaluator(avg_duration_limit=config.avg_duration_limit)
        eval_result = evaluator.evaluate(solution)
        
        result = ExperimentResult(
            config_name=config.name,
            instance_name=instance_name,
            efficiency=eval_result.efficiency,
            avg_delivery_time=eval_result.avg_delivery_time,
            total_travel_time=eval_result.total_travel_time,
            num_routes=eval_result.num_routes,
            feasible=eval_result.feasible,
            clustering_time=clustering_time,
            insertion_time=insertion_time,
            alns_time=alns_time,
            total_time=total_time,
            num_requests=len(instance),
            actual_clusters=actual_clusters,
            alns_iterations=alns_iterations,
            alns_improvements=alns_improvements,
        )
        
        self.results.append(result)
        
        if self.verbose:
            print(f"\n  Results:")
            print(f"    Efficiency: {result.efficiency:.2f} deliveries/hour")
            print(f"    Avg Delivery Time: {result.avg_delivery_time:.1f} min")
            print(f"    Routes: {result.num_routes}")
            print(f"    Feasible: {result.feasible}")
            print(f"    Total Time: {result.total_time:.2f}s")
        
        return result
    
    def run_comparison(
        self,
        instance: PDPInstance,
        configs: List[ExperimentConfig],
        instance_name: str = 'instance'
    ) -> pd.DataFrame:
        """
        Compare multiple configurations on same instance.
        
        Returns:
            DataFrame with results for each config.
        """
        results = []
        for config in configs:
            result = self.run_single(instance, config, instance_name)
            results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def run_benchmark(
        self,
        instances: Dict[str, PDPInstance],
        configs: List[ExperimentConfig]
    ) -> pd.DataFrame:
        """
        Run full benchmark across instances and configurations.
        
        Args:
            instances: Dict mapping instance names to instances.
            configs: List of configurations to test.
            
        Returns:
            DataFrame with all results.
        """
        all_results = []
        
        total = len(instances) * len(configs)
        current = 0
        
        for inst_name, instance in instances.items():
            for config in configs:
                current += 1
                if self.verbose:
                    print(f"\n[{current}/{total}] {config.name} on {inst_name}")
                
                result = self.run_single(instance, config, inst_name)
                all_results.append(result.to_dict())
        
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(self.output_dir, f'results_{timestamp}.csv'), index=False)
        
        return df
    
    def generate_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics table."""
        summary = df.groupby('config_name').agg({
            'efficiency': ['mean', 'std'],
            'avg_delivery_time': ['mean', 'std'],
            'total_travel_time': ['mean', 'std'],
            'num_routes': ['mean', 'std'],
            'total_time': ['mean', 'std'],
            'feasible': ['mean'],  # Fraction feasible
        }).round(2)
        
        return summary
    
    def save_results(self, filepath: Optional[str] = None):
        """Save all results to file."""
        if not self.results:
            return
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f'results_{timestamp}.json')
        
        data = [r.to_dict() for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def get_default_configs() -> List[ExperimentConfig]:
    """Get default set of experiment configurations."""
    return [
        # Baselines
        ExperimentConfig(
            name='greedy_geo',
            clustering_method='geographic',
            insertion_method='greedy',
            use_alns=False,
        ),
        ExperimentConfig(
            name='greedy_spatiotemporal',
            clustering_method='spatio_temporal',
            insertion_method='greedy',
            use_alns=False,
        ),
        
        # With ALNS
        ExperimentConfig(
            name='greedy_st_alns100',
            clustering_method='spatio_temporal',
            insertion_method='greedy',
            use_alns=True,
            alns_iterations=100,
        ),
        ExperimentConfig(
            name='greedy_st_alns500',
            clustering_method='spatio_temporal',
            insertion_method='greedy',
            use_alns=True,
            alns_iterations=500,
        ),
        ExperimentConfig(
            name='greedy_st_alns1000',
            clustering_method='spatio_temporal',
            insertion_method='greedy',
            use_alns=True,
            alns_iterations=1000,
        ),
        
        # With learning
        ExperimentConfig(
            name='ml_clusters_alns',
            clustering_method='spatio_temporal',
            predict_clusters=True,
            insertion_method='greedy',
            use_alns=True,
            alns_iterations=1000,
        ),
        ExperimentConfig(
            name='gnn_clusters_alns',
            clustering_method='gnn',
            insertion_method='greedy',
            use_alns=True,
            alns_iterations=1000,
        ),
    ]


def run_quick_test(instance_path: str, output_dir: str = 'test_results'):
    """Quick test run on a single instance."""
    instance = PDPInstance.from_csv(instance_path)
    
    runner = ExperimentRunner(output_dir=output_dir)
    
    configs = [
        ExperimentConfig(name='baseline', use_alns=False),
        ExperimentConfig(name='alns_100', use_alns=True, alns_iterations=100),
    ]
    
    results_df = runner.run_comparison(
        instance, 
        configs, 
        instance_name=os.path.basename(instance_path)
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results_df[['config_name', 'efficiency', 'avg_delivery_time', 
                      'num_routes', 'total_time', 'feasible']])
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run PDP experiments')
    parser.add_argument('instance', help='Instance CSV file')
    parser.add_argument('-o', '--output', default='experiment_results',
                       help='Output directory')
    parser.add_argument('--full', action='store_true',
                       help='Run full benchmark with all configs')
    
    args = parser.parse_args()
    
    if args.full:
        instance = PDPInstance.from_csv(args.instance)
        runner = ExperimentRunner(output_dir=args.output)
        configs = get_default_configs()
        results = runner.run_comparison(
            instance, configs,
            instance_name=os.path.basename(args.instance)
        )
        print("\n" + runner.generate_summary_table(results).to_string())
    else:
        run_quick_test(args.instance, args.output)

