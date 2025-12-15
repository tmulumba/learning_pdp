"""
Main PDP Solver combining clustering and insertion.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import time

from .core import PDPInstance, Solution
from .clustering import (
    BaseClusterer, 
    SpatioTemporalClusterer, 
    get_requests_by_cluster
)
from .insertion import (
    BaseInsertion,
    CheapestInsertion,
    GreedyInsertion,
    build_solution_with_insertion
)


class PDPSolver:
    """
    Main solver for Pickup-Delivery Problems.
    
    Implements a cluster-first route-second approach:
    1. Cluster requests based on spatio-temporal features
    2. Build routes within each cluster using insertion heuristics
    3. Optionally improve with local search / ALNS
    """
    
    def __init__(
        self,
        n_clusters: int = 50,
        avg_duration_limit: float = 45.0,
        clustering_method: str = 'spatio_temporal',
        insertion_method: str = 'greedy',
        scaling: str = 'standard',
        random_state: int = 0
    ):
        """
        Args:
            n_clusters: Number of clusters for decomposition.
            avg_duration_limit: Maximum average delivery duration (minutes).
            clustering_method: 'spatio_temporal' or 'geographic'.
            insertion_method: 'cheapest', 'greedy', or 'regret'.
            scaling: Feature scaling method ('standard' or 'minmax').
            random_state: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.avg_duration_limit = avg_duration_limit
        self.clustering_method = clustering_method
        self.insertion_method = insertion_method
        self.scaling = scaling
        self.random_state = random_state
        
        self._clusterer: Optional[BaseClusterer] = None
        self._insertion: Optional[BaseInsertion] = None
        self._solution: Optional[Solution] = None
        self._stats: Dict[str, Any] = {}
    
    def _create_clusterer(self) -> BaseClusterer:
        """Create the clustering component."""
        if self.clustering_method == 'spatio_temporal':
            return SpatioTemporalClusterer(
                n_clusters=self.n_clusters,
                scaling=self.scaling,
                random_state=self.random_state
            )
        elif self.clustering_method == 'geographic':
            from .clustering import GeographicClusterer
            return GeographicClusterer(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def _create_insertion(self) -> BaseInsertion:
        """Create the insertion heuristic."""
        if self.insertion_method == 'cheapest':
            return CheapestInsertion(avg_duration_limit=self.avg_duration_limit)
        elif self.insertion_method == 'greedy':
            return GreedyInsertion(avg_duration_limit=self.avg_duration_limit)
        elif self.insertion_method == 'regret':
            from .insertion import RegretInsertion
            return RegretInsertion(k=2, avg_duration_limit=self.avg_duration_limit)
        else:
            raise ValueError(f"Unknown insertion method: {self.insertion_method}")
    
    def solve(self, instance: PDPInstance, verbose: bool = False) -> Solution:
        """
        Solve the PDP instance.
        
        Args:
            instance: The PDP instance to solve.
            verbose: Whether to print progress information.
            
        Returns:
            Solution object containing routes.
        """
        start_time = time.time()
        
        # Phase 1: Clustering
        if verbose:
            print(f"Phase 1: Clustering {len(instance)} requests into {self.n_clusters} clusters...")
        
        cluster_start = time.time()
        self._clusterer = self._create_clusterer()
        cluster_map = self._clusterer.fit_predict(instance)
        cluster_time = time.time() - cluster_start
        
        actual_clusters = self._clusterer.get_num_clusters()
        if verbose:
            print(f"  Created {actual_clusters} clusters in {cluster_time:.2f}s")
        
        # Precompute travel times per cluster for efficiency
        cluster_requests = get_requests_by_cluster(instance)
        for cid, requests in cluster_requests.items():
            request_ids = [r.id for r in requests]
            instance.precompute_travel_times(request_ids)
        
        # Phase 2: Route construction via insertion
        if verbose:
            print(f"Phase 2: Building routes with {self.insertion_method} insertion...")
        
        insertion_start = time.time()
        self._insertion = self._create_insertion()
        self._solution = build_solution_with_insertion(
            instance=instance,
            cluster_requests=cluster_requests,
            insertion_heuristic=self._insertion,
            order_by='food_ready_time'
        )
        insertion_time = time.time() - insertion_start
        
        if verbose:
            print(f"  Built {self._solution.num_routes} routes in {insertion_time:.2f}s")
        
        # Calculate solution statistics
        total_time = time.time() - start_time
        self._stats = {
            'total_time': total_time,
            'cluster_time': cluster_time,
            'insertion_time': insertion_time,
            'num_clusters': actual_clusters,
            'num_routes': self._solution.num_routes,
            'efficiency': self._solution.calculate_efficiency(),
            'total_travel_time': self._solution.calculate_total_travel_time(),
        }
        
        # Check feasibility
        feasible, message = self._solution.is_feasible(self.avg_duration_limit)
        self._stats['feasible'] = feasible
        self._stats['feasibility_message'] = message
        
        if verbose:
            print(f"\nSolution Summary:")
            print(f"  Routes: {self._stats['num_routes']}")
            print(f"  Efficiency: {self._stats['efficiency']:.2f} deliveries/hour")
            print(f"  Total travel time: {self._stats['total_travel_time']:.1f} minutes")
            print(f"  Feasible: {feasible} ({message})")
            print(f"  Total time: {total_time:.2f}s")
        
        return self._solution
    
    @property
    def solution(self) -> Optional[Solution]:
        """Get the current solution."""
        return self._solution
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return self._stats
    
    def get_efficiency(self) -> float:
        """Get solution efficiency (deliveries per courier-hour)."""
        if self._solution is None:
            return 0.0
        return self._solution.calculate_efficiency()


def solve_from_csv(
    input_file: str,
    output_file: Optional[str] = None,
    n_clusters: int = 50,
    avg_duration_limit: float = 45.0,
    verbose: bool = True
) -> Solution:
    """
    Convenience function to solve a PDP instance from CSV.
    
    Args:
        input_file: Path to input CSV file.
        output_file: Optional path to save solution CSV.
        n_clusters: Number of clusters.
        avg_duration_limit: Max average delivery duration.
        verbose: Whether to print progress.
        
    Returns:
        Solution object.
    """
    # Load instance
    instance = PDPInstance.from_csv(input_file)
    
    # Solve
    solver = PDPSolver(
        n_clusters=n_clusters,
        avg_duration_limit=avg_duration_limit
    )
    solution = solver.solve(instance, verbose=verbose)
    
    # Save if requested
    if output_file:
        solution.save_csv(output_file)
        if verbose:
            print(f"\nSolution saved to {output_file}")
    
    return solution


# Allow running as script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve PDP instance')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('-o', '--output', help='Output CSV file', default='output.csv')
    parser.add_argument('-k', '--clusters', type=int, default=50, help='Number of clusters')
    parser.add_argument('-t', '--time-limit', type=float, default=45.0, 
                       help='Avg delivery time limit (minutes)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    solve_from_csv(
        input_file=args.input,
        output_file=args.output,
        n_clusters=args.clusters,
        avg_duration_limit=args.time_limit,
        verbose=not args.quiet
    )

