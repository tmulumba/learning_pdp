"""
Solution evaluator for PDP solutions.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .core import Solution, PDPInstance, haversine, DASHER_SPEED


@dataclass
class EvaluationResult:
    """Results from solution evaluation."""
    feasible: bool
    efficiency: float
    avg_delivery_time: float
    num_routes: int
    total_travel_time: float
    violations: List[str]
    
    def __str__(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        s = f"Evaluation Result: {status}\n"
        s += f"  Efficiency: {self.efficiency:.2f} deliveries/hour\n"
        s += f"  Avg Delivery Time: {self.avg_delivery_time:.1f} minutes\n"
        s += f"  Num Routes: {self.num_routes}\n"
        s += f"  Total Travel Time: {self.total_travel_time:.1f} minutes\n"
        if self.violations:
            s += f"  Violations:\n"
            for v in self.violations[:10]:  # Show first 10
                s += f"    - {v}\n"
            if len(self.violations) > 10:
                s += f"    ... and {len(self.violations) - 10} more\n"
        return s


class SolutionEvaluator:
    """
    Evaluates PDP solutions for feasibility and quality.
    """
    
    TRAVEL_TIME_TOLERANCE_SECS = 3  # Tolerance for travel time validation
    
    def __init__(self, avg_duration_limit: float = 45.0):
        self.avg_duration_limit = avg_duration_limit
    
    def evaluate(self, solution: Solution) -> EvaluationResult:
        """
        Evaluate a solution for feasibility and compute metrics.
        
        Args:
            solution: Solution to evaluate.
            
        Returns:
            EvaluationResult with metrics and violations.
        """
        violations = []
        
        # Check completeness
        completeness_violations = self._check_completeness(solution)
        violations.extend(completeness_violations)
        
        # Check precedence
        precedence_violations = self._check_precedence(solution)
        violations.extend(precedence_violations)
        
        # Check travel times
        travel_violations = self._check_travel_times(solution)
        violations.extend(travel_violations)
        
        # Check average delivery time
        duration_violations = self._check_duration_constraint(solution)
        violations.extend(duration_violations)
        
        # Calculate metrics
        efficiency = solution.calculate_efficiency()
        avg_delivery = self._calculate_avg_delivery_time(solution)
        total_travel = solution.calculate_total_travel_time()
        
        return EvaluationResult(
            feasible=len(violations) == 0,
            efficiency=efficiency,
            avg_delivery_time=avg_delivery,
            num_routes=solution.num_routes,
            total_travel_time=total_travel,
            violations=violations
        )
    
    def _check_completeness(self, solution: Solution) -> List[str]:
        """Check that all requests are served exactly once."""
        violations = []
        
        expected = set(solution.instance.requests.keys())
        assigned = solution.get_all_request_ids()
        
        missing = expected - assigned
        if missing:
            violations.append(f"Missing requests: {missing}")
        
        extra = assigned - expected
        if extra:
            violations.append(f"Extra/duplicate requests: {extra}")
        
        # Check each request has exactly one pickup and one dropoff
        pickup_counts: Dict[int, int] = {}
        dropoff_counts: Dict[int, int] = {}
        
        for route in solution.routes:
            for node in route.nodes:
                if node.is_pickup:
                    pickup_counts[node.request_id] = pickup_counts.get(node.request_id, 0) + 1
                else:
                    dropoff_counts[node.request_id] = dropoff_counts.get(node.request_id, 0) + 1
        
        for rid in expected:
            p_count = pickup_counts.get(rid, 0)
            d_count = dropoff_counts.get(rid, 0)
            if p_count != 1:
                violations.append(f"Request {rid} has {p_count} pickups (expected 1)")
            if d_count != 1:
                violations.append(f"Request {rid} has {d_count} dropoffs (expected 1)")
        
        return violations
    
    def _check_precedence(self, solution: Solution) -> List[str]:
        """Check that pickups precede dropoffs for each request."""
        violations = []
        
        for route in solution.routes:
            seen_pickups = set()
            for node in route.nodes:
                if node.is_pickup:
                    seen_pickups.add(node.request_id)
                else:
                    if node.request_id not in seen_pickups:
                        violations.append(
                            f"Route {route.route_id}: Dropoff for {node.request_id} "
                            f"before pickup"
                        )
        
        return violations
    
    def _check_travel_times(self, solution: Solution) -> List[str]:
        """Check that travel times between nodes are feasible."""
        violations = []
        
        for route in solution.routes:
            solution.calculate_route_times(route)
            
            for i in range(1, len(route)):
                prev = route.nodes[i - 1]
                curr = route.nodes[i]
                
                # Get coordinates
                prev_req = solution.instance.get_request(prev.request_id)
                curr_req = solution.instance.get_request(curr.request_id)
                
                if prev.is_pickup:
                    prev_coords = prev_req.pickup_coords
                else:
                    prev_coords = prev_req.dropoff_coords
                
                if curr.is_pickup:
                    curr_coords = curr_req.pickup_coords
                else:
                    curr_coords = curr_req.dropoff_coords
                
                # Calculate required travel time
                dist_km = haversine(
                    prev_coords[0], prev_coords[1],
                    curr_coords[0], curr_coords[1]
                )
                required_seconds = dist_km * 1000 / DASHER_SPEED
                
                # Check actual travel time
                actual_minutes = curr.arrival_time - prev.arrival_time
                actual_seconds = actual_minutes * 60
                
                tolerance = self.TRAVEL_TIME_TOLERANCE_SECS
                if actual_seconds < required_seconds - tolerance:
                    violations.append(
                        f"Route {route.route_id}: Insufficient travel time "
                        f"from node {i-1} to {i} "
                        f"(need {required_seconds:.1f}s, have {actual_seconds:.1f}s)"
                    )
        
        return violations
    
    def _check_duration_constraint(self, solution: Solution) -> List[str]:
        """Check average delivery time constraint."""
        violations = []
        
        all_durations = []
        for route in solution.routes:
            stats = solution.calculate_route_duration_stats(route)
            if stats['count'] > 0:
                all_durations.extend(
                    [stats['total'] / stats['count']] * int(stats['count'])
                )
        
        if all_durations:
            overall_avg = np.mean(all_durations)
            if overall_avg > self.avg_duration_limit:
                violations.append(
                    f"Average delivery time {overall_avg:.1f} min "
                    f"> limit {self.avg_duration_limit} min"
                )
        
        return violations
    
    def _calculate_avg_delivery_time(self, solution: Solution) -> float:
        """Calculate overall average delivery time."""
        all_durations = []
        
        for route in solution.routes:
            solution.calculate_route_times(route)
            for node in route.nodes:
                if not node.is_pickup:
                    req = solution.instance.get_request(node.request_id)
                    duration = node.arrival_time - req.created_at
                    all_durations.append(duration)
        
        return np.mean(all_durations) if all_durations else 0.0


def evaluate_from_files(
    solution_file: str,
    instance_file: str,
    avg_duration_limit: float = 45.0
) -> EvaluationResult:
    """
    Evaluate a solution from CSV files.
    
    Args:
        solution_file: Path to solution CSV.
        instance_file: Path to instance CSV.
        avg_duration_limit: Maximum average delivery duration.
        
    Returns:
        EvaluationResult.
    """
    # Load instance
    instance = PDPInstance.from_csv(instance_file)
    
    # Load solution
    sol_df = pd.read_csv(solution_file)
    
    # Reconstruct solution
    solution = Solution(instance)
    
    # Group by route
    route_groups = sol_df.groupby('Route ID')
    
    for route_id, group in route_groups:
        from .core import Route, RouteNode
        route = Route(int(route_id))
        
        group_sorted = group.sort_values('Route Point Index')
        for _, row in group_sorted.iterrows():
            is_pickup = str(row['Route Point Type']).strip() == 'Pickup'
            node = RouteNode(
                request_id=int(row['Delivery ID']),
                is_pickup=is_pickup
            )
            route.nodes.append(node)
            if is_pickup:
                route._request_ids.add(node.request_id)
        
        solution.routes.append(route)
    
    # Evaluate
    evaluator = SolutionEvaluator(avg_duration_limit=avg_duration_limit)
    return evaluator.evaluate(solution)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <solution.csv> <instance.csv>")
        sys.exit(1)
    
    result = evaluate_from_files(sys.argv[1], sys.argv[2])
    print(result)

