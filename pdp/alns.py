"""
Adaptive Large Neighborhood Search (ALNS) for PDP improvement.
"""

from __future__ import annotations
import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import time

from .core import Request, Route, RouteNode, Solution, PDPInstance


# =============================================================================
# Operator Base Classes
# =============================================================================

class DestroyOperator(ABC):
    """Base class for destroy operators."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __call__(
        self, 
        solution: Solution, 
        num_remove: int
    ) -> Tuple[Solution, List[Request]]:
        """
        Remove requests from solution.
        
        Args:
            solution: Current solution (will be modified).
            num_remove: Number of requests to remove.
            
        Returns:
            Tuple of (modified solution, list of removed requests).
        """
        pass


class RepairOperator(ABC):
    """Base class for repair operators."""
    
    def __init__(self, name: str, avg_duration_limit: float = 45.0):
        self.name = name
        self.avg_duration_limit = avg_duration_limit
    
    @abstractmethod
    def __call__(
        self, 
        solution: Solution, 
        requests: List[Request]
    ) -> Solution:
        """
        Reinsert requests into solution.
        
        Args:
            solution: Current solution (will be modified).
            requests: Requests to insert.
            
        Returns:
            Modified solution.
        """
        pass


# =============================================================================
# Destroy Operators
# =============================================================================

class RandomRemoval(DestroyOperator):
    """Remove random requests from solution."""
    
    def __init__(self):
        super().__init__("random_removal")
    
    def __call__(
        self, 
        solution: Solution, 
        num_remove: int
    ) -> Tuple[Solution, List[Request]]:
        all_requests = list(solution.get_all_request_ids())
        num_remove = min(num_remove, len(all_requests))
        
        to_remove = random.sample(all_requests, num_remove)
        removed = []
        
        for rid in to_remove:
            req = solution.instance.get_request(rid)
            removed.append(req)
            
            # Find and remove from route
            for route in solution.routes:
                if route.remove_request(rid):
                    break
        
        solution.remove_empty_routes()
        return solution, removed


class WorstRemoval(DestroyOperator):
    """Remove requests with highest cost contribution."""
    
    def __init__(self, randomization: float = 0.3):
        super().__init__("worst_removal")
        self.randomization = randomization
    
    def _calculate_removal_savings(
        self, 
        solution: Solution, 
        request_id: int
    ) -> float:
        """Calculate savings from removing a request."""
        route = solution.get_request_route(request_id)
        if route is None:
            return 0.0
        
        # Current travel time
        current_travel = solution.calculate_route_travel_time(route)
        
        # Travel time without request
        temp_route = route.copy()
        temp_route.remove_request(request_id)
        
        if len(temp_route) == 0:
            return current_travel
        
        new_travel = solution.calculate_route_travel_time(temp_route)
        return current_travel - new_travel
    
    def __call__(
        self, 
        solution: Solution, 
        num_remove: int
    ) -> Tuple[Solution, List[Request]]:
        removed = []
        
        for _ in range(num_remove):
            if len(solution.get_all_request_ids()) == 0:
                break
            
            # Calculate savings for each request
            savings = []
            for rid in solution.get_all_request_ids():
                s = self._calculate_removal_savings(solution, rid)
                savings.append((rid, s))
            
            # Sort by savings (highest first)
            savings.sort(key=lambda x: -x[1])
            
            # Select with randomization
            # y^p where p is randomization parameter, y uniform [0,1]
            y = random.random()
            idx = int(y ** self.randomization * len(savings))
            idx = min(idx, len(savings) - 1)
            
            rid = savings[idx][0]
            req = solution.instance.get_request(rid)
            removed.append(req)
            
            for route in solution.routes:
                if route.remove_request(rid):
                    break
        
        solution.remove_empty_routes()
        return solution, removed


class RelatedRemoval(DestroyOperator):
    """Remove geographically/temporally related requests."""
    
    def __init__(self, weight_distance: float = 0.5, weight_time: float = 0.5):
        super().__init__("related_removal")
        self.weight_distance = weight_distance
        self.weight_time = weight_time
    
    def _calculate_relatedness(
        self, 
        req1: Request, 
        req2: Request
    ) -> float:
        """Calculate relatedness score (lower = more related)."""
        # Spatial distance (pickup to pickup + dropoff to dropoff)
        from .core import haversine
        pickup_dist = haversine(
            req1.pickup_lat, req1.pickup_lon,
            req2.pickup_lat, req2.pickup_lon
        )
        dropoff_dist = haversine(
            req1.dropoff_lat, req1.dropoff_lon,
            req2.dropoff_lat, req2.dropoff_lon
        )
        spatial = pickup_dist + dropoff_dist
        
        # Temporal distance
        temporal = abs(req1.food_ready_time - req2.food_ready_time)
        
        # Normalize and combine
        # (Using rough estimates for normalization)
        spatial_norm = spatial / 20.0  # ~20km max reasonable
        temporal_norm = temporal / 120.0  # ~2 hours max
        
        return self.weight_distance * spatial_norm + self.weight_time * temporal_norm
    
    def __call__(
        self, 
        solution: Solution, 
        num_remove: int
    ) -> Tuple[Solution, List[Request]]:
        all_rids = list(solution.get_all_request_ids())
        if not all_rids:
            return solution, []
        
        # Start with random request
        seed_rid = random.choice(all_rids)
        seed_req = solution.instance.get_request(seed_rid)
        
        # Calculate relatedness to seed
        relatedness = []
        for rid in all_rids:
            req = solution.instance.get_request(rid)
            rel = self._calculate_relatedness(seed_req, req)
            relatedness.append((rid, rel))
        
        # Sort by relatedness (most related first)
        relatedness.sort(key=lambda x: x[1])
        
        # Remove most related
        removed = []
        for i in range(min(num_remove, len(relatedness))):
            rid = relatedness[i][0]
            req = solution.instance.get_request(rid)
            removed.append(req)
            
            for route in solution.routes:
                if route.remove_request(rid):
                    break
        
        solution.remove_empty_routes()
        return solution, removed


class ShawRemoval(DestroyOperator):
    """
    Shaw removal - removes related requests with some randomization.
    Classic ALNS operator from Shaw (1998).
    """
    
    def __init__(
        self, 
        phi_distance: float = 9.0,
        phi_time: float = 3.0,
        phi_demand: float = 2.0,
        randomization: float = 6.0
    ):
        super().__init__("shaw_removal")
        self.phi_distance = phi_distance
        self.phi_time = phi_time
        self.phi_demand = phi_demand
        self.randomization = randomization
    
    def _relatedness(self, req1: Request, req2: Request) -> float:
        """Shaw relatedness measure."""
        from .core import haversine
        
        dist = haversine(
            req1.pickup_lat, req1.pickup_lon,
            req2.pickup_lat, req2.pickup_lon
        )
        time_diff = abs(req1.food_ready_time - req2.food_ready_time)
        
        return self.phi_distance * dist + self.phi_time * time_diff
    
    def __call__(
        self, 
        solution: Solution, 
        num_remove: int
    ) -> Tuple[Solution, List[Request]]:
        all_rids = list(solution.get_all_request_ids())
        if not all_rids:
            return solution, []
        
        removed_rids: Set[int] = set()
        
        # Start with random request
        seed_rid = random.choice(all_rids)
        removed_rids.add(seed_rid)
        
        while len(removed_rids) < num_remove and len(removed_rids) < len(all_rids):
            # Pick random removed request as reference
            ref_rid = random.choice(list(removed_rids))
            ref_req = solution.instance.get_request(ref_rid)
            
            # Find most related non-removed request
            candidates = [r for r in all_rids if r not in removed_rids]
            if not candidates:
                break
            
            # Calculate relatedness
            relatedness = []
            for cid in candidates:
                creq = solution.instance.get_request(cid)
                rel = self._relatedness(ref_req, creq)
                relatedness.append((cid, rel))
            
            relatedness.sort(key=lambda x: x[1])
            
            # Select with randomization
            y = random.random()
            idx = int(y ** self.randomization * len(relatedness))
            idx = min(idx, len(relatedness) - 1)
            
            removed_rids.add(relatedness[idx][0])
        
        # Actually remove from solution
        removed = []
        for rid in removed_rids:
            req = solution.instance.get_request(rid)
            removed.append(req)
            for route in solution.routes:
                if route.remove_request(rid):
                    break
        
        solution.remove_empty_routes()
        return solution, removed


# =============================================================================
# Repair Operators
# =============================================================================

class GreedyRepair(RepairOperator):
    """Greedy insertion repair - insert each request at cheapest position."""
    
    def __init__(self, avg_duration_limit: float = 45.0):
        super().__init__("greedy_repair", avg_duration_limit)
    
    def _find_best_insertion(
        self,
        solution: Solution,
        request: Request
    ) -> Tuple[Optional[Route], int, int, float]:
        """Find best insertion position for a request."""
        best_route = None
        best_pickup_pos = 0
        best_dropoff_pos = 0
        best_cost = float('inf')
        
        for route in solution.routes:
            for p_pos in range(len(route) + 1):
                for d_pos in range(p_pos, len(route) + 1):
                    cost = self._insertion_cost(
                        solution, route, request, p_pos, d_pos
                    )
                    if cost < best_cost:
                        # Check feasibility
                        if self._check_feasible(solution, route, request, p_pos, d_pos):
                            best_cost = cost
                            best_route = route
                            best_pickup_pos = p_pos
                            best_dropoff_pos = d_pos
        
        return best_route, best_pickup_pos, best_dropoff_pos, best_cost
    
    def _insertion_cost(
        self,
        solution: Solution,
        route: Route,
        request: Request,
        pickup_pos: int,
        dropoff_pos: int
    ) -> float:
        """Calculate insertion cost."""
        if len(route) == 0:
            return 0.0
        
        # Create temporary route
        temp = route.copy()
        temp.nodes.insert(pickup_pos, RouteNode(request.id, True))
        temp.nodes.insert(dropoff_pos + 1, RouteNode(request.id, False))
        
        new_travel = solution.calculate_route_travel_time(temp)
        old_travel = solution.calculate_route_travel_time(route)
        
        return new_travel - old_travel
    
    def _check_feasible(
        self,
        solution: Solution,
        route: Route,
        request: Request,
        pickup_pos: int,
        dropoff_pos: int
    ) -> bool:
        """Check if insertion maintains feasibility."""
        temp = route.copy()
        temp.nodes.insert(pickup_pos, RouteNode(request.id, True))
        temp.nodes.insert(dropoff_pos + 1, RouteNode(request.id, False))
        temp._request_ids.add(request.id)
        
        stats = solution.calculate_route_duration_stats(temp)
        return stats['avg'] <= self.avg_duration_limit
    
    def __call__(
        self, 
        solution: Solution, 
        requests: List[Request]
    ) -> Solution:
        # Shuffle requests for randomization
        shuffled = list(requests)
        random.shuffle(shuffled)
        
        for request in shuffled:
            best_route, p_pos, d_pos, cost = self._find_best_insertion(
                solution, request
            )
            
            if best_route is not None:
                # Insert into existing route
                best_route.nodes.insert(p_pos, RouteNode(request.id, True))
                best_route.nodes.insert(d_pos + 1, RouteNode(request.id, False))
                best_route._request_ids.add(request.id)
            else:
                # Create new route
                route = solution.create_route()
                route.nodes.append(RouteNode(request.id, True))
                route.nodes.append(RouteNode(request.id, False))
                route._request_ids.add(request.id)
        
        return solution


class RegretRepair(RepairOperator):
    """
    Regret-k insertion - prioritize requests with high insertion regret.
    """
    
    def __init__(self, k: int = 2, avg_duration_limit: float = 45.0):
        super().__init__("regret_repair", avg_duration_limit)
        self.k = k
        self._greedy = GreedyRepair(avg_duration_limit)
    
    def _calculate_regret(
        self,
        solution: Solution,
        request: Request
    ) -> Tuple[float, Tuple[Optional[Route], int, int]]:
        """Calculate k-regret for a request."""
        # Get costs for each route
        route_costs = []
        
        for route in solution.routes:
            best_cost = float('inf')
            best_pos = (0, 0)
            
            for p_pos in range(len(route) + 1):
                for d_pos in range(p_pos, len(route) + 1):
                    cost = self._greedy._insertion_cost(
                        solution, route, request, p_pos, d_pos
                    )
                    if cost < best_cost:
                        if self._greedy._check_feasible(
                            solution, route, request, p_pos, d_pos
                        ):
                            best_cost = cost
                            best_pos = (p_pos, d_pos)
            
            if best_cost < float('inf'):
                route_costs.append((best_cost, route, best_pos))
        
        if not route_costs:
            return float('inf'), (None, 0, 0)
        
        route_costs.sort(key=lambda x: x[0])
        
        best = route_costs[0]
        k_idx = min(self.k - 1, len(route_costs) - 1)
        k_cost = route_costs[k_idx][0]
        
        regret = k_cost - best[0]
        return regret, (best[1], best[2][0], best[2][1])
    
    def __call__(
        self, 
        solution: Solution, 
        requests: List[Request]
    ) -> Solution:
        remaining = list(requests)
        
        while remaining:
            # Calculate regret for all remaining requests
            regrets = []
            for req in remaining:
                regret, best = self._calculate_regret(solution, req)
                regrets.append((regret, req, best))
            
            # Sort by regret (highest first)
            regrets.sort(key=lambda x: -x[0])
            
            # Insert request with highest regret
            _, request, (best_route, p_pos, d_pos) = regrets[0]
            
            if best_route is not None:
                best_route.nodes.insert(p_pos, RouteNode(request.id, True))
                best_route.nodes.insert(d_pos + 1, RouteNode(request.id, False))
                best_route._request_ids.add(request.id)
            else:
                route = solution.create_route()
                route.nodes.append(RouteNode(request.id, True))
                route.nodes.append(RouteNode(request.id, False))
                route._request_ids.add(request.id)
            
            remaining.remove(request)
        
        return solution


# =============================================================================
# ALNS Framework
# =============================================================================

@dataclass
class ALNSConfig:
    """Configuration for ALNS algorithm."""
    max_iterations: int = 1000
    max_time: float = 300.0  # seconds
    
    # Removal size
    min_removal: int = 3
    max_removal: int = 20
    removal_fraction: float = 0.15
    
    # Acceptance criterion (simulated annealing)
    initial_temperature: float = 100.0
    cooling_rate: float = 0.9995
    
    # Adaptive weight updates
    sigma_best: float = 33.0    # New global best
    sigma_better: float = 9.0   # Improving current
    sigma_accept: float = 13.0  # Accepted
    weight_decay: float = 0.8   # Decay factor
    
    # Update frequency
    segment_length: int = 100


@dataclass
class ALNSStats:
    """Statistics from ALNS run."""
    iterations: int = 0
    time_elapsed: float = 0.0
    initial_objective: float = 0.0
    final_objective: float = 0.0
    best_iteration: int = 0
    improvements: int = 0
    accepted: int = 0
    operator_stats: Dict = None
    
    def __post_init__(self):
        if self.operator_stats is None:
            self.operator_stats = {}


class ALNS:
    """
    Adaptive Large Neighborhood Search implementation.
    """
    
    def __init__(
        self,
        destroy_operators: List[DestroyOperator],
        repair_operators: List[RepairOperator],
        config: Optional[ALNSConfig] = None,
        random_state: Optional[int] = None
    ):
        self.destroy_ops = destroy_operators
        self.repair_ops = repair_operators
        self.config = config or ALNSConfig()
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Initialize weights
        self.destroy_weights = np.ones(len(destroy_operators))
        self.repair_weights = np.ones(len(repair_operators))
        
        # Scores for current segment
        self.destroy_scores = np.zeros(len(destroy_operators))
        self.repair_scores = np.zeros(len(repair_operators))
        self.destroy_uses = np.zeros(len(destroy_operators))
        self.repair_uses = np.zeros(len(repair_operators))
    
    def _select_operator(
        self, 
        weights: np.ndarray
    ) -> int:
        """Roulette wheel selection based on weights."""
        probs = weights / weights.sum()
        return np.random.choice(len(weights), p=probs)
    
    def _get_removal_count(self, solution: Solution) -> int:
        """Determine number of requests to remove."""
        n = len(solution.get_all_request_ids())
        target = int(n * self.config.removal_fraction)
        return max(
            self.config.min_removal,
            min(self.config.max_removal, target)
        )
    
    def _calculate_objective(self, solution: Solution) -> float:
        """Calculate objective value (lower is better)."""
        # Primary: total travel time
        # Could add penalty terms for constraint violations
        return solution.calculate_total_travel_time()
    
    def _accept(
        self, 
        delta: float, 
        temperature: float
    ) -> bool:
        """Simulated annealing acceptance criterion."""
        if delta < 0:
            return True
        if temperature <= 0:
            return False
        return random.random() < np.exp(-delta / temperature)
    
    def _update_weights(self):
        """Update operator weights based on scores."""
        cfg = self.config
        
        for i in range(len(self.destroy_ops)):
            if self.destroy_uses[i] > 0:
                self.destroy_weights[i] = (
                    cfg.weight_decay * self.destroy_weights[i] +
                    (1 - cfg.weight_decay) * self.destroy_scores[i] / self.destroy_uses[i]
                )
        
        for i in range(len(self.repair_ops)):
            if self.repair_uses[i] > 0:
                self.repair_weights[i] = (
                    cfg.weight_decay * self.repair_weights[i] +
                    (1 - cfg.weight_decay) * self.repair_scores[i] / self.repair_uses[i]
                )
        
        # Reset scores
        self.destroy_scores.fill(0)
        self.repair_scores.fill(0)
        self.destroy_uses.fill(0)
        self.repair_uses.fill(0)
    
    def run(
        self,
        solution: Solution,
        verbose: bool = False
    ) -> Tuple[Solution, ALNSStats]:
        """
        Run ALNS optimization.
        
        Args:
            solution: Initial solution.
            verbose: Print progress.
            
        Returns:
            Tuple of (best solution, statistics).
        """
        cfg = self.config
        start_time = time.time()
        
        # Initialize
        current = solution.copy()
        best = solution.copy()
        
        current_obj = self._calculate_objective(current)
        best_obj = current_obj
        initial_obj = current_obj
        
        temperature = cfg.initial_temperature
        
        stats = ALNSStats(
            initial_objective=initial_obj,
            operator_stats={
                'destroy': {op.name: {'uses': 0, 'score': 0} 
                           for op in self.destroy_ops},
                'repair': {op.name: {'uses': 0, 'score': 0} 
                          for op in self.repair_ops}
            }
        )
        
        iteration = 0
        while iteration < cfg.max_iterations:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > cfg.max_time:
                break
            
            # Select operators
            d_idx = self._select_operator(self.destroy_weights)
            r_idx = self._select_operator(self.repair_weights)
            
            destroy_op = self.destroy_ops[d_idx]
            repair_op = self.repair_ops[r_idx]
            
            # Apply operators
            candidate = current.copy()
            num_remove = self._get_removal_count(candidate)
            
            candidate, removed = destroy_op(candidate, num_remove)
            candidate = repair_op(candidate, removed)
            
            # Evaluate
            candidate_obj = self._calculate_objective(candidate)
            delta = candidate_obj - current_obj
            
            # Update scores
            self.destroy_uses[d_idx] += 1
            self.repair_uses[r_idx] += 1
            
            score = 0
            if candidate_obj < best_obj:
                # New global best
                best = candidate.copy()
                best_obj = candidate_obj
                score = cfg.sigma_best
                stats.best_iteration = iteration
                stats.improvements += 1
            elif candidate_obj < current_obj:
                # Improving
                score = cfg.sigma_better
                stats.improvements += 1
            elif self._accept(delta, temperature):
                # Accepted (possibly worse)
                score = cfg.sigma_accept
                stats.accepted += 1
            
            if score > 0 or delta < 0:
                current = candidate
                current_obj = candidate_obj
            
            self.destroy_scores[d_idx] += score
            self.repair_scores[r_idx] += score
            
            # Update operator weights periodically
            if (iteration + 1) % cfg.segment_length == 0:
                self._update_weights()
            
            # Cool down
            temperature *= cfg.cooling_rate
            
            # Progress
            if verbose and iteration % 100 == 0:
                print(f"  Iter {iteration}: best={best_obj:.2f}, "
                      f"current={current_obj:.2f}, T={temperature:.2f}")
            
            iteration += 1
        
        # Final stats
        stats.iterations = iteration
        stats.time_elapsed = time.time() - start_time
        stats.final_objective = best_obj
        
        # Record operator usage
        for i, op in enumerate(self.destroy_ops):
            stats.operator_stats['destroy'][op.name]['uses'] = int(self.destroy_uses[i])
        for i, op in enumerate(self.repair_ops):
            stats.operator_stats['repair'][op.name]['uses'] = int(self.repair_uses[i])
        
        return best, stats


def create_default_alns(
    avg_duration_limit: float = 45.0,
    config: Optional[ALNSConfig] = None
) -> ALNS:
    """Create ALNS with default operators."""
    destroy_ops = [
        RandomRemoval(),
        WorstRemoval(randomization=0.3),
        RelatedRemoval(weight_distance=0.5, weight_time=0.5),
        ShawRemoval(),
    ]
    
    repair_ops = [
        GreedyRepair(avg_duration_limit=avg_duration_limit),
        RegretRepair(k=2, avg_duration_limit=avg_duration_limit),
        RegretRepair(k=3, avg_duration_limit=avg_duration_limit),
    ]
    
    return ALNS(destroy_ops, repair_ops, config)

