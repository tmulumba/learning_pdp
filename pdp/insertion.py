"""
Insertion heuristics for route construction.
"""

from __future__ import annotations
import heapq
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

from .core import Request, Route, RouteNode, Solution, PDPInstance


class InsertionResult:
    """Result of an insertion cost calculation."""
    
    def __init__(
        self,
        request: Request,
        route: Route,
        pickup_pos: int,
        dropoff_pos: int,
        cost: float,
        feasible: bool = True
    ):
        self.request = request
        self.route = route
        self.pickup_pos = pickup_pos
        self.dropoff_pos = dropoff_pos
        self.cost = cost
        self.feasible = feasible


class BaseInsertion(ABC):
    """Abstract base class for insertion heuristics."""
    
    def __init__(self, avg_duration_limit: float = 45.0):
        self.avg_duration_limit = avg_duration_limit
    
    @abstractmethod
    def find_best_insertion(
        self,
        request: Request,
        solution: Solution,
        routes: Optional[List[Route]] = None
    ) -> Optional[InsertionResult]:
        """Find the best insertion position for a request."""
        pass
    
    @abstractmethod
    def insert_request(
        self,
        request: Request,
        solution: Solution,
        result: InsertionResult
    ) -> bool:
        """Insert request into solution using the given result."""
        pass


class CheapestInsertion(BaseInsertion):
    """
    Cheapest insertion heuristic for pickup-delivery problems.
    
    For each request, finds the position in existing routes that minimizes
    additional travel time while respecting precedence and duration constraints.
    """
    
    def __init__(self, avg_duration_limit: float = 45.0):
        super().__init__(avg_duration_limit)
    
    def _calculate_insertion_cost(
        self,
        solution: Solution,
        route: Route,
        request: Request,
        pickup_pos: int,
        dropoff_pos: int
    ) -> float:
        """
        Calculate the cost of inserting pickup at pickup_pos and dropoff at dropoff_pos.
        Cost = additional travel time from the insertion.
        """
        instance = solution.instance
        
        # Create temporary route with insertion
        temp_route = route.copy()
        pickup_node = RouteNode(request.id, is_pickup=True)
        dropoff_node = RouteNode(request.id, is_pickup=False)
        
        temp_route.nodes.insert(pickup_pos, pickup_node)
        temp_route.nodes.insert(dropoff_pos + 1, dropoff_node)  # +1 because pickup was inserted
        
        # Calculate travel time of new route
        new_travel = solution.calculate_route_travel_time(temp_route)
        old_travel = solution.calculate_route_travel_time(route)
        
        return new_travel - old_travel
    
    def _check_feasibility(
        self,
        solution: Solution,
        route: Route,
        request: Request,
        pickup_pos: int,
        dropoff_pos: int
    ) -> bool:
        """Check if insertion maintains feasibility (avg duration constraint)."""
        # Create temporary route
        temp_route = route.copy()
        pickup_node = RouteNode(request.id, is_pickup=True)
        dropoff_node = RouteNode(request.id, is_pickup=False)
        
        temp_route.nodes.insert(pickup_pos, pickup_node)
        temp_route.nodes.insert(dropoff_pos + 1, dropoff_node)
        
        # Check average duration
        stats = solution.calculate_route_duration_stats(temp_route)
        return stats['avg'] <= self.avg_duration_limit
    
    def find_best_insertion(
        self,
        request: Request,
        solution: Solution,
        routes: Optional[List[Route]] = None
    ) -> Optional[InsertionResult]:
        """
        Find the cheapest feasible insertion position for a request.
        
        Args:
            request: The request to insert.
            solution: Current solution.
            routes: Specific routes to consider (default: all routes).
            
        Returns:
            InsertionResult with best position, or None if no feasible insertion.
        """
        if routes is None:
            routes = solution.routes
        
        best_result: Optional[InsertionResult] = None
        best_cost = float('inf')
        
        for route in routes:
            # Try all pickup positions
            for pickup_pos in range(len(route) + 1):
                # Dropoff must be after pickup
                for dropoff_pos in range(pickup_pos, len(route) + 1):
                    cost = self._calculate_insertion_cost(
                        solution, route, request, pickup_pos, dropoff_pos
                    )
                    
                    if cost < best_cost:
                        # Check feasibility
                        if self._check_feasibility(
                            solution, route, request, pickup_pos, dropoff_pos
                        ):
                            best_cost = cost
                            best_result = InsertionResult(
                                request=request,
                                route=route,
                                pickup_pos=pickup_pos,
                                dropoff_pos=dropoff_pos,
                                cost=cost,
                                feasible=True
                            )
        
        return best_result
    
    def insert_request(
        self,
        request: Request,
        solution: Solution,
        result: InsertionResult
    ) -> bool:
        """Insert request using the computed result."""
        if result is None or not result.feasible:
            return False
        
        pickup_node = RouteNode(request.id, is_pickup=True)
        dropoff_node = RouteNode(request.id, is_pickup=False)
        
        result.route.nodes.insert(result.pickup_pos, pickup_node)
        result.route.nodes.insert(result.dropoff_pos + 1, dropoff_node)
        result.route._request_ids.add(request.id)
        
        return True


class RegretInsertion(BaseInsertion):
    """
    Regret-based insertion heuristic.
    
    Prioritizes requests where the difference between best and k-th best
    insertion is largest (high regret = insert now to avoid bad placement later).
    """
    
    def __init__(self, k: int = 2, avg_duration_limit: float = 45.0):
        super().__init__(avg_duration_limit)
        self.k = k
        self._cheapest = CheapestInsertion(avg_duration_limit)
    
    def calculate_regret(
        self,
        request: Request,
        solution: Solution
    ) -> Tuple[float, Optional[InsertionResult]]:
        """
        Calculate k-regret for a request.
        
        Returns:
            Tuple of (regret_value, best_insertion_result)
        """
        # Find costs for all routes
        costs_and_results: List[Tuple[float, InsertionResult]] = []
        
        for route in solution.routes:
            result = self._cheapest.find_best_insertion(
                request, solution, routes=[route]
            )
            if result is not None:
                costs_and_results.append((result.cost, result))
        
        if not costs_and_results:
            return float('inf'), None
        
        # Sort by cost
        costs_and_results.sort(key=lambda x: x[0])
        
        best_result = costs_and_results[0][1]
        best_cost = costs_and_results[0][0]
        
        # Calculate regret (difference to k-th best, or last if < k options)
        k_idx = min(self.k - 1, len(costs_and_results) - 1)
        k_cost = costs_and_results[k_idx][0]
        
        regret = k_cost - best_cost
        return regret, best_result
    
    def find_best_insertion(
        self,
        request: Request,
        solution: Solution,
        routes: Optional[List[Route]] = None
    ) -> Optional[InsertionResult]:
        """Find best insertion (delegates to cheapest for single request)."""
        return self._cheapest.find_best_insertion(request, solution, routes)
    
    def insert_request(
        self,
        request: Request,
        solution: Solution,
        result: InsertionResult
    ) -> bool:
        """Insert request using the computed result."""
        return self._cheapest.insert_request(request, solution, result)


class GreedyInsertion(BaseInsertion):
    """
    Greedy insertion that creates new route if no feasible insertion exists.
    """
    
    def __init__(self, avg_duration_limit: float = 45.0):
        super().__init__(avg_duration_limit)
        self._cheapest = CheapestInsertion(avg_duration_limit)
    
    def find_best_insertion(
        self,
        request: Request,
        solution: Solution,
        routes: Optional[List[Route]] = None
    ) -> Optional[InsertionResult]:
        """Find best insertion or signal need for new route."""
        result = self._cheapest.find_best_insertion(request, solution, routes)
        
        if result is None:
            # Create a dummy result indicating new route needed
            return InsertionResult(
                request=request,
                route=None,  # type: ignore
                pickup_pos=0,
                dropoff_pos=0,
                cost=0,
                feasible=False  # Flag to create new route
            )
        
        return result
    
    def insert_request(
        self,
        request: Request,
        solution: Solution,
        result: InsertionResult
    ) -> bool:
        """Insert request, creating new route if needed."""
        if result.feasible and result.route is not None:
            return self._cheapest.insert_request(request, solution, result)
        else:
            # Create new route for this request
            route = solution.create_route()
            route.nodes.append(RouteNode(request.id, is_pickup=True))
            route.nodes.append(RouteNode(request.id, is_pickup=False))
            route._request_ids.add(request.id)
            return True


def build_solution_with_insertion(
    instance: PDPInstance,
    cluster_requests: Dict[int, List[Request]],
    insertion_heuristic: BaseInsertion,
    order_by: str = 'food_ready_time'
) -> Solution:
    """
    Build a complete solution using insertion heuristic.
    
    Args:
        instance: PDP instance.
        cluster_requests: Requests grouped by cluster.
        insertion_heuristic: Heuristic to use for insertion.
        order_by: How to order requests ('food_ready_time', 'created_at', 'random').
        
    Returns:
        Complete Solution object.
    """
    solution = Solution(instance)
    
    for cluster_id, requests in cluster_requests.items():
        # Order requests within cluster
        if order_by == 'food_ready_time':
            # Use heap for efficient ordering
            heap = [(r.food_ready_time, r.created_at, r.id, r) for r in requests]
            heapq.heapify(heap)
            ordered = [heapq.heappop(heap)[3] for _ in range(len(heap))]
        elif order_by == 'created_at':
            ordered = sorted(requests, key=lambda r: r.created_at)
        else:
            ordered = requests
        
        # Create initial route for cluster
        cluster_route = solution.create_route()
        
        for i, request in enumerate(ordered):
            if i == 0:
                # First request: add directly
                cluster_route.nodes.append(RouteNode(request.id, is_pickup=True))
                cluster_route.nodes.append(RouteNode(request.id, is_pickup=False))
                cluster_route._request_ids.add(request.id)
            else:
                # Find best insertion in cluster routes
                cluster_routes = [r for r in solution.routes 
                                 if any(instance.requests[rid].cluster_id == cluster_id 
                                       for rid in r.request_ids)]
                
                result = insertion_heuristic.find_best_insertion(
                    request, solution, routes=cluster_routes
                )
                
                if not insertion_heuristic.insert_request(request, solution, result):
                    # Fallback: create new route
                    new_route = solution.create_route()
                    new_route.nodes.append(RouteNode(request.id, is_pickup=True))
                    new_route.nodes.append(RouteNode(request.id, is_pickup=False))
                    new_route._request_ids.add(request.id)
    
    solution.remove_empty_routes()
    return solution

