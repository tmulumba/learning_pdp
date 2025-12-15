"""
Core data structures for the PDP solver.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from math import radians, cos, sin, asin, sqrt
from copy import deepcopy


# Constants
DASHER_SPEED = 4.5  # meters per second
EARTH_RADIUS_KM = 6371


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * EARTH_RADIUS_KM


def travel_time_minutes(lat1: float, lon1: float, lat2: float, lon2: float,
                        speed_mps: float = DASHER_SPEED) -> float:
    """Calculate travel time in minutes between two points."""
    dist_km = haversine(lat1, lon1, lat2, lon2)
    dist_m = dist_km * 1000
    time_seconds = dist_m / speed_mps
    return time_seconds / 60


@dataclass
class Request:
    """A pickup-delivery request."""
    id: int
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    created_at: float  # minutes from epoch
    food_ready_time: float  # minutes from epoch
    cluster_id: Optional[int] = None
    
    @property
    def pickup_coords(self) -> Tuple[float, float]:
        return (self.pickup_lat, self.pickup_lon)
    
    @property
    def dropoff_coords(self) -> Tuple[float, float]:
        return (self.dropoff_lat, self.dropoff_lon)
    
    def travel_time_pickup_to_dropoff(self) -> float:
        """Travel time from pickup to dropoff in minutes."""
        return travel_time_minutes(
            self.pickup_lat, self.pickup_lon,
            self.dropoff_lat, self.dropoff_lon
        )


@dataclass
class RouteNode:
    """A node in a route (either pickup or dropoff)."""
    request_id: int
    is_pickup: bool
    arrival_time: float = 0.0
    
    @property
    def node_id(self) -> int:
        """Unique node ID (request_id for pickup, request_id + n for dropoff)."""
        # This will be set based on total requests
        return self.request_id if self.is_pickup else -self.request_id


class Route:
    """A dasher route containing pickup and delivery nodes."""
    
    def __init__(self, route_id: int):
        self.route_id = route_id
        self.nodes: List[RouteNode] = []
        self._request_ids: Set[int] = set()
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, idx: int) -> RouteNode:
        return self.nodes[idx]
    
    def copy(self) -> Route:
        """Create a deep copy of the route."""
        new_route = Route(self.route_id)
        new_route.nodes = [RouteNode(n.request_id, n.is_pickup, n.arrival_time) 
                          for n in self.nodes]
        new_route._request_ids = self._request_ids.copy()
        return new_route
    
    @property
    def request_ids(self) -> Set[int]:
        return self._request_ids
    
    def add_request(self, request: Request, pickup_idx: int, dropoff_idx: int):
        """Insert a request's pickup and dropoff at specified positions."""
        pickup_node = RouteNode(request.id, is_pickup=True)
        dropoff_node = RouteNode(request.id, is_pickup=False)
        
        self.nodes.insert(pickup_idx, pickup_node)
        # Adjust dropoff index since pickup was inserted
        adjusted_dropoff_idx = dropoff_idx + 1 if dropoff_idx >= pickup_idx else dropoff_idx
        self.nodes.insert(adjusted_dropoff_idx, dropoff_node)
        self._request_ids.add(request.id)
    
    def remove_request(self, request_id: int) -> bool:
        """Remove a request's pickup and dropoff from the route."""
        if request_id not in self._request_ids:
            return False
        
        self.nodes = [n for n in self.nodes if n.request_id != request_id]
        self._request_ids.remove(request_id)
        return True
    
    def insert_node(self, node: RouteNode, position: int):
        """Insert a node at a specific position."""
        self.nodes.insert(position, node)
        if node.is_pickup:
            self._request_ids.add(node.request_id)
    
    def get_pickup_position(self, request_id: int) -> Optional[int]:
        """Get the position of a request's pickup node."""
        for i, node in enumerate(self.nodes):
            if node.request_id == request_id and node.is_pickup:
                return i
        return None
    
    def get_dropoff_position(self, request_id: int) -> Optional[int]:
        """Get the position of a request's dropoff node."""
        for i, node in enumerate(self.nodes):
            if node.request_id == request_id and not node.is_pickup:
                return i
        return None
    
    def is_valid(self) -> bool:
        """Check if all pickups precede their dropoffs."""
        seen_pickups = set()
        for node in self.nodes:
            if node.is_pickup:
                seen_pickups.add(node.request_id)
            else:
                if node.request_id not in seen_pickups:
                    return False
        return True


class PDPInstance:
    """A PDP instance with requests and precomputed data."""
    
    def __init__(self, requests: List[Request]):
        self.requests = {r.id: r for r in requests}
        self.n = len(requests)
        self._travel_time_cache: Dict[Tuple, float] = {}
    
    def __len__(self) -> int:
        return self.n
    
    def get_request(self, request_id: int) -> Request:
        return self.requests[request_id]
    
    def get_travel_time(self, from_request_id: int, from_is_pickup: bool,
                        to_request_id: int, to_is_pickup: bool) -> float:
        """Get travel time between two nodes with caching."""
        cache_key = (from_request_id, from_is_pickup, to_request_id, to_is_pickup)
        
        if cache_key not in self._travel_time_cache:
            from_req = self.requests[from_request_id]
            to_req = self.requests[to_request_id]
            
            from_coords = from_req.pickup_coords if from_is_pickup else from_req.dropoff_coords
            to_coords = to_req.pickup_coords if to_is_pickup else to_req.dropoff_coords
            
            self._travel_time_cache[cache_key] = travel_time_minutes(
                from_coords[0], from_coords[1],
                to_coords[0], to_coords[1]
            )
        
        return self._travel_time_cache[cache_key]
    
    def precompute_travel_times(self, request_ids: Optional[List[int]] = None):
        """Precompute all travel times for given requests."""
        if request_ids is None:
            request_ids = list(self.requests.keys())
        
        for r1_id in request_ids:
            for r2_id in request_ids:
                for from_pickup in [True, False]:
                    for to_pickup in [True, False]:
                        self.get_travel_time(r1_id, from_pickup, r2_id, to_pickup)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, epoch_str: str = '2015-02-03 02:00:00') -> PDPInstance:
        """Create instance from a pandas DataFrame."""
        epoch = pd.to_datetime(epoch_str)
        
        requests = []
        for _, row in df.iterrows():
            food_ready = pd.to_datetime(row['food_ready_time'])
            created = pd.to_datetime(row['created_at'])
            
            req = Request(
                id=int(row['delivery_id']),
                pickup_lat=float(row['pickup_lat']),
                pickup_lon=float(row['pickup_long']),
                dropoff_lat=float(row['dropoff_lat']),
                dropoff_lon=float(row['dropoff_long']),
                created_at=(created - epoch).total_seconds() / 60,
                food_ready_time=(food_ready - epoch).total_seconds() / 60,
            )
            requests.append(req)
        
        return cls(requests)
    
    @classmethod
    def from_csv(cls, filepath: str) -> PDPInstance:
        """Load instance from CSV file."""
        df = pd.read_csv(filepath)
        return cls.from_dataframe(df)


class Solution:
    """A complete solution to a PDP instance."""
    
    def __init__(self, instance: PDPInstance):
        self.instance = instance
        self.routes: List[Route] = []
        self._route_counter = 0
    
    def copy(self) -> Solution:
        """Create a deep copy of the solution."""
        new_sol = Solution(self.instance)
        new_sol.routes = [r.copy() for r in self.routes]
        new_sol._route_counter = self._route_counter
        return new_sol
    
    @property
    def num_routes(self) -> int:
        return len(self.routes)
    
    def create_route(self) -> Route:
        """Create a new empty route."""
        route = Route(self._route_counter)
        self._route_counter += 1
        self.routes.append(route)
        return route
    
    def add_route(self, route: Route):
        """Add an existing route to the solution."""
        self.routes.append(route)
    
    def remove_empty_routes(self):
        """Remove routes with no nodes."""
        self.routes = [r for r in self.routes if len(r) > 0]
    
    def get_request_route(self, request_id: int) -> Optional[Route]:
        """Find which route contains a request."""
        for route in self.routes:
            if request_id in route.request_ids:
                return route
        return None
    
    def get_all_request_ids(self) -> Set[int]:
        """Get all request IDs in the solution."""
        all_ids = set()
        for route in self.routes:
            all_ids.update(route.request_ids)
        return all_ids
    
    def get_unassigned_requests(self) -> List[int]:
        """Get request IDs not assigned to any route."""
        assigned = self.get_all_request_ids()
        return [rid for rid in self.instance.requests.keys() if rid not in assigned]
    
    def calculate_route_times(self, route: Route) -> Dict[int, float]:
        """Calculate arrival times for all nodes in a route."""
        if len(route) == 0:
            return {}
        
        times = {}
        for i, node in enumerate(route.nodes):
            req = self.instance.get_request(node.request_id)
            
            if i == 0:
                # First node: start at food ready time
                times[i] = req.food_ready_time
            else:
                prev_node = route.nodes[i - 1]
                travel = self.instance.get_travel_time(
                    prev_node.request_id, prev_node.is_pickup,
                    node.request_id, node.is_pickup
                )
                times[i] = times[i - 1] + travel
                
                # Wait if food not ready yet (for pickups)
                if node.is_pickup:
                    times[i] = max(times[i], req.food_ready_time)
            
            node.arrival_time = times[i]
        
        return times
    
    def calculate_route_duration_stats(self, route: Route) -> Dict[str, float]:
        """Calculate delivery duration statistics for a route."""
        self.calculate_route_times(route)
        
        durations = []
        for node in route.nodes:
            if not node.is_pickup:
                req = self.instance.get_request(node.request_id)
                duration = node.arrival_time - req.created_at
                durations.append(duration)
        
        if not durations:
            return {'avg': 0, 'max': 0, 'min': 0, 'total': 0}
        
        return {
            'avg': np.mean(durations),
            'max': np.max(durations),
            'min': np.min(durations),
            'total': np.sum(durations),
            'count': len(durations)
        }
    
    def calculate_route_travel_time(self, route: Route) -> float:
        """Calculate total travel time for a route."""
        if len(route) <= 1:
            return 0.0
        
        total = 0.0
        for i in range(1, len(route)):
            prev = route.nodes[i - 1]
            curr = route.nodes[i]
            total += self.instance.get_travel_time(
                prev.request_id, prev.is_pickup,
                curr.request_id, curr.is_pickup
            )
        return total
    
    def calculate_total_travel_time(self) -> float:
        """Calculate total travel time across all routes."""
        return sum(self.calculate_route_travel_time(r) for r in self.routes)
    
    def calculate_efficiency(self) -> float:
        """Calculate deliveries per dasher-hour."""
        total_time_hours = 0.0
        total_deliveries = 0
        
        for route in self.routes:
            if len(route) == 0:
                continue
            
            self.calculate_route_times(route)
            first_time = route.nodes[0].arrival_time
            last_time = route.nodes[-1].arrival_time
            route_hours = (last_time - first_time) / 60
            total_time_hours += route_hours
            total_deliveries += len(route.request_ids)
        
        return total_deliveries / total_time_hours if total_time_hours > 0 else 0.0
    
    def is_feasible(self, avg_duration_limit: float = 45.0) -> Tuple[bool, str]:
        """Check if solution is feasible."""
        # Check all requests covered
        assigned = self.get_all_request_ids()
        expected = set(self.instance.requests.keys())
        if assigned != expected:
            missing = expected - assigned
            extra = assigned - expected
            return False, f"Missing requests: {missing}, Extra: {extra}"
        
        # Check precedence constraints
        for route in self.routes:
            if not route.is_valid():
                return False, f"Route {route.route_id} has precedence violation"
        
        # Check average duration constraint (per route)
        for route in self.routes:
            stats = self.calculate_route_duration_stats(route)
            if stats['avg'] > avg_duration_limit:
                return False, f"Route {route.route_id} avg duration {stats['avg']:.1f} > {avg_duration_limit}"
        
        return True, "Feasible"
    
    def to_dataframe(self, epoch_str: str = '2015-02-03 02:00:00') -> pd.DataFrame:
        """Convert solution to output DataFrame format."""
        epoch = pd.to_datetime(epoch_str)
        epoch_unix = (epoch - pd.to_datetime('1970-01-01')).total_seconds()
        
        rows = []
        for route in self.routes:
            self.calculate_route_times(route)
            for i, node in enumerate(route.nodes):
                rows.append({
                    'Route ID': route.route_id,
                    'Route Point Index': i,
                    'Delivery ID': node.request_id,
                    'Route Point Type': 'Pickup' if node.is_pickup else 'DropOff',
                    'Route Point Time': epoch_unix + node.arrival_time * 60
                })
        
        return pd.DataFrame(rows)
    
    def save_csv(self, filepath: str):
        """Save solution to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

