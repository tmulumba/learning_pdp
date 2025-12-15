"""
ML-based cluster count prediction.

Predicts the optimal number of clusters based on instance features,
eliminating manual parameter tuning.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import os

from ..core import PDPInstance, Request, haversine


@dataclass
class InstanceFeatures:
    """Features extracted from a PDP instance for cluster prediction."""
    
    # Basic statistics
    num_requests: int
    
    # Spatial features
    pickup_centroid: Tuple[float, float]
    dropoff_centroid: Tuple[float, float]
    pickup_spread: float  # Std dev of distances from centroid
    dropoff_spread: float
    avg_pickup_dropoff_dist: float
    max_pickup_dropoff_dist: float
    
    # Temporal features
    time_span: float  # Range of food ready times
    avg_ready_time: float
    std_ready_time: float
    
    # Density features
    avg_5nn_distance: float  # Average 5-nearest-neighbor distance
    density_coefficient: float  # Requests per sq km
    
    # Derived
    aspect_ratio: float  # pickup_spread / dropoff_spread
    
    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.num_requests,
            self.pickup_spread,
            self.dropoff_spread,
            self.avg_pickup_dropoff_dist,
            self.max_pickup_dropoff_dist,
            self.time_span,
            self.avg_ready_time,
            self.std_ready_time,
            self.avg_5nn_distance,
            self.density_coefficient,
            self.aspect_ratio,
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names."""
        return [
            'num_requests',
            'pickup_spread',
            'dropoff_spread',
            'avg_pickup_dropoff_dist',
            'max_pickup_dropoff_dist',
            'time_span',
            'avg_ready_time',
            'std_ready_time',
            'avg_5nn_distance',
            'density_coefficient',
            'aspect_ratio',
        ]


class InstanceFeatureExtractor:
    """Extracts features from PDP instances."""
    
    def extract(self, instance: PDPInstance) -> InstanceFeatures:
        """Extract features from an instance."""
        requests = list(instance.requests.values())
        n = len(requests)
        
        # Collect coordinates
        pickup_lats = [r.pickup_lat for r in requests]
        pickup_lons = [r.pickup_lon for r in requests]
        dropoff_lats = [r.dropoff_lat for r in requests]
        dropoff_lons = [r.dropoff_lon for r in requests]
        ready_times = [r.food_ready_time for r in requests]
        
        # Centroids
        pickup_centroid = (np.mean(pickup_lats), np.mean(pickup_lons))
        dropoff_centroid = (np.mean(dropoff_lats), np.mean(dropoff_lons))
        
        # Spreads (distance from centroid)
        pickup_dists = [
            haversine(r.pickup_lat, r.pickup_lon, 
                     pickup_centroid[0], pickup_centroid[1])
            for r in requests
        ]
        dropoff_dists = [
            haversine(r.dropoff_lat, r.dropoff_lon,
                     dropoff_centroid[0], dropoff_centroid[1])
            for r in requests
        ]
        pickup_spread = np.std(pickup_dists)
        dropoff_spread = np.std(dropoff_dists)
        
        # Pickup to dropoff distances
        pd_dists = [
            haversine(r.pickup_lat, r.pickup_lon,
                     r.dropoff_lat, r.dropoff_lon)
            for r in requests
        ]
        avg_pd_dist = np.mean(pd_dists)
        max_pd_dist = np.max(pd_dists)
        
        # Temporal features
        time_span = np.max(ready_times) - np.min(ready_times)
        avg_ready = np.mean(ready_times)
        std_ready = np.std(ready_times)
        
        # 5-NN density
        avg_5nn = self._compute_avg_knn(requests, k=5)
        
        # Density coefficient
        # Approximate area covered
        lat_range = np.max(pickup_lats) - np.min(pickup_lats)
        lon_range = np.max(pickup_lons) - np.min(pickup_lons)
        # Convert to km (approximate)
        area_km2 = (lat_range * 111) * (lon_range * 85)  # rough conversion
        density = n / max(area_km2, 0.01)
        
        # Aspect ratio
        aspect = pickup_spread / max(dropoff_spread, 0.001)
        
        return InstanceFeatures(
            num_requests=n,
            pickup_centroid=pickup_centroid,
            dropoff_centroid=dropoff_centroid,
            pickup_spread=pickup_spread,
            dropoff_spread=dropoff_spread,
            avg_pickup_dropoff_dist=avg_pd_dist,
            max_pickup_dropoff_dist=max_pd_dist,
            time_span=time_span,
            avg_ready_time=avg_ready,
            std_ready_time=std_ready,
            avg_5nn_distance=avg_5nn,
            density_coefficient=density,
            aspect_ratio=aspect,
        )
    
    def _compute_avg_knn(self, requests: List[Request], k: int = 5) -> float:
        """Compute average k-nearest-neighbor distance."""
        if len(requests) <= k:
            return 0.0
        
        total = 0.0
        count = 0
        
        for r in requests:
            distances = []
            for other in requests:
                if other.id != r.id:
                    d = haversine(r.pickup_lat, r.pickup_lon,
                                 other.pickup_lat, other.pickup_lon)
                    distances.append(d)
            
            distances.sort()
            total += np.mean(distances[:k])
            count += 1
        
        return total / count


class ClusterCountPredictor:
    """
    Predicts optimal cluster count for a PDP instance.
    
    Uses a regression model trained on instance features and
    their optimal cluster counts (found via grid search).
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = InstanceFeatureExtractor()
        self.model = None
        self.scaler = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            # Use a simple heuristic model by default
            self._use_heuristic = True
    
    def predict(self, instance: PDPInstance) -> int:
        """
        Predict optimal cluster count for an instance.
        
        Args:
            instance: PDP instance.
            
        Returns:
            Predicted optimal cluster count.
        """
        features = self.feature_extractor.extract(instance)
        
        if hasattr(self, '_use_heuristic') and self._use_heuristic:
            return self._heuristic_predict(features)
        
        # Use trained model
        X = features.to_array().reshape(1, -1)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        k_pred = self.model.predict(X)[0]
        
        # Ensure reasonable bounds
        k_pred = max(5, min(int(round(k_pred)), features.num_requests // 2))
        
        return k_pred
    
    def _heuristic_predict(self, features: InstanceFeatures) -> int:
        """
        Heuristic prediction based on instance characteristics.
        
        Based on empirical observations:
        - More requests -> more clusters
        - Higher spread -> more clusters
        - Longer time span -> more clusters (temporal separation)
        """
        n = features.num_requests
        
        # Base: roughly sqrt(n) clusters
        base = int(np.sqrt(n) * 2.5)
        
        # Adjust for spread
        spread_factor = (features.pickup_spread + features.dropoff_spread) / 10.0
        base = int(base * (1 + 0.2 * spread_factor))
        
        # Adjust for time span (longer span = more clusters)
        if features.time_span > 60:  # More than 1 hour
            time_factor = features.time_span / 120.0
            base = int(base * (1 + 0.1 * time_factor))
        
        # Bounds
        k = max(10, min(base, n // 3))
        
        return k
    
    def train(
        self,
        instances: List[PDPInstance],
        optimal_ks: List[int],
        model_type: str = 'rf'
    ):
        """
        Train the cluster count predictor.
        
        Args:
            instances: List of training instances.
            optimal_ks: Corresponding optimal cluster counts.
            model_type: 'rf' (Random Forest), 'gbm' (Gradient Boosting), or 'linear'.
        """
        from sklearn.preprocessing import StandardScaler
        
        # Extract features
        X = []
        for instance in instances:
            features = self.feature_extractor.extract(instance)
            X.append(features.to_array())
        X = np.array(X)
        y = np.array(optimal_ks)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gbm':
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
        
        self.model.fit(X_scaled, y)
        self._use_heuristic = False
    
    def save(self, filepath: str):
        """Save trained model."""
        data = {
            'model': self.model,
            'scaler': self.scaler,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self._use_heuristic = False
    
    def find_optimal_k(
        self,
        instance: PDPInstance,
        k_range: Tuple[int, int] = (10, 100),
        step: int = 5,
        metric: str = 'efficiency'
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal K via grid search (for training data collection).
        
        Args:
            instance: PDP instance.
            k_range: Range of K values to try.
            k_step: Step size for grid search.
            metric: 'efficiency' or 'travel_time'.
            
        Returns:
            Tuple of (best_k, {k: metric_value}).
        """
        from ..solver import PDPSolver
        
        results = {}
        
        for k in range(k_range[0], k_range[1] + 1, step):
            if k > len(instance) // 2:
                break
            
            solver = PDPSolver(n_clusters=k)
            solution = solver.solve(instance, verbose=False)
            
            if metric == 'efficiency':
                results[k] = solution.calculate_efficiency()
            else:
                results[k] = -solution.calculate_total_travel_time()
        
        best_k = max(results.keys(), key=lambda k: results[k])
        return best_k, results

