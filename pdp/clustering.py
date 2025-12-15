"""
Clustering module for spatial-temporal request grouping.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .core import Request, PDPInstance


class BaseClusterer(ABC):
    """Abstract base class for clustering strategies."""
    
    @abstractmethod
    def fit_predict(self, instance: PDPInstance) -> Dict[int, int]:
        """
        Cluster requests and return mapping from request_id to cluster_id.
        """
        pass
    
    @abstractmethod
    def get_num_clusters(self) -> int:
        """Return the number of clusters."""
        pass


class SpatioTemporalClusterer(BaseClusterer):
    """
    Clusters requests using K-Means on spatio-temporal features.
    
    Features:
        - Pickup latitude, longitude
        - Dropoff latitude, longitude  
        - Food ready time
    """
    
    def __init__(
        self,
        n_clusters: int = 50,
        scaling: str = 'standard',
        random_state: int = 42,
        feature_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            n_clusters: Number of clusters (K).
            scaling: 'standard' for z-score, 'minmax' for [0,1] scaling.
            random_state: Random seed for reproducibility.
            feature_weights: Optional weights for each feature dimension.
        """
        self.n_clusters = n_clusters
        self.scaling = scaling
        self.random_state = random_state
        self.feature_weights = feature_weights or {}
        
        self._kmeans: Optional[KMeans] = None
        self._scaler = None
        self._labels: Optional[np.ndarray] = None
    
    def _extract_features(self, instance: PDPInstance) -> Tuple[np.ndarray, List[int]]:
        """Extract feature matrix from instance."""
        request_ids = []
        features = []
        
        for rid, req in instance.requests.items():
            request_ids.append(rid)
            features.append([
                req.pickup_lat,
                req.pickup_lon,
                req.dropoff_lat,
                req.dropoff_lon,
                req.food_ready_time
            ])
        
        return np.array(features), request_ids
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Apply feature scaling."""
        if self.scaling == 'standard':
            self._scaler = StandardScaler()
        elif self.scaling == 'minmax':
            self._scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")
        
        X_scaled = self._scaler.fit_transform(X)
        
        # Apply feature weights if specified
        if self.feature_weights:
            weights = np.array([
                self.feature_weights.get('pickup_lat', 1.0),
                self.feature_weights.get('pickup_lon', 1.0),
                self.feature_weights.get('dropoff_lat', 1.0),
                self.feature_weights.get('dropoff_lon', 1.0),
                self.feature_weights.get('food_ready_time', 1.0),
            ])
            X_scaled = X_scaled * weights
        
        return X_scaled
    
    def fit_predict(self, instance: PDPInstance) -> Dict[int, int]:
        """
        Cluster requests and return mapping from request_id to cluster_id.
        """
        X, request_ids = self._extract_features(instance)
        X_scaled = self._scale_features(X)
        
        # Adjust n_clusters if more clusters than samples
        actual_clusters = min(self.n_clusters, len(request_ids))
        
        self._kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self._labels = self._kmeans.fit_predict(X_scaled)
        
        # Update requests with cluster assignments
        cluster_map = {}
        for rid, label in zip(request_ids, self._labels):
            cluster_map[rid] = int(label)
            instance.requests[rid].cluster_id = int(label)
        
        return cluster_map
    
    def get_num_clusters(self) -> int:
        """Return the actual number of clusters used."""
        if self._labels is None:
            return self.n_clusters
        return len(np.unique(self._labels))
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Return size of each cluster."""
        if self._labels is None:
            return {}
        unique, counts = np.unique(self._labels, return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int)))
    
    def get_cluster_centroids(self) -> Optional[np.ndarray]:
        """Return cluster centroids in scaled feature space."""
        if self._kmeans is None:
            return None
        return self._kmeans.cluster_centers_


class GeographicClusterer(BaseClusterer):
    """Clusters requests using only geographic coordinates (baseline)."""
    
    def __init__(
        self,
        n_clusters: int = 50,
        use_dropoff: bool = True,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.use_dropoff = use_dropoff
        self.random_state = random_state
        self._labels = None
    
    def fit_predict(self, instance: PDPInstance) -> Dict[int, int]:
        request_ids = []
        features = []
        
        for rid, req in instance.requests.items():
            request_ids.append(rid)
            if self.use_dropoff:
                features.append([req.pickup_lat, req.pickup_lon, 
                                req.dropoff_lat, req.dropoff_lon])
            else:
                features.append([req.pickup_lat, req.pickup_lon])
        
        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        actual_clusters = min(self.n_clusters, len(request_ids))
        kmeans = KMeans(n_clusters=actual_clusters, random_state=self.random_state, n_init=10)
        self._labels = kmeans.fit_predict(X_scaled)
        
        cluster_map = {}
        for rid, label in zip(request_ids, self._labels):
            cluster_map[rid] = int(label)
            instance.requests[rid].cluster_id = int(label)
        
        return cluster_map
    
    def get_num_clusters(self) -> int:
        if self._labels is None:
            return self.n_clusters
        return len(np.unique(self._labels))


class SingleCluster(BaseClusterer):
    """Puts all requests in a single cluster (for testing)."""
    
    def fit_predict(self, instance: PDPInstance) -> Dict[int, int]:
        cluster_map = {}
        for rid in instance.requests:
            cluster_map[rid] = 0
            instance.requests[rid].cluster_id = 0
        return cluster_map
    
    def get_num_clusters(self) -> int:
        return 1


def get_requests_by_cluster(instance: PDPInstance) -> Dict[int, List[Request]]:
    """Group requests by their cluster assignment."""
    clusters: Dict[int, List[Request]] = {}
    
    for req in instance.requests.values():
        cid = req.cluster_id
        if cid is None:
            cid = 0  # Default cluster
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(req)
    
    return clusters

