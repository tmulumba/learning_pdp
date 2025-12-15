"""
Graph Neural Network for Request Embeddings.

Learns embeddings that capture routing-relevant similarities
between requests for improved clustering.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pickle

from ..core import Request, PDPInstance, haversine


@dataclass
class RequestGraph:
    """Graph representation of requests."""
    
    # Node features: [n_requests, n_features]
    node_features: np.ndarray
    
    # Adjacency matrix or edge list
    adjacency: np.ndarray  # [n_requests, n_requests]
    
    # Edge features (optional)
    edge_features: Optional[np.ndarray] = None  # [n_edges, n_edge_features]
    
    # Request IDs for mapping back
    request_ids: List[int] = None


class RequestGraphBuilder:
    """Builds graph representation from PDP instance."""
    
    def __init__(
        self,
        k_neighbors: int = 10,
        distance_threshold: float = 5.0,  # km
        include_temporal: bool = True
    ):
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.include_temporal = include_temporal
    
    def build(self, instance: PDPInstance) -> RequestGraph:
        """Build graph from instance."""
        requests = list(instance.requests.values())
        n = len(requests)
        request_ids = [r.id for r in requests]
        
        # Node features
        node_features = self._extract_node_features(requests)
        
        # Build adjacency based on k-NN and distance threshold
        adjacency = self._build_adjacency(requests)
        
        return RequestGraph(
            node_features=node_features,
            adjacency=adjacency,
            request_ids=request_ids
        )
    
    def _extract_node_features(self, requests: List[Request]) -> np.ndarray:
        """Extract node features for each request."""
        features = []
        
        for r in requests:
            f = [
                r.pickup_lat,
                r.pickup_lon,
                r.dropoff_lat,
                r.dropoff_lon,
            ]
            
            if self.include_temporal:
                f.extend([
                    r.food_ready_time / 300,  # Normalize to ~[0,1]
                    r.created_at / 300,
                    r.travel_time_pickup_to_dropoff(),
                ])
            
            features.append(f)
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-6
        features = (features - mean) / std
        
        return features
    
    def _build_adjacency(self, requests: List[Request]) -> np.ndarray:
        """Build adjacency matrix based on spatial proximity."""
        n = len(requests)
        adjacency = np.zeros((n, n), dtype=np.float32)
        
        # Compute pairwise distances
        for i, r1 in enumerate(requests):
            distances = []
            for j, r2 in enumerate(requests):
                if i == j:
                    distances.append((j, float('inf')))
                    continue
                
                # Distance between pickup locations
                d_pickup = haversine(
                    r1.pickup_lat, r1.pickup_lon,
                    r2.pickup_lat, r2.pickup_lon
                )
                # Distance between dropoff locations
                d_dropoff = haversine(
                    r1.dropoff_lat, r1.dropoff_lon,
                    r2.dropoff_lat, r2.dropoff_lon
                )
                # Combined distance
                d = (d_pickup + d_dropoff) / 2
                distances.append((j, d))
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Connect to k nearest neighbors within threshold
            for j, d in distances[:self.k_neighbors]:
                if d <= self.distance_threshold:
                    adjacency[i, j] = 1.0
        
        # Make symmetric
        adjacency = np.maximum(adjacency, adjacency.T)
        
        # Add self-loops
        np.fill_diagonal(adjacency, 1.0)
        
        return adjacency


class SimpleGNNLayer:
    """
    Simple Graph Neural Network layer (message passing).
    
    Implements: h' = ReLU(A @ h @ W + b)
    where A is normalized adjacency.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
    
    def forward(
        self, 
        node_features: np.ndarray, 
        adjacency: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            node_features: [n_nodes, in_dim]
            adjacency: [n_nodes, n_nodes]
            
        Returns:
            Updated node features [n_nodes, out_dim]
        """
        # Normalize adjacency (symmetric normalization)
        degree = adjacency.sum(axis=1, keepdims=True)
        degree = np.maximum(degree, 1e-6)  # Avoid division by zero
        norm_adj = adjacency / np.sqrt(degree) / np.sqrt(degree.T)
        
        # Message passing: aggregate neighbor features
        aggregated = norm_adj @ node_features
        
        # Transform
        output = aggregated @ self.W + self.b
        
        # Activation
        output = np.maximum(0, output)  # ReLU
        
        return output


class GNNClusterer:
    """
    GNN-based clusterer for PDP requests.
    
    Uses learned embeddings instead of raw features for clustering.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [32, 16],
        n_clusters: int = 50,
        random_state: int = 42
    ):
        self.hidden_dims = hidden_dims
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.graph_builder = RequestGraphBuilder()
        self.layers: List[SimpleGNNLayer] = []
        self._initialized = False
        self._actual_clusters: Optional[int] = None #TM
    
    def _initialize_layers(self, input_dim: int):
        """Initialize GNN layers."""
        dims = [input_dim] + self.hidden_dims
        
        np.random.seed(self.random_state)
        self.layers = []
        for i in range(len(dims) - 1):
            layer = SimpleGNNLayer(dims[i], dims[i + 1])
            self.layers.append(layer)
        
        self._initialized = True
    
    def get_embeddings(self, instance: PDPInstance) -> Tuple[np.ndarray, List[int]]:
        """
        Get learned embeddings for requests.
        
        Args:
            instance: PDP instance.
            
        Returns:
            Tuple of (embeddings, request_ids)
        """
        # Build graph
        graph = self.graph_builder.build(instance)
        
        # Initialize layers if needed
        if not self._initialized:
            self._initialize_layers(graph.node_features.shape[1])
        
        # Forward through GNN layers
        h = graph.node_features
        for layer in self.layers:
            h = layer.forward(h, graph.adjacency)
        
        return h, graph.request_ids
    
    def fit_predict(self, instance: PDPInstance) -> Dict[int, int]:
        """
        Cluster requests using learned embeddings.
        
        Args:
            instance: PDP instance.
            
        Returns:
            Mapping from request_id to cluster_id.
        """
        from sklearn.cluster import KMeans
        
        embeddings, request_ids = self.get_embeddings(instance)
        
        # Cluster embeddings
        actual_clusters = min(self.n_clusters, len(request_ids))
        self._actual_clusters = actual_clusters #TM
        
        kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)
        
        # Build mapping
        cluster_map = {}
        for rid, label in zip(request_ids, labels):
            cluster_map[rid] = int(label)
            instance.requests[rid].cluster_id = int(label)
        
        return cluster_map
    
    def train(
        self,
        instances: List[PDPInstance],
        cluster_labels: Optional[List[Dict[int, int]]] = None,
        n_epochs: int = 100,
        lr: float = 0.01
    ):
        """
        Train GNN parameters (self-supervised or supervised).
        
        For self-supervised: uses contrastive learning to bring
        spatially close requests together in embedding space.
        
        Args:
            instances: Training instances.
            cluster_labels: Optional ground truth cluster labels.
            n_epochs: Number of training epochs.
            lr: Learning rate.
        """
        if not instances:
            return
        
        # Build graphs for all instances
        graphs = [self.graph_builder.build(inst) for inst in instances]
        
        # Initialize if needed
        if not self._initialized:
            self._initialize_layers(graphs[0].node_features.shape[1])
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for graph in graphs:
                # Forward pass
                h = graph.node_features
                activations = [h]
                
                for layer in self.layers:
                    h = layer.forward(h, graph.adjacency)
                    activations.append(h)
                
                # Compute contrastive loss
                # Positive pairs: connected nodes, negative: random
                loss, gradients = self._contrastive_loss(
                    h, graph.adjacency
                )
                total_loss += loss
                
                # Backward pass (simplified gradient descent)
                # For a proper implementation, use autograd
                grad = gradients
                for i in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[i]
                    h_prev = activations[i]
                    
                    # Gradient for W and b
                    # This is a simplified approximation
                    norm_adj = graph.adjacency / np.maximum(
                        graph.adjacency.sum(axis=1, keepdims=True), 1e-6
                    )
                    agg = norm_adj @ h_prev
                    
                    dW = agg.T @ grad
                    db = grad.sum(axis=0)
                    
                    layer.W -= lr * dW
                    layer.b -= lr * db
                    
                    # Propagate gradient (simplified)
                    grad = grad @ layer.W.T
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {total_loss:.4f}")
    
    def _contrastive_loss(
        self, 
        embeddings: np.ndarray,
        adjacency: np.ndarray,
        temperature: float = 0.1
    ) -> Tuple[float, np.ndarray]:
        """
        Compute contrastive loss for self-supervised learning.
        
        Similar to graph contrastive learning objectives.
        """
        n = embeddings.shape[0]
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        emb_norm = embeddings / norms
        
        # Compute similarity matrix
        similarity = emb_norm @ emb_norm.T / temperature
        
        # Mask for positive pairs (connected nodes)
        pos_mask = adjacency > 0
        np.fill_diagonal(pos_mask, False)
        
        # Log-softmax for each row
        exp_sim = np.exp(similarity - similarity.max(axis=1, keepdims=True))
        log_prob = similarity - np.log(exp_sim.sum(axis=1, keepdims=True) + 1e-6)
        
        # Loss: negative log likelihood of positive pairs
        pos_log_prob = log_prob * pos_mask
        num_pos = pos_mask.sum(axis=1) + 1e-6
        loss = -(pos_log_prob.sum(axis=1) / num_pos).mean()
        
        # Gradient (simplified)
        grad = emb_norm / temperature  # Approximate gradient
        
        return loss, grad
    
    def save(self, filepath: str):
        """Save model parameters."""
        data = {
            'hidden_dims': self.hidden_dims,
            'n_clusters': self.n_clusters,
            'layers': [(l.W, l.b) for l in self.layers]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load model parameters."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.hidden_dims = data['hidden_dims']
        self.n_clusters = data['n_clusters']
        
        # Rebuild layers
        self.layers = []
        for W, b in data['layers']:
            layer = SimpleGNNLayer(W.shape[0], W.shape[1])
            layer.W = W
            layer.b = b
            self.layers.append(layer)
        
        self._initialized = True

    def get_num_clusters(self) -> int:
        """Return how many clusters the last fit used."""
        if self._actual_clusters is not None:
            return self._actual_clusters
        return self.n_clusters


