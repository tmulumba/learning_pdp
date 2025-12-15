"""
Synthetic Instance Generator for PDP Benchmarks.

Generates instances matching the statistical properties of real-world
food delivery operations while being fully shareable for reproducibility.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import json
import os

from .core import Request, PDPInstance


@dataclass
class GeneratorConfig:
    """Configuration for synthetic instance generation."""
    
    # Geometry
    bbox_size_km: float = 10.0  # Bounding box side length
    depot_at_center: bool = True
    min_separation_m: float = 100.0  # Min distance between locations
    
    # Travel model
    dasher_speed_kmh: float = 16.2  # ~4.5 m/s
    drone_speed_kmh: float = 60.0
    detour_mean: float = 1.25  # Road detour factor mean
    detour_std: float = 0.1
    
    # Service times (minutes)
    service_time_min: float = 2.0
    service_time_max: float = 5.0
    
    # Requests
    num_requests_min: int = 100
    num_requests_max: int = 200
    num_requests: Optional[int] = None  # If set, overrides min/max
    
    # Payload (kg)
    payload_mean_log: float = 1.2
    payload_std_log: float = 0.5
    payload_min: float = 0.2
    payload_max: float = 8.0
    
    # Drone eligibility
    drone_max_payload_kg: float = 5.0
    drone_max_distance_km: float = 7.0
    
    # Time windows
    time_horizon_min: float = 300.0  # Total horizon (minutes)
    tw_start_max: float = 300.0  # Max start time
    tw_slack_min: float = 20.0
    tw_slack_mode: float = 60.0
    tw_slack_max: float = 120.0
    tight_tw_fraction: float = 0.2  # Fraction with halved slack
    
    # Food ready time offset from tw start (minutes)
    food_ready_offset_min: float = 5.0
    food_ready_offset_max: float = 25.0
    
    # Capacity
    truck_capacity: int = 200
    drone_endurance_options: List[int] = field(default_factory=lambda: [15, 20, 25])
    
    # Dynamic reveal (optional)
    enable_dynamic: bool = False
    num_epochs_min: int = 5
    num_epochs_max: int = 9
    reveal_rate_mean: float = 18.0  # Poisson mean per epoch
    max_active_requests: int = 100
    
    # Objective
    drone_cost_factor: float = 0.3
    penalty_scale_min: float = 1.0
    penalty_scale_max: float = 3.0
    
    # Random seed
    seed: Optional[int] = None
    
    # Output
    region_id: int = 1


class SyntheticGenerator:
    """
    Generates synthetic PDP instances matching real-world statistics.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Set up coordinate system
        # Using a local coordinate system centered at depot
        self.center_lat = 37.45  # Example: Bay Area
        self.center_lon = -122.15
        
        # Conversion factors (approximate at this latitude)
        self.km_per_lat = 111.0
        self.km_per_lon = 111.0 * np.cos(np.radians(self.center_lat))
    
    def _km_to_lat(self, km: float) -> float:
        """Convert km to latitude degrees."""
        return km / self.km_per_lat
    
    def _km_to_lon(self, km: float) -> float:
        """Convert km to longitude degrees."""
        return km / self.km_per_lon
    
    def _sample_location(self) -> Tuple[float, float]:
        """Sample a random location within bounding box."""
        cfg = self.config
        half_size = cfg.bbox_size_km / 2
        
        # Sample in km from center
        x = self.rng.uniform(-half_size, half_size)
        y = self.rng.uniform(-half_size, half_size)
        
        # Convert to lat/lon
        lat = self.center_lat + self._km_to_lat(y)
        lon = self.center_lon + self._km_to_lon(x)
        
        return lat, lon
    
    def _sample_separated_locations(
        self, 
        n: int
    ) -> List[Tuple[float, float]]:
        """Sample n locations with minimum separation."""
        cfg = self.config
        min_sep_km = cfg.min_separation_m / 1000
        
        locations = []
        max_attempts = n * 100
        attempts = 0
        
        while len(locations) < n and attempts < max_attempts:
            lat, lon = self._sample_location()
            
            # Check separation from existing locations
            valid = True
            for other_lat, other_lon in locations:
                dist = np.sqrt(
                    (self._km_to_lat(1) * (lat - other_lat)) ** 2 +
                    (self._km_to_lon(1) * (lon - other_lon)) ** 2
                ) * self.km_per_lat  # Approximate distance
                
                if dist < min_sep_km:
                    valid = False
                    break
            
            if valid:
                locations.append((lat, lon))
            
            attempts += 1
        
        # If we couldn't get enough separated locations, fill randomly
        while len(locations) < n:
            locations.append(self._sample_location())
        
        return locations
    
    def _sample_time_window(self) -> Tuple[float, float, float]:
        """
        Sample time window and food ready time.
        
        Returns:
            (start_time, end_time, food_ready_time) in minutes from epoch
        """
        cfg = self.config
        
        # Start time
        start = self.rng.uniform(0, cfg.tw_start_max)
        
        # Slack (triangular distribution)
        slack = self.rng.triangular(
            cfg.tw_slack_min, 
            cfg.tw_slack_mode, 
            cfg.tw_slack_max
        )
        
        # Tighten some windows
        if self.rng.random() < cfg.tight_tw_fraction:
            slack /= 2
        
        end = start + slack
        
        # Food ready time
        food_ready_offset = self.rng.uniform(
            cfg.food_ready_offset_min,
            cfg.food_ready_offset_max
        )
        food_ready = start + food_ready_offset
        
        return start, end, food_ready
    
    def _sample_payload(self) -> float:
        """Sample payload weight (kg)."""
        cfg = self.config
        
        # Log-normal distribution
        payload = self.rng.lognormal(
            cfg.payload_mean_log,
            cfg.payload_std_log
        )
        
        # Truncate
        payload = np.clip(payload, cfg.payload_min, cfg.payload_max)
        
        return payload
    
    def _is_drone_eligible(
        self, 
        payload: float, 
        pickup: Tuple[float, float],
        dropoff: Tuple[float, float]
    ) -> bool:
        """Check if request is drone-eligible."""
        cfg = self.config
        
        if payload > cfg.drone_max_payload_kg:
            return False
        
        # Calculate distance
        from .core import haversine
        dist = haversine(pickup[0], pickup[1], dropoff[0], dropoff[1])
        
        return dist <= cfg.drone_max_distance_km
    
    def _compute_features(
        self,
        request_id: int,
        pickup: Tuple[float, float],
        dropoff: Tuple[float, float],
        payload: float,
        drone_eligible: bool,
        all_requests: List[Dict]
    ) -> Dict:
        """Compute precomputed features for ML/RL."""
        from .core import haversine
        
        # Distance
        dist = haversine(pickup[0], pickup[1], dropoff[0], dropoff[1])
        
        # 5-NN density (if we have enough requests)
        if len(all_requests) >= 5:
            distances = []
            for other in all_requests:
                if other['delivery_id'] != request_id:
                    d = haversine(
                        pickup[0], pickup[1],
                        other['pickup_lat'], other['pickup_long']
                    )
                    distances.append(d)
            distances.sort()
            density_5nn = np.mean(distances[:5]) if len(distances) >= 5 else 0.0
        else:
            density_5nn = 0.0
        
        # Detour ratio (simplified)
        detour_ratio = self.rng.normal(
            self.config.detour_mean,
            self.config.detour_std
        )
        detour_ratio = max(1.0, detour_ratio)
        
        return {
            'drone_eligible': drone_eligible,
            'distance_km': dist,
            'density_5nn': density_5nn,
            'detour_ratio': detour_ratio,
            'payload_kg': payload,
        }
    
    def generate(self, num_requests: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a synthetic PDP instance.
        
        Args:
            num_requests: Number of requests (overrides config if provided).
            
        Returns:
            DataFrame with columns matching real data format.
        """
        cfg = self.config
        
        # Determine number of requests
        if num_requests is not None:
            n = num_requests
        elif cfg.num_requests is not None:
            n = cfg.num_requests
        else:
            n = self.rng.integers(cfg.num_requests_min, cfg.num_requests_max + 1)
        
        # Sample all locations (pickup + dropoff for each request)
        all_locations = self._sample_separated_locations(n * 2)
        pickups = all_locations[:n]
        dropoffs = all_locations[n:]
        
        # Generate requests
        requests = []
        epoch = datetime(2015, 2, 3, 2, 0, 0)  # Match original data format
        
        for i in range(n):
            pickup = pickups[i]
            dropoff = dropoffs[i]
            
            # Time window
            tw_start, tw_end, food_ready = self._sample_time_window()
            
            # Created at is tw_start (order placed)
            created_at = epoch + timedelta(minutes=tw_start)
            food_ready_time = epoch + timedelta(minutes=food_ready)
            
            # Payload
            payload = self._sample_payload()
            
            # Drone eligibility
            drone_eligible = self._is_drone_eligible(payload, pickup, dropoff)
            
            request = {
                'delivery_id': i + 1,
                'created_at': created_at.strftime('%m/%d/%y %H:%M'),
                'food_ready_time': food_ready_time.strftime('%m/%d/%y %H:%M'),
                'region_id': cfg.region_id,
                'pickup_lat': pickup[0],
                'pickup_long': pickup[1],
                'dropoff_lat': dropoff[0],
                'dropoff_long': dropoff[1],
            }
            requests.append(request)
        
        df = pd.DataFrame(requests)
        
        # Sort by created_at for consistency
        df = df.sort_values('delivery_id').reset_index(drop=True)
        
        return df
    
    def generate_with_features(
        self, 
        num_requests: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate instance with precomputed features.
        
        Returns:
            Tuple of (instance_df, features_df)
        """
        instance_df = self.generate(num_requests)
        
        # Compute features
        features = []
        requests_list = instance_df.to_dict('records')
        
        for req in requests_list:
            pickup = (req['pickup_lat'], req['pickup_long'])
            dropoff = (req['dropoff_lat'], req['dropoff_long'])
            
            # Sample payload (not in output but needed for features)
            payload = self._sample_payload()
            drone_eligible = self._is_drone_eligible(payload, pickup, dropoff)
            
            feat = self._compute_features(
                req['delivery_id'],
                pickup,
                dropoff,
                payload,
                drone_eligible,
                requests_list
            )
            feat['delivery_id'] = req['delivery_id']
            features.append(feat)
        
        features_df = pd.DataFrame(features)
        
        return instance_df, features_df
    
    def generate_dynamic(self) -> List[pd.DataFrame]:
        """
        Generate dynamic instance with requests revealed over epochs.
        
        Returns:
            List of DataFrames, one per epoch.
        """
        cfg = self.config
        
        # Generate full instance
        full_df = self.generate()
        n = len(full_df)
        
        # Number of epochs
        num_epochs = self.rng.integers(
            cfg.num_epochs_min, 
            cfg.num_epochs_max + 1
        )
        
        # Assign requests to epochs based on food ready time
        full_df['epoch'] = 0
        
        # Sort by food ready time
        full_df = full_df.sort_values('food_ready_time').reset_index(drop=True)
        
        # Distribute across epochs
        epoch_dfs = []
        remaining_indices = list(range(n))
        
        for epoch in range(num_epochs):
            # How many to reveal this epoch
            reveal_count = self.rng.poisson(cfg.reveal_rate_mean)
            reveal_count = min(reveal_count, len(remaining_indices))
            reveal_count = min(reveal_count, cfg.max_active_requests)
            
            if reveal_count > 0:
                # Take from remaining (approximately in time order)
                epoch_indices = remaining_indices[:reveal_count]
                remaining_indices = remaining_indices[reveal_count:]
                
                epoch_df = full_df.iloc[epoch_indices].copy()
                epoch_df['epoch'] = epoch
                epoch_dfs.append(epoch_df)
        
        # Add any remaining to last epoch
        if remaining_indices:
            last_df = full_df.iloc[remaining_indices].copy()
            last_df['epoch'] = num_epochs - 1
            if epoch_dfs:
                epoch_dfs[-1] = pd.concat([epoch_dfs[-1], last_df])
            else:
                epoch_dfs.append(last_df)
        
        return epoch_dfs
    
    def to_pdp_instance(self, df: pd.DataFrame) -> PDPInstance:
        """Convert DataFrame to PDPInstance."""
        return PDPInstance.from_dataframe(df)
    
    def save_instance(
        self, 
        df: pd.DataFrame, 
        filepath: str,
        config_filepath: Optional[str] = None
    ):
        """Save instance to CSV and optionally save config."""
        df.to_csv(filepath, index=False)
        
        if config_filepath:
            # Save config as JSON
            config_dict = {
                k: v for k, v in self.config.__dict__.items()
                if not k.startswith('_')
            }
            # Handle non-serializable types
            for k, v in config_dict.items():
                if isinstance(v, np.ndarray):
                    config_dict[k] = v.tolist()
            
            with open(config_filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)


def generate_benchmark_suite(
    output_dir: str,
    num_instances: int = 50,
    sizes: List[int] = [100, 150, 200],
    seed: int = 42
) -> List[str]:
    """
    Generate a suite of benchmark instances.
    
    Args:
        output_dir: Directory to save instances.
        num_instances: Total number of instances.
        sizes: List of instance sizes to generate.
        seed: Random seed.
        
    Returns:
        List of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = []
    rng = np.random.default_rng(seed)
    
    instances_per_size = num_instances // len(sizes)
    
    for size in sizes:
        for i in range(instances_per_size):
            # Create generator with unique seed
            instance_seed = rng.integers(0, 2**31)
            config = GeneratorConfig(
                num_requests=size,
                seed=instance_seed
            )
            generator = SyntheticGenerator(config)
            
            # Generate and save
            df = generator.generate()
            filename = f"instance_n{size}_{i+1:03d}.csv"
            filepath = os.path.join(output_dir, filename)
            generator.save_instance(df, filepath)
            files.append(filepath)
    
    # Save metadata
    metadata = {
        'num_instances': len(files),
        'sizes': sizes,
        'seed': seed,
        'files': [os.path.basename(f) for f in files]
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return files


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic PDP instances')
    parser.add_argument('-n', '--num-requests', type=int, default=None,
                       help='Number of requests')
    parser.add_argument('-o', '--output', default='synthetic_instance.csv',
                       help='Output file path')
    parser.add_argument('-s', '--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--suite', action='store_true',
                       help='Generate benchmark suite instead of single instance')
    parser.add_argument('--suite-dir', default='benchmark_instances',
                       help='Directory for benchmark suite')
    parser.add_argument('--suite-count', type=int, default=50,
                       help='Number of instances in suite')
    
    args = parser.parse_args()
    
    if args.suite:
        files = generate_benchmark_suite(
            output_dir=args.suite_dir,
            num_instances=args.suite_count,
            seed=args.seed or 42
        )
        print(f"Generated {len(files)} instances in {args.suite_dir}/")
    else:
        config = GeneratorConfig(
            num_requests=args.num_requests,
            seed=args.seed
        )
        generator = SyntheticGenerator(config)
        df = generator.generate()
        generator.save_instance(df, args.output)
        print(f"Generated instance with {len(df)} requests: {args.output}")

