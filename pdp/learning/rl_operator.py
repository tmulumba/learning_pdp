"""
Reinforcement Learning for ALNS Operator Selection.

Uses a contextual bandit / policy gradient approach to learn
which destroy-repair operator pairs to use based on solution state.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random

from ..core import Solution, PDPInstance
from ..alns import (
    ALNS, ALNSConfig, 
    DestroyOperator, RepairOperator,
    RandomRemoval, WorstRemoval, RelatedRemoval, ShawRemoval,
    GreedyRepair, RegretRepair
)


@dataclass
class SolutionState:
    """State representation of current solution for RL."""
    
    # Objective metrics
    total_travel_time: float
    efficiency: float
    num_routes: int
    
    # Search progress
    iterations_since_improvement: int
    current_temperature: float
    iteration_fraction: float  # current_iter / max_iter
    
    # Solution structure
    avg_route_length: float
    max_route_length: int
    min_route_length: int
    route_length_std: float
    
    # Recent history
    recent_improvements: int  # In last 10 iterations
    recent_accepts: int
    
    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.total_travel_time / 1000,  # Normalize
            self.efficiency,
            self.num_routes / 100,
            self.iterations_since_improvement / 100,
            self.current_temperature / 100,
            self.iteration_fraction,
            self.avg_route_length / 20,
            self.max_route_length / 50,
            self.min_route_length / 50,
            self.route_length_std / 10,
            self.recent_improvements / 10,
            self.recent_accepts / 10,
        ])
    
    @staticmethod
    def dim() -> int:
        """State dimension."""
        return 12


class OperatorSelectionEnv:
    """
    Environment for RL-based operator selection.
    
    Provides a gym-like interface for training operator selection policies.
    """
    
    def __init__(
        self,
        destroy_operators: List[DestroyOperator],
        repair_operators: List[RepairOperator],
        initial_solution: Solution,
        max_steps: int = 1000,
        removal_fraction: float = 0.15
    ):
        self.destroy_ops = destroy_operators
        self.repair_ops = repair_operators
        self.initial_solution = initial_solution
        self.max_steps = max_steps
        self.removal_fraction = removal_fraction
        
        self.num_destroy = len(destroy_operators)
        self.num_repair = len(repair_operators)
        self.action_dim = self.num_destroy * self.num_repair
        self.state_dim = SolutionState.dim()
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_solution = self.initial_solution.copy()
        self.best_solution = self.initial_solution.copy()
        self.best_obj = self._objective(self.best_solution)
        self.current_obj = self.best_obj
        
        self.step_count = 0
        self.iters_since_improvement = 0
        self.temperature = 100.0
        
        self.recent_history = deque(maxlen=10)
        
        return self._get_state().to_array()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer encoding (destroy_idx * num_repair + repair_idx).
            
        Returns:
            (next_state, reward, done, info)
        """
        # Decode action
        destroy_idx = action // self.num_repair
        repair_idx = action % self.num_repair
        
        destroy_op = self.destroy_ops[destroy_idx]
        repair_op = self.repair_ops[repair_idx]
        
        # Apply operators
        candidate = self.current_solution.copy()
        num_remove = int(len(candidate.get_all_request_ids()) * self.removal_fraction)
        num_remove = max(3, min(20, num_remove))
        
        candidate, removed = destroy_op(candidate, num_remove)
        candidate = repair_op(candidate, removed)
        
        # Evaluate
        candidate_obj = self._objective(candidate)
        delta = candidate_obj - self.current_obj
        
        # Acceptance and reward
        reward = 0.0
        accepted = False
        improved = False
        
        if candidate_obj < self.best_obj:
            # New global best
            self.best_solution = candidate.copy()
            self.best_obj = candidate_obj
            self.current_solution = candidate
            self.current_obj = candidate_obj
            reward = 10.0  # Big reward for new best
            accepted = True
            improved = True
            self.iters_since_improvement = 0
        elif candidate_obj < self.current_obj:
            # Improving
            self.current_solution = candidate
            self.current_obj = candidate_obj
            reward = 3.0
            accepted = True
            improved = True
            self.iters_since_improvement = 0
        elif self._accept_worse(delta):
            # Accept worse solution (exploration)
            self.current_solution = candidate
            self.current_obj = candidate_obj
            reward = 0.5
            accepted = True
            self.iters_since_improvement += 1
        else:
            # Reject
            reward = -0.1
            self.iters_since_improvement += 1
        
        # Update history
        self.recent_history.append({
            'improved': improved,
            'accepted': accepted,
            'delta': delta
        })
        
        # Update temperature
        self.temperature *= 0.995
        self.step_count += 1
        
        done = self.step_count >= self.max_steps
        
        info = {
            'destroy_op': destroy_op.name,
            'repair_op': repair_op.name,
            'accepted': accepted,
            'improved': improved,
            'best_obj': self.best_obj,
            'current_obj': self.current_obj,
        }
        
        return self._get_state().to_array(), reward, done, info
    
    def _objective(self, solution: Solution) -> float:
        """Calculate objective (lower is better)."""
        return solution.calculate_total_travel_time()
    
    def _accept_worse(self, delta: float) -> bool:
        """Simulated annealing acceptance."""
        if self.temperature <= 0:
            return False
        return random.random() < np.exp(-delta / self.temperature)
    
    def _get_state(self) -> SolutionState:
        """Get current state representation."""
        sol = self.current_solution
        
        route_lengths = [len(r) for r in sol.routes]
        
        recent_improvements = sum(1 for h in self.recent_history if h['improved'])
        recent_accepts = sum(1 for h in self.recent_history if h['accepted'])
        
        return SolutionState(
            total_travel_time=sol.calculate_total_travel_time(),
            efficiency=sol.calculate_efficiency(),
            num_routes=sol.num_routes,
            iterations_since_improvement=self.iters_since_improvement,
            current_temperature=self.temperature,
            iteration_fraction=self.step_count / self.max_steps,
            avg_route_length=np.mean(route_lengths) if route_lengths else 0,
            max_route_length=max(route_lengths) if route_lengths else 0,
            min_route_length=min(route_lengths) if route_lengths else 0,
            route_length_std=np.std(route_lengths) if route_lengths else 0,
            recent_improvements=recent_improvements,
            recent_accepts=recent_accepts,
        )


class RLOperatorSelector:
    """
    RL agent for selecting ALNS operators.
    
    Uses a simple policy network (or contextual bandit) to select
    destroy-repair pairs based on current solution state.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Simple linear policy (can be upgraded to neural network)
        self.weights = np.random.randn(state_dim, action_dim) * 0.01
        self.bias = np.zeros(action_dim)
        
        # Experience buffer
        self.buffer: List[Tuple] = []
        self.batch_size = 32
        
        # For tracking
        self.episode_rewards: List[float] = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State vector.
            training: Whether in training mode (uses exploration).
            
        Returns:
            Action index.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Compute action values
        logits = state @ self.weights + self.bias
        probs = self._softmax(logits)
        
        if training:
            return np.random.choice(self.action_dim, p=probs)
        else:
            return np.argmax(probs)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
        # Keep buffer bounded
        if len(self.buffer) > 10000:
            self.buffer = self.buffer[-10000:]
    
    def update(self) -> Optional[float]:
        """
        Update policy using stored transitions.
        
        Returns:
            Loss value if update performed, None otherwise.
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            # Compute current Q-value
            logits = state @ self.weights + self.bias
            q_current = logits[action]
            
            # Compute target
            if done:
                target = reward
            else:
                next_logits = next_state @ self.weights + self.bias
                target = reward + self.gamma * np.max(next_logits)
            
            # Update (simple gradient descent)
            error = target - q_current
            
            # Gradient update for linear model
            self.weights[:, action] += self.lr * error * state
            self.bias[action] += self.lr * error
            
            total_loss += error ** 2
        
        return total_loss / self.batch_size
    
    def train_episode(
        self,
        env: OperatorSelectionEnv,
        max_steps: int = 1000
    ) -> float:
        """
        Train for one episode.
        
        Returns:
            Total episode reward.
        """
        state = env.reset()
        total_reward = 0.0
        
        for step in range(max_steps):
            action = self.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            self.store_transition(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            # Update periodically
            if step % 10 == 0:
                self.update()
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return total_reward
    
    def save(self, filepath: str):
        """Save model weights."""
        np.savez(filepath, weights=self.weights, bias=self.bias)
    
    def load(self, filepath: str):
        """Load model weights."""
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']


class RLGuidedALNS(ALNS):
    """
    ALNS with RL-guided operator selection.
    
    Uses a trained RL agent instead of adaptive weights.
    """
    
    def __init__(
        self,
        destroy_operators: List[DestroyOperator],
        repair_operators: List[RepairOperator],
        rl_agent: RLOperatorSelector,
        config: Optional[ALNSConfig] = None
    ):
        super().__init__(destroy_operators, repair_operators, config)
        self.rl_agent = rl_agent
    
    def _select_operators(
        self, 
        state: SolutionState
    ) -> Tuple[int, int]:
        """Select operators using RL agent."""
        state_arr = state.to_array()
        action = self.rl_agent.select_action(state_arr, training=False)
        
        destroy_idx = action // len(self.repair_ops)
        repair_idx = action % len(self.repair_ops)
        
        return destroy_idx, repair_idx


def create_default_rl_alns(
    avg_duration_limit: float = 45.0,
    model_path: Optional[str] = None
) -> Tuple[List[DestroyOperator], List[RepairOperator], RLOperatorSelector]:
    """
    Create default operators and RL agent.
    
    Returns:
        Tuple of (destroy_ops, repair_ops, rl_agent)
    """
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
    
    state_dim = SolutionState.dim()
    action_dim = len(destroy_ops) * len(repair_ops)
    
    agent = RLOperatorSelector(state_dim, action_dim)
    
    if model_path:
        agent.load(model_path)
    
    return destroy_ops, repair_ops, agent

