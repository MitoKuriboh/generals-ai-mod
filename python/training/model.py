"""
Neural Network Model for Generals Zero Hour Learning AI

Architecture:
- Input: Game state features (44 floats)
- Hidden: 2 fully-connected layers with ReLU
- Output: Action probabilities + value estimate

The model uses an actor-critic architecture for PPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


# State feature dimensions
STATE_DIM = 44  # Total features from MLGameState

# Action space dimensions
ACTION_DIMS = {
    'priority_economy': 1,    # 0-1 continuous
    'priority_defense': 1,
    'priority_military': 1,
    'priority_tech': 1,
    'prefer_infantry': 1,
    'prefer_vehicles': 1,
    'prefer_aircraft': 1,
    'aggression': 1,
}
TOTAL_ACTION_DIM = sum(ACTION_DIMS.values())  # 8 continuous outputs


def state_dict_to_tensor(state: Dict) -> torch.Tensor:
    """Convert game state dictionary to tensor."""
    features = []

    # Economy (4)
    features.append(state.get('money', 0))
    features.append(state.get('power', 0) / 100.0)  # Normalize
    features.append(state.get('income', 0) / 10.0)
    features.append(state.get('supply', 0.5))

    # Own forces (12) - 3 features per category
    for key in ['own_infantry', 'own_vehicles', 'own_aircraft', 'own_structures']:
        arr = state.get(key, [0, 0, 0])
        features.extend(arr[:3] if len(arr) >= 3 else arr + [0] * (3 - len(arr)))

    # Enemy forces (12)
    for key in ['enemy_infantry', 'enemy_vehicles', 'enemy_aircraft', 'enemy_structures']:
        arr = state.get(key, [0, 0, 0])
        features.extend(arr[:3] if len(arr) >= 3 else arr + [0] * (3 - len(arr)))

    # Strategic (8)
    features.append(state.get('game_time', 0) / 30.0)  # Normalize to ~1 at 30 min
    features.append(state.get('tech_level', 0))
    features.append(state.get('base_threat', 0))
    features.append(state.get('army_strength', 1.0) / 2.0)  # Normalize
    features.append(state.get('under_attack', 0))
    features.append(state.get('distance_to_enemy', 0.5))
    features.append(0.0)  # Padding
    features.append(0.0)  # Padding

    # Ensure correct length
    while len(features) < STATE_DIM:
        features.append(0.0)
    features = features[:STATE_DIM]

    return torch.tensor(features, dtype=torch.float32)


def action_tensor_to_dict(action: torch.Tensor) -> Dict:
    """Convert action tensor to recommendation dictionary."""
    # Action tensor has 8 values, all sigmoid-activated (0-1)
    a = action.detach().cpu().numpy()

    # Build priorities (normalize to sum to 1)
    priorities = np.array([a[0], a[1], a[2], a[3]])
    priorities = np.maximum(priorities, 0.05)  # Minimum 5%
    priorities = priorities / priorities.sum()

    # Army preferences (normalize to sum to 1)
    army = np.array([a[4], a[5], a[6]])
    army = np.maximum(army, 0.1)  # Minimum 10%
    army = army / army.sum()

    return {
        'priority_economy': float(priorities[0]),
        'priority_defense': float(priorities[1]),
        'priority_military': float(priorities[2]),
        'priority_tech': float(priorities[3]),
        'prefer_infantry': float(army[0]),
        'prefer_vehicles': float(army[1]),
        'prefer_aircraft': float(army[2]),
        'aggression': float(np.clip(a[7], 0.0, 1.0)),
        'target_player': -1
    }


class PolicyNetwork(nn.Module):
    """
    Actor-Critic Policy Network for PPO.

    Uses continuous action space with Beta distribution for bounded outputs.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = TOTAL_ACTION_DIM,
                 hidden_dim: int = 128):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy) - outputs mean and log_std for each action
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Smaller initialization for output layers
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Batch of states [batch_size, state_dim]

        Returns:
            action_mean: Mean of action distribution [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
        """
        features = self.shared(state)
        action_mean = torch.sigmoid(self.actor_mean(features))  # Bound to [0, 1]
        value = self.critic(features)
        return action_mean, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            deterministic: If True, return mean action instead of sampling

        Returns:
            action: Sampled action [batch_size, action_dim]
            log_prob: Log probability of action
            value: State value estimate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_mean, value = self.forward(state)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(state.size(0), device=state.device)
        else:
            # Use Beta distribution for bounded continuous actions
            # Reparameterize: action_mean is the mode, std controls spread
            std = torch.exp(self.actor_log_std).expand_as(action_mean)
            std = torch.clamp(std, min=0.01, max=0.5)

            # Normal distribution, then sigmoid to bound
            dist = torch.distributions.Normal(action_mean, std)
            action_raw = dist.rsample()
            action = torch.sigmoid(action_raw)

            # Log probability with correction for sigmoid transform
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            # Correction for sigmoid (change of variables)
            log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]

        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        action_mean, values = self.forward(states)

        std = torch.exp(self.actor_log_std).expand_as(action_mean)
        std = torch.clamp(std, min=0.01, max=0.5)

        # Inverse sigmoid to get raw action
        actions_clamped = torch.clamp(actions, 0.001, 0.999)
        action_raw = torch.log(actions_clamped / (1 - actions_clamped))

        dist = torch.distributions.Normal(action_mean, std)
        log_probs = dist.log_prob(action_raw).sum(dim=-1)
        log_probs -= torch.log(actions_clamped * (1 - actions_clamped) + 1e-8).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'PolicyNetwork':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(
            state_dim=checkpoint.get('state_dim', STATE_DIM),
            action_dim=checkpoint.get('action_dim', TOTAL_ACTION_DIM)
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


class ReplayBuffer:
    """
    Simple replay buffer for storing trajectories.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value,
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        import random
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def get_all(self):
        """Get all stored transitions."""
        return self.buffer[:len(self.buffer)]

    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    # Test the model
    print("Testing PolicyNetwork...")

    model = PolicyNetwork()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with random state
    state = torch.randn(STATE_DIM)
    action, log_prob, value = model.get_action(state)

    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action values: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value}")

    # Convert to recommendation dict
    rec = action_tensor_to_dict(action)
    print(f"\nRecommendation:")
    for k, v in rec.items():
        print(f"  {k}: {v:.3f}")

    # Test batch processing
    states = torch.randn(32, STATE_DIM)
    actions = torch.rand(32, TOTAL_ACTION_DIM)
    log_probs, values, entropy = model.evaluate_actions(states, actions)
    print(f"\nBatch test - log_probs: {log_probs.shape}, values: {values.shape}, entropy: {entropy.mean():.3f}")

    print("\nModel test passed!")
