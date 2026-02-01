"""
Tactical Network for Team-Level Decision Making

Architecture:
- Input: 64 floats (team state + strategic embedding)
- Hidden: 2x128 MLP with LayerNorm
- Output: Hybrid action space
  - Discrete: 8 tactical actions (Categorical)
  - Continuous: target_x, target_y, attitude (Beta distributions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Tuple, Dict, Optional
import numpy as np


# State and action dimensions
TACTICAL_STATE_DIM = 64
TACTICAL_ACTION_DIM = 8  # Discrete actions
TACTICAL_CONTINUOUS_DIM = 3  # target_x, target_y, attitude


class TacticalAction:
    """Tactical action types."""
    ATTACK_MOVE = 0       # Attack-move to position
    ATTACK_TARGET = 1     # Focus on specific target
    DEFEND_POSITION = 2   # Guard location
    RETREAT = 3           # Fall back to base
    HOLD = 4              # Hold position
    HUNT = 5              # Seek and destroy
    REINFORCE = 6         # Merge with another team
    SPECIAL = 7           # Use special ability

    NAMES = [
        'ATTACK_MOVE', 'ATTACK_TARGET', 'DEFEND_POSITION', 'RETREAT',
        'HOLD', 'HUNT', 'REINFORCE', 'SPECIAL'
    ]

    @classmethod
    def name(cls, action_id: int) -> str:
        if 0 <= action_id < len(cls.NAMES):
            return cls.NAMES[action_id]
        return f'UNKNOWN_{action_id}'


class TacticalNetwork(nn.Module):
    """
    Actor-Critic network for tactical layer.

    Uses hybrid action space:
    - Categorical distribution for discrete action selection
    - Beta distributions for continuous parameters (bounded [0,1])
    """

    def __init__(self, state_dim: int = TACTICAL_STATE_DIM,
                 num_actions: int = TACTICAL_ACTION_DIM,
                 hidden_dim: int = 128):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Discrete action head (Categorical)
        self.action_logits = nn.Linear(hidden_dim, num_actions)

        # Continuous parameter heads (Beta distributions for position)
        # Position (x, y) - 2 dims
        self.pos_alpha = nn.Linear(hidden_dim, 2)
        self.pos_beta = nn.Linear(hidden_dim, 2)

        # Attitude - 1 dim (0=passive, 1=aggressive)
        self.attitude_alpha = nn.Linear(hidden_dim, 1)
        self.attitude_beta = nn.Linear(hidden_dim, 1)

        # Critic head
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Initialize action logits to uniform distribution
        nn.init.orthogonal_(self.action_logits.weight, gain=0.01)
        nn.init.constant_(self.action_logits.bias, 0)

        # Initialize Beta params for mode at 0.5
        for param_layer in [self.pos_alpha, self.pos_beta,
                           self.attitude_alpha, self.attitude_beta]:
            nn.init.orthogonal_(param_layer.weight, gain=0.01)
            nn.init.constant_(param_layer.bias, 1.0)

        # Initialize value head
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]

        Returns:
            action_logits: [batch_size, num_actions]
            pos_alpha, pos_beta: [batch_size, 2] each
            att_alpha, att_beta: [batch_size, 1] each
            value: [batch_size, 1]
        """
        features = self.shared(state)

        action_logits = self.action_logits(features)

        # Softplus + offset to ensure valid Beta parameters
        pos_alpha = F.softplus(self.pos_alpha(features)) + 1.0
        pos_beta = F.softplus(self.pos_beta(features)) + 1.0
        att_alpha = F.softplus(self.attitude_alpha(features)) + 1.0
        att_beta = F.softplus(self.attitude_beta(features)) + 1.0

        value = self.value(features)

        return action_logits, pos_alpha, pos_beta, att_alpha, att_beta, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor [state_dim] or [batch_size, state_dim]
            deterministic: If True, return mode of distributions

        Returns:
            action_dict: {'action': int, 'target_x': float, 'target_y': float, 'attitude': float}
            log_prob: Total log probability of the action
            value: State value estimate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_logits, pos_alpha, pos_beta, att_alpha, att_beta, value = self.forward(state)

        # Discrete action
        action_dist = Categorical(logits=action_logits)

        # Continuous parameters
        pos_dist = Beta(pos_alpha, pos_beta)
        att_dist = Beta(att_alpha, att_beta)

        if deterministic:
            # Take mode
            action = action_logits.argmax(dim=-1)
            pos = (pos_alpha - 1) / (pos_alpha + pos_beta - 2 + 1e-8)
            pos = pos.clamp(0, 1)
            attitude = (att_alpha - 1) / (att_alpha + att_beta - 2 + 1e-8)
            attitude = attitude.clamp(0, 1)
            log_prob = torch.zeros(state.size(0), device=state.device)
        else:
            action = action_dist.sample()
            pos_raw = pos_dist.rsample()
            attitude_raw = att_dist.rsample()

            # Compute log probs
            log_prob_action = action_dist.log_prob(action)
            log_prob_pos = pos_dist.log_prob(pos_raw).sum(dim=-1)
            log_prob_att = att_dist.log_prob(attitude_raw).squeeze(-1)

            log_prob = log_prob_action + log_prob_pos + log_prob_att
            log_prob = log_prob.clamp(min=-100.0)

            pos = pos_raw.clamp(1e-7, 1 - 1e-7)
            attitude = attitude_raw.clamp(1e-7, 1 - 1e-7)

        action_dict = {
            'action': action.squeeze(0),
            'target_x': pos[:, 0].squeeze(0),
            'target_y': pos[:, 1].squeeze(0),
            'attitude': attitude.squeeze(-1).squeeze(0),
        }

        return action_dict, log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor,
                         positions: torch.Tensor, attitudes: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size] discrete actions
            positions: [batch_size, 2] target positions
            attitudes: [batch_size] attitude values

        Returns:
            log_probs: [batch_size]
            values: [batch_size]
            entropy: [batch_size]
        """
        action_logits, pos_alpha, pos_beta, att_alpha, att_beta, values = self.forward(states)

        action_dist = Categorical(logits=action_logits)
        pos_dist = Beta(pos_alpha, pos_beta)
        att_dist = Beta(att_alpha, att_beta)

        # Clamp for numerical stability
        positions = positions.clamp(1e-7, 1 - 1e-7)
        attitudes = attitudes.clamp(1e-7, 1 - 1e-7)

        log_prob_action = action_dist.log_prob(actions)
        log_prob_pos = pos_dist.log_prob(positions).sum(dim=-1)
        log_prob_att = att_dist.log_prob(attitudes.unsqueeze(-1)).squeeze(-1)

        log_probs = log_prob_action + log_prob_pos + log_prob_att
        log_probs = log_probs.clamp(min=-100.0)

        entropy = action_dist.entropy() + pos_dist.entropy().sum(dim=-1) + att_dist.entropy().squeeze(-1)

        return log_probs, values.squeeze(-1), entropy

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'hidden_dim': self.hidden_dim,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'TacticalNetwork':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(
            state_dim=checkpoint.get('state_dim', TACTICAL_STATE_DIM),
            num_actions=checkpoint.get('num_actions', TACTICAL_ACTION_DIM),
            hidden_dim=checkpoint.get('hidden_dim', 128),
        )
        # Handle both checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'policy_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['policy_state_dict'])
        else:
            raise KeyError(f"Checkpoint missing state_dict. Keys: {checkpoint.keys()}")
        return model


def action_dict_to_command(action_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Convert action dict to JSON-serializable command."""
    return {
        'action': int(action_dict['action'].item()) if hasattr(action_dict['action'], 'item') else int(action_dict['action']),
        'target_x': float(action_dict['target_x'].item()) if hasattr(action_dict['target_x'], 'item') else float(action_dict['target_x']),
        'target_y': float(action_dict['target_y'].item()) if hasattr(action_dict['target_y'], 'item') else float(action_dict['target_y']),
        'attitude': float(action_dict['attitude'].item()) if hasattr(action_dict['attitude'], 'item') else float(action_dict['attitude']),
    }


if __name__ == '__main__':
    print("Testing TacticalNetwork...")

    model = TacticalNetwork()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    state = torch.randn(TACTICAL_STATE_DIM)
    action_dict, log_prob, value = model.get_action(state)

    print(f"\nAction: {TacticalAction.name(action_dict['action'].item())}")
    print(f"Target: ({action_dict['target_x']:.3f}, {action_dict['target_y']:.3f})")
    print(f"Attitude: {action_dict['attitude']:.3f}")
    print(f"Log prob: {log_prob:.3f}")
    print(f"Value: {value.item():.3f}")

    # Test batch processing
    states = torch.randn(32, TACTICAL_STATE_DIM)
    actions = torch.randint(0, TACTICAL_ACTION_DIM, (32,))
    positions = torch.rand(32, 2)
    attitudes = torch.rand(32)

    log_probs, values, entropy = model.evaluate_actions(states, actions, positions, attitudes)
    print(f"\nBatch test - log_probs: {log_probs.shape}, entropy: {entropy.mean():.3f}")

    # Test deterministic
    action_dict_det, _, _ = model.get_action(state, deterministic=True)
    print(f"\nDeterministic action: {TacticalAction.name(action_dict_det['action'].item())}")

    print("\nTacticalNetwork test passed!")
