"""
Micro Network for Unit-Level Control

Architecture:
- Input: 32 floats (unit state + team objective)
- LSTM: 64 hidden units for temporal coherence
- Output: Hybrid action space
  - Discrete: 11 micro actions (Categorical)
  - Continuous: move_angle, move_distance (Beta distributions)

The LSTM allows the network to remember recent actions and enemy positions,
enabling coherent kiting and target tracking behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Tuple, Dict, Optional
import numpy as np


# State and action dimensions
MICRO_STATE_DIM = 32
MICRO_ACTION_DIM = 11  # Discrete actions
MICRO_CONTINUOUS_DIM = 2  # move_angle, move_distance


class MicroAction:
    """Micro action types."""
    ATTACK_CURRENT = 0     # Continue current target
    ATTACK_NEAREST = 1     # Switch to nearest enemy
    ATTACK_WEAKEST = 2     # Focus weakest enemy
    ATTACK_PRIORITY = 3    # High-value target
    MOVE_FORWARD = 4       # Advance toward enemy
    MOVE_BACKWARD = 5      # Kite backward
    MOVE_FLANK = 6         # Circle strafe
    HOLD_FIRE = 7          # Stealth/hold position
    USE_ABILITY = 8        # Special power
    RETREAT = 9            # Full retreat
    FOLLOW_TEAM = 10       # Default team behavior

    NAMES = [
        'ATTACK_CURRENT', 'ATTACK_NEAREST', 'ATTACK_WEAKEST', 'ATTACK_PRIORITY',
        'MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_FLANK', 'HOLD_FIRE',
        'USE_ABILITY', 'RETREAT', 'FOLLOW_TEAM'
    ]

    @classmethod
    def name(cls, action_id: int) -> str:
        if 0 <= action_id < len(cls.NAMES):
            return cls.NAMES[action_id]
        return f'UNKNOWN_{action_id}'


class MicroNetwork(nn.Module):
    """
    Actor-Critic network for micro layer with LSTM for temporal coherence.

    The LSTM maintains hidden state across time steps, allowing the network
    to track enemy movements and execute multi-step maneuvers like kiting.
    """

    def __init__(self, state_dim: int = MICRO_STATE_DIM,
                 num_actions: int = MICRO_ACTION_DIM,
                 hidden_dim: int = 64,
                 lstm_layers: int = 1):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # Input processing
        self.input_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # LSTM for temporal coherence
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Discrete action head (Categorical)
        self.action_logits = nn.Linear(hidden_dim, num_actions)

        # Continuous parameter heads (Beta distributions)
        # angle: 0-1 mapped to -pi to pi
        # distance: 0-1 mapped to 0 to max_move_distance
        self.move_alpha = nn.Linear(hidden_dim, 2)
        self.move_beta = nn.Linear(hidden_dim, 2)

        # Critic head
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Hidden state storage
        self.hidden = None

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # LSTM initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        # Initialize Beta params for uniform [0,1]
        nn.init.orthogonal_(self.move_alpha.weight, gain=0.01)
        nn.init.constant_(self.move_alpha.bias, 1.0)
        nn.init.orthogonal_(self.move_beta.weight, gain=0.01)
        nn.init.constant_(self.move_beta.bias, 1.0)

        # Initialize value head
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset LSTM hidden state."""
        if device is None:
            device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        )

    def forward(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through network.

        Args:
            state: [batch_size, state_dim] or [batch_size, seq_len, state_dim]
            hidden: Optional LSTM hidden state tuple (h, c)

        Returns:
            action_logits, move_alpha, move_beta, value, new_hidden
        """
        # Handle different input shapes
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        elif state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        batch_size, seq_len, _ = state.shape

        # Input processing
        x = self.input_fc(state)  # [batch, seq, hidden]

        # LSTM
        if hidden is None:
            if self.hidden is None or self.hidden[0].size(1) != batch_size:
                self.reset_hidden(batch_size, state.device)
            hidden = self.hidden

        lstm_out, new_hidden = self.lstm(x, hidden)
        self.hidden = new_hidden

        # Use last time step for outputs
        features = lstm_out[:, -1, :]  # [batch, hidden]

        action_logits = self.action_logits(features)

        move_alpha = F.softplus(self.move_alpha(features)) + 1.0
        move_beta = F.softplus(self.move_beta(features)) + 1.0

        value = self.value(features)

        return action_logits, move_alpha, move_beta, value, new_hidden

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor [state_dim] or [batch_size, state_dim]
            deterministic: If True, return mode of distributions

        Returns:
            action_dict: {'action': int, 'move_angle': float, 'move_distance': float}
            log_prob: Total log probability
            value: State value estimate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.size(0)
        if self.hidden is not None and self.hidden[0].size(1) != batch_size:
            self.reset_hidden(batch_size, state.device)

        action_logits, move_alpha, move_beta, value, _ = self.forward(state)

        action_dist = Categorical(logits=action_logits)
        move_dist = Beta(move_alpha, move_beta)

        if deterministic:
            action = action_logits.argmax(dim=-1)
            move = (move_alpha - 1) / (move_alpha + move_beta - 2 + 1e-8)
            move = move.clamp(0, 1)
            log_prob = torch.zeros(state.size(0), device=state.device)
        else:
            action = action_dist.sample()
            move_raw = move_dist.rsample()

            log_prob_action = action_dist.log_prob(action)
            log_prob_move = move_dist.log_prob(move_raw).sum(dim=-1)

            log_prob = log_prob_action + log_prob_move
            log_prob = log_prob.clamp(min=-100.0)

            move = move_raw.clamp(1e-7, 1 - 1e-7)

        # Convert move params to angle and distance
        # angle: [0,1] -> [-pi, pi]
        # distance: [0,1] -> [0, 1] (normalized)
        action_dict = {
            'action': action.squeeze(0),
            'move_angle': (move[:, 0] * 2 - 1) * np.pi,  # [-pi, pi]
            'move_distance': move[:, 1],
        }
        action_dict['move_angle'] = action_dict['move_angle'].squeeze(0)
        action_dict['move_distance'] = action_dict['move_distance'].squeeze(0)

        return action_dict, log_prob.squeeze(0), value.squeeze()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor,
                         moves: torch.Tensor, hidden: Optional[Tuple] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: [batch_size, seq_len, state_dim]
            actions: [batch_size] discrete actions
            moves: [batch_size, 2] (angle_normalized, distance)
            hidden: Optional LSTM hidden state

        Returns:
            log_probs: [batch_size]
            values: [batch_size]
            entropy: [batch_size]
        """
        if states.dim() == 2:
            states = states.unsqueeze(1)

        action_logits, move_alpha, move_beta, values, _ = self.forward(states, hidden)

        action_dist = Categorical(logits=action_logits)
        move_dist = Beta(move_alpha, move_beta)

        moves_clamped = moves.clamp(1e-7, 1 - 1e-7)

        log_prob_action = action_dist.log_prob(actions)
        log_prob_move = move_dist.log_prob(moves_clamped).sum(dim=-1)

        log_probs = log_prob_action + log_prob_move
        log_probs = log_probs.clamp(min=-100.0)

        entropy = action_dist.entropy() + move_dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'hidden_dim': self.hidden_dim,
            'lstm_layers': self.lstm_layers,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'MicroNetwork':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(
            state_dim=checkpoint.get('state_dim', MICRO_STATE_DIM),
            num_actions=checkpoint.get('num_actions', MICRO_ACTION_DIM),
            hidden_dim=checkpoint.get('hidden_dim', 64),
            lstm_layers=checkpoint.get('lstm_layers', 1),
        )
        # Handle both checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'policy_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['policy_state_dict'])
        else:
            raise KeyError(f"Checkpoint missing state_dict. Keys: {checkpoint.keys()}")
        return model


def _sanitize_float(x, default: float = 0.0, min_val: float = -np.pi, max_val: float = np.pi) -> float:
    """Sanitize float value, replacing NaN/Inf with default."""
    import math
    val = float(x.item()) if hasattr(x, 'item') else float(x)
    if math.isnan(val) or math.isinf(val):
        return default
    return max(min_val, min(max_val, val))


def action_dict_to_command(action_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Convert action dict to JSON-serializable command."""
    action_val = action_dict['action']
    action_int = int(action_val.item()) if hasattr(action_val, 'item') else int(action_val)
    # Clamp to valid action range
    action_int = max(0, min(MICRO_ACTION_DIM - 1, action_int))

    return {
        'action': action_int,
        'move_angle': _sanitize_float(action_dict['move_angle'], 0.0, -np.pi, np.pi),
        'move_distance': _sanitize_float(action_dict['move_distance'], 0.0, 0.0, 1.0),
    }


if __name__ == '__main__':
    print("Testing MicroNetwork...")

    model = MicroNetwork()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test single step
    state = torch.randn(MICRO_STATE_DIM)
    model.reset_hidden(1)

    action_dict, log_prob, value = model.get_action(state)

    print(f"\nAction: {MicroAction.name(action_dict['action'].item())}")
    print(f"Move angle: {action_dict['move_angle']:.3f} rad")
    print(f"Move distance: {action_dict['move_distance']:.3f}")
    print(f"Log prob: {log_prob:.3f}")
    print(f"Value: {value.item():.3f}")

    # Test sequential processing (temporal coherence)
    print("\nTesting temporal coherence...")
    model.reset_hidden(1)

    actions = []
    for i in range(5):
        state = torch.randn(MICRO_STATE_DIM)
        action_dict, _, _ = model.get_action(state)
        actions.append(MicroAction.name(action_dict['action'].item()))

    print(f"Action sequence: {actions}")

    # Test batch processing
    print("\nTesting batch processing...")
    model.reset_hidden(32)
    states = torch.randn(32, MICRO_STATE_DIM)
    actions = torch.randint(0, MICRO_ACTION_DIM, (32,))
    moves = torch.rand(32, 2)

    log_probs, values, entropy = model.evaluate_actions(states, actions, moves)
    print(f"Batch - log_probs: {log_probs.shape}, entropy: {entropy.mean():.3f}")

    # Test deterministic
    model.reset_hidden(1)
    action_dict_det, _, _ = model.get_action(torch.randn(MICRO_STATE_DIM), deterministic=True)
    print(f"\nDeterministic: {MicroAction.name(action_dict_det['action'].item())}")

    print("\nMicroNetwork test passed!")
