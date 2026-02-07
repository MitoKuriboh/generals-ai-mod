"""
Tactical Network for Team-Level Decision Making

Architecture:
- Input: 64 floats (team state + strategic embedding)
- Attention: Multi-head attention over unit features (optional)
- Hidden: 2x128 MLP with LayerNorm
- Output: Hybrid action space
  - Discrete: 8 tactical actions (Categorical)
  - Continuous: target_x, target_y, attitude (Beta distributions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Tuple, Dict, Optional, List
import numpy as np
import math


# State and action dimensions
TACTICAL_STATE_DIM = 64
TACTICAL_ACTION_DIM = 8  # Discrete actions
TACTICAL_CONTINUOUS_DIM = 3  # target_x, target_y, attitude

# Unit features within tactical state (for attention)
# State layout: [strategy_emb(8), inf(3), veh(3), air(3), mixed(3), status(8), situational(16), objective(8), temporal(4)]
UNIT_FEATURE_DIM = 12  # Each unit group has 3 features, 4 groups = 12
UNIT_GROUPS = 4  # infantry, vehicles, aircraft, mixed


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


class UnitAttention(nn.Module):
    """
    Multi-head attention over unit group features.

    Learns to weight which unit types (infantry, vehicles, aircraft, mixed)
    are most relevant for the current tactical decision.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Project unit features to query, key, value
        self.unit_proj = nn.Linear(3, embed_dim)  # 3 features per unit group

        # Context embedding (strategy + status)
        self.context_proj = nn.Linear(16, embed_dim)  # strategy(8) + status subset(8)

        # Multi-head attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply attention over unit features.

        Args:
            state: Full tactical state [batch_size, 64]

        Returns:
            Attention-weighted features [batch_size, embed_dim]
        """
        batch_size = state.size(0) if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Extract unit features from state
        # Layout: strategy(8), inf(3), veh(3), air(3), mixed(3), status(8), ...
        strategy_emb = state[:, :8]
        infantry = state[:, 8:11]
        vehicles = state[:, 11:14]
        aircraft = state[:, 14:17]
        mixed = state[:, 17:20]
        status = state[:, 20:28]

        # Stack unit groups: [batch, 4, 3]
        units = torch.stack([infantry, vehicles, aircraft, mixed], dim=1)

        # Project units to embed_dim: [batch, 4, embed_dim]
        unit_embeddings = self.unit_proj(units)

        # Create context query from strategy + status
        context = torch.cat([strategy_emb, status], dim=-1)
        context_emb = self.context_proj(context)  # [batch, embed_dim]

        # Query is context, keys and values are unit embeddings
        Q = self.q_proj(context_emb).unsqueeze(1)  # [batch, 1, embed_dim]
        K = self.k_proj(unit_embeddings)  # [batch, 4, embed_dim]
        V = self.v_proj(unit_embeddings)  # [batch, 4, embed_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 4, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 4, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, heads, 1, 4]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, 1, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
        attn_output = self.out_proj(attn_output).squeeze(1)  # [batch, embed_dim]

        # Add residual from context and normalize
        output = self.norm(attn_output + context_emb)

        return output

    def get_attention_weights(self, state: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization."""
        batch_size = state.size(0) if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)

        strategy_emb = state[:, :8]
        infantry = state[:, 8:11]
        vehicles = state[:, 11:14]
        aircraft = state[:, 14:17]
        mixed = state[:, 17:20]
        status = state[:, 20:28]

        units = torch.stack([infantry, vehicles, aircraft, mixed], dim=1)
        unit_embeddings = self.unit_proj(units)

        context = torch.cat([strategy_emb, status], dim=-1)
        context_emb = self.context_proj(context)

        Q = self.q_proj(context_emb).unsqueeze(1)
        K = self.k_proj(unit_embeddings)

        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 4, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Average across heads: [batch, 4]
        return attn_weights.mean(dim=1).squeeze(1)


class TacticalNetwork(nn.Module):
    """
    Actor-Critic network for tactical layer.

    Uses hybrid action space:
    - Categorical distribution for discrete action selection
    - Beta distributions for continuous parameters (bounded [0,1])

    Optional attention mechanism for unit-aware decisions.
    """

    def __init__(self, state_dim: int = TACTICAL_STATE_DIM,
                 num_actions: int = TACTICAL_ACTION_DIM,
                 hidden_dim: int = 128,
                 use_attention: bool = True,
                 num_attention_heads: int = 4):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Optional attention layer for unit features
        if use_attention:
            self.attention = UnitAttention(
                embed_dim=64,
                num_heads=num_attention_heads,
                dropout=0.1
            )
            # Combine attention output with non-unit state features
            # State layout: strategy(8), units(12 = 4 groups x 3 features), rest(44)
            # Attention output: 64, other features: strategy(8) + rest(44) = 52
            # Combined: 64 + 52 = 116
            combined_dim = 64 + 8 + (state_dim - 20)  # 64 + 8 + 44 = 116
        else:
            self.attention = None
            combined_dim = state_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
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
        # Apply attention if enabled
        if self.use_attention and self.attention is not None:
            # Get attention-weighted unit features
            attn_features = self.attention(state)  # [batch, 64]

            # Get non-unit features (everything after unit groups)
            # Layout: strategy(8), units(12), status(8), situational(16), objective(8), temporal(4)
            # Units are at positions 8:20, rest is 0:8 and 20:64
            if state.dim() == 1:
                other_features = torch.cat([state[:8], state[20:]], dim=-1)
            else:
                other_features = torch.cat([state[:, :8], state[:, 20:]], dim=-1)

            # Combine attention output with other features
            combined = torch.cat([attn_features, other_features], dim=-1)
            features = self.shared(combined)
        else:
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
            'use_attention': self.use_attention,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'TacticalNetwork':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(
            state_dim=checkpoint.get('state_dim', TACTICAL_STATE_DIM),
            num_actions=checkpoint.get('num_actions', TACTICAL_ACTION_DIM),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            use_attention=checkpoint.get('use_attention', True),
        )
        # Handle both checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'policy_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['policy_state_dict'], strict=False)
        else:
            raise KeyError(f"Checkpoint missing state_dict. Keys: {checkpoint.keys()}")
        return model

    def get_attention_weights(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights for visualization (unit importance)."""
        if self.use_attention and self.attention is not None:
            return self.attention.get_attention_weights(state)
        return None


def _sanitize_float(x, default: float = 0.5) -> float:
    """Sanitize float value, replacing NaN/Inf with default."""
    import math
    val = float(x.item()) if hasattr(x, 'item') else float(x)
    if math.isnan(val) or math.isinf(val):
        return default
    return max(0.0, min(1.0, val))  # Clamp to [0, 1]


def action_dict_to_command(action_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Convert action dict to JSON-serializable command."""
    action_val = action_dict['action']
    action_int = int(action_val.item()) if hasattr(action_val, 'item') else int(action_val)
    # Clamp to valid action range
    action_int = max(0, min(TACTICAL_ACTION_DIM - 1, action_int))

    return {
        'action': action_int,
        'target_x': _sanitize_float(action_dict['target_x'], 0.5),
        'target_y': _sanitize_float(action_dict['target_y'], 0.5),
        'attitude': _sanitize_float(action_dict['attitude'], 0.5),
    }


if __name__ == '__main__':
    print("Testing TacticalNetwork...")

    # Test with attention
    print("\n=== Testing with Attention ===")
    model = TacticalNetwork(use_attention=True)
    print(f"Parameters (with attention): {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    state = torch.randn(TACTICAL_STATE_DIM)
    action_dict, log_prob, value = model.get_action(state)

    print(f"\nAction: {TacticalAction.name(action_dict['action'].item())}")
    print(f"Target: ({action_dict['target_x']:.3f}, {action_dict['target_y']:.3f})")
    print(f"Attitude: {action_dict['attitude']:.3f}")
    print(f"Log prob: {log_prob:.3f}")
    print(f"Value: {value.item():.3f}")

    # Test attention weights
    attn_weights = model.get_attention_weights(state)
    if attn_weights is not None:
        print(f"\nAttention weights (infantry, vehicles, aircraft, mixed):")
        print(f"  {attn_weights.squeeze().tolist()}")

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

    # Test without attention for comparison
    print("\n=== Testing without Attention ===")
    model_no_attn = TacticalNetwork(use_attention=False)
    print(f"Parameters (no attention): {sum(p.numel() for p in model_no_attn.parameters()):,}")

    action_dict2, _, _ = model_no_attn.get_action(state)
    print(f"Action (no attention): {TacticalAction.name(action_dict2['action'].item())}")

    # Test UnitAttention directly
    print("\n=== Testing UnitAttention ===")
    attention = UnitAttention(embed_dim=64, num_heads=4)
    attn_out = attention(state)
    print(f"Attention output shape: {attn_out.shape}")
    print(f"Attention output norm: {attn_out.norm():.3f}")

    print("\nTacticalNetwork test passed!")
