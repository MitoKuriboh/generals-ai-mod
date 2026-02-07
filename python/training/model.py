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
from typing import Tuple, Dict, Optional
import numpy as np
import math


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
    # FIX: Added clipping to prevent outliers from destabilizing training
    features.append(np.clip(state.get('money', 0) / 5.0, 0.0, 2.0))  # Log-scaled money, clip outliers
    features.append(np.clip(state.get('power', 0) / 100.0, -2.0, 2.0))  # Power can be negative
    features.append(np.clip(state.get('income', 0) / 10.0, -1.0, 2.0))  # Income rate
    features.append(np.clip(state.get('supply', 0.5), 0.0, 1.0))

    # Own forces (12) - 3 features per category
    # Format: [log10(count+1), avg_health_ratio (0-1), production_queue_count]
    # health_ratio indicates army condition, useful for retreat/reinforce decisions
    for key in ['own_infantry', 'own_vehicles', 'own_aircraft', 'own_structures']:
        arr = state.get(key, [0, 0, 0])
        features.extend(arr[:3] if len(arr) >= 3 else arr + [0] * (3 - len(arr)))

    # Enemy forces (12) - same format as own forces
    for key in ['enemy_infantry', 'enemy_vehicles', 'enemy_aircraft', 'enemy_structures']:
        arr = state.get(key, [0, 0, 0])
        features.extend(arr[:3] if len(arr) >= 3 else arr + [0] * (3 - len(arr)))

    # Strategic (8)
    # FIX: Added clipping to prevent outliers
    features.append(np.clip(state.get('game_time', 0) / 30.0, 0.0, 3.0))  # Clip at 90 min
    features.append(np.clip(state.get('tech_level', 0), 0.0, 1.0))
    features.append(np.clip(state.get('base_threat', 0), 0.0, 1.0))
    features.append(np.clip(state.get('army_strength', 1.0) / 2.0, 0.0, 2.0))  # Clip at 4x advantage
    features.append(np.clip(state.get('under_attack', 0), 0.0, 1.0))
    features.append(np.clip(state.get('distance_to_enemy', 0.5), 0.0, 1.0))

    # Faction one-hot (3)
    features.append(state.get('is_usa', 0))
    features.append(state.get('is_china', 0))
    features.append(state.get('is_gla', 0))

    # Ensure correct length (dynamic padding to STATE_DIM=44)
    # Current: 4 economy + 12 own + 12 enemy + 6 strategic + 3 faction = 37
    while len(features) < STATE_DIM:
        features.append(0.0)

    return torch.tensor(features, dtype=torch.float32)


def action_tensor_to_dict(action: torch.Tensor) -> Dict:
    """Convert action tensor to recommendation dictionary."""
    # Action tensor has 8 values, all sigmoid-activated (0-1)
    a = action.detach().cpu().numpy()

    # FIX H3: Check for NaN values and fall back to safe defaults
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        print("[Model] WARNING: NaN/Inf detected in action tensor, using safe defaults")
        return {
            'priority_economy': 0.25,
            'priority_defense': 0.25,
            'priority_military': 0.25,
            'priority_tech': 0.25,
            'prefer_infantry': 0.33,
            'prefer_vehicles': 0.34,
            'prefer_aircraft': 0.33,
            'aggression': 0.5,
            'target_player': -1
        }

    # Build priorities (normalize to sum to 1)
    priorities = np.array([a[0], a[1], a[2], a[3]])
    priorities = np.maximum(priorities, 0.05)  # Minimum 5%
    # FIX P3: Add epsilon to prevent division by zero (NaN input safety)
    priorities = priorities / (priorities.sum() + 1e-8)

    # Army preferences (normalize to sum to 1)
    army = np.array([a[4], a[5], a[6]])
    army = np.maximum(army, 0.1)  # Minimum 10%
    # FIX P3: Add epsilon to prevent division by zero (NaN input safety)
    army = army / (army.sum() + 1e-8)

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

    Uses Beta distribution for proper bounded [0,1] outputs with correct log-probs.
    This avoids the log-prob corruption caused by double-clamping approaches.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = TOTAL_ACTION_DIM,
                 hidden_dim: int = 256):  # Increased from 128 for RTS complexity
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor with LayerNorm for training stability
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy) - outputs alpha and beta parameters for Beta distribution
        # Using softplus to ensure positive concentration parameters
        self.actor_alpha = nn.Linear(hidden_dim, action_dim)
        self.actor_beta = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function) - also increased capacity
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
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

        # Initialize alpha/beta outputs to produce Beta(2,2) initially (mode at 0.5)
        nn.init.orthogonal_(self.actor_alpha.weight, gain=0.01)
        nn.init.constant_(self.actor_alpha.bias, 1.0)  # softplus(1) ≈ 1.31
        nn.init.orthogonal_(self.actor_beta.weight, gain=0.01)
        nn.init.constant_(self.actor_beta.bias, 1.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Batch of states [batch_size, state_dim]

        Returns:
            alpha: Beta distribution alpha parameter [batch_size, action_dim]
            beta: Beta distribution beta parameter [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
        """
        features = self.shared(state)
        # Softplus ensures positive concentration parameters, add 1 to avoid very peaked distributions
        alpha = F.softplus(self.actor_alpha(features)) + 1.0
        beta = F.softplus(self.actor_beta(features)) + 1.0
        value = self.critic(features)
        return alpha, beta, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy using Beta distribution.

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            deterministic: If True, return mode of Beta distribution

        Returns:
            action: Sampled action in [0,1] [batch_size, action_dim]
            log_prob: Log probability of action (properly computed)
            value: State value estimate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        alpha, beta, value = self.forward(state)

        if deterministic:
            # Mode of Beta distribution: (alpha - 1) / (alpha + beta - 2), clamped to [0,1]
            action = ((alpha - 1) / (alpha + beta - 2 + 1e-8)).clamp(0, 1)
            log_prob = torch.zeros(state.size(0), device=state.device)
        else:
            # Sample from Beta distribution - proper [0,1] bounded with correct log-probs
            dist = torch.distributions.Beta(alpha, beta)
            action_raw = dist.rsample()
            # CRITICAL FIX: Compute log_prob on RAW sample before clamping
            # Computing log_prob after clamping corrupts policy gradients by pushing
            # towards clamped boundaries rather than true optimal actions
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            # Clamp log_prob to prevent -inf from extreme samples
            log_prob = log_prob.clamp(min=-100.0)
            # Clamp action for output only (wider range to reduce interference)
            action = action_raw.clamp(1e-7, 1 - 1e-7)

        # FIX: Keep consistent 1D shapes to avoid torch.stack() failures
        # squeeze(0) produces scalars when batch=1, view() ensures 1D tensors
        return action.view(-1), log_prob.view(1), value.view(1)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update using Beta distribution.

        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]

        Returns:
            log_probs: Log probabilities of actions (correctly computed)
            values: State value estimates
            entropy: Policy entropy
        """
        alpha, beta, values = self.forward(states)

        # Create Beta distribution and evaluate
        dist = torch.distributions.Beta(alpha, beta)
        # CRITICAL FIX: Use wider clamp range to minimize log-prob distortion
        # Actions stored in buffer were already clamped to [1e-7, 1-1e-7] during get_action
        # Re-clamping here with same range ensures numerical stability without bias
        actions_clamped = actions.clamp(1e-7, 1 - 1e-7)
        log_probs = dist.log_prob(actions_clamped).sum(dim=-1)
        # Clamp log_probs to prevent -inf from corrupting gradients
        log_probs = log_probs.clamp(min=-100.0)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'policy_state_dict': self.state_dict(),  # Consistent with PPOAgent.save()
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'PolicyNetwork':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        state_dict_key = 'policy_state_dict' if 'policy_state_dict' in checkpoint else 'state_dict'
        if state_dict_key not in checkpoint:
            raise KeyError(f"Checkpoint missing state_dict. Keys: {checkpoint.keys()}")

        model = cls(
            state_dim=checkpoint.get('state_dim', STATE_DIM),
            action_dim=checkpoint.get('action_dim', TOTAL_ACTION_DIM)
        )
        model.load_state_dict(checkpoint[state_dict_key])
        return model


class StrategicTransformer(nn.Module):
    """
    Decision Transformer for Strategic Layer.

    Treats RL as sequence prediction - conditioned on desired return,
    predicts actions that achieve that return.

    Reference: https://arxiv.org/abs/2106.01345
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = TOTAL_ACTION_DIM,
        hidden_dim: int = 128,
        max_seq_len: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embeddings for each modality
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_timestep = nn.Embedding(max_seq_len, hidden_dim)

        # Layer norm for embeddings
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action prediction head (outputs Beta distribution parameters)
        self.action_alpha = nn.Linear(hidden_dim, action_dim)
        self.action_beta = nn.Linear(hidden_dim, action_dim)

        # Value prediction head
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        # Initialize output heads for Beta(2,2) initial distribution
        nn.init.constant_(self.action_alpha.bias, 1.0)
        nn.init.constant_(self.action_beta.bias, 1.0)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Decision Transformer.

        Args:
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]
            returns_to_go: [batch, seq_len, 1]
            timesteps: [batch, seq_len]
            attention_mask: [batch, seq_len] optional mask

        Returns:
            action_alpha: [batch, seq_len, action_dim]
            action_beta: [batch, seq_len, action_dim]
            values: [batch, seq_len]
        """
        batch_size, seq_len = states.shape[:2]

        # Embed each modality
        state_emb = self.embed_state(states)
        action_emb = self.embed_action(actions)
        return_emb = self.embed_return(returns_to_go)
        time_emb = self.embed_timestep(timesteps.clamp(0, self.max_seq_len - 1))

        # Add timestep embeddings
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        return_emb = return_emb + time_emb

        # Interleave: [R_1, S_1, A_1, R_2, S_2, A_2, ...]
        # Shape: [batch, seq_len * 3, hidden_dim]
        stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
        stacked = stacked.reshape(batch_size, seq_len * 3, self.hidden_dim)
        stacked = self.embed_ln(stacked)

        # Create causal mask if not provided
        if attention_mask is None:
            # Causal mask for transformer
            causal_mask = torch.triu(
                torch.ones(seq_len * 3, seq_len * 3, device=states.device),
                diagonal=1
            ).bool()
        else:
            # Expand mask for interleaved sequence
            expanded_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(batch_size, -1)
            causal_mask = None  # Use padding mask instead

        # Transform
        hidden = self.transformer(stacked, mask=causal_mask)

        # Extract state positions (indices 1, 4, 7, ...) for action prediction
        state_hidden = hidden[:, 1::3, :]  # [batch, seq_len, hidden_dim]

        # Predict action distribution parameters
        alpha = F.softplus(self.action_alpha(state_hidden)) + 1.0
        beta = F.softplus(self.action_beta(state_hidden)) + 1.0

        # Predict values
        values = self.value_head(state_hidden).squeeze(-1)

        return alpha, beta, values

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action for the last timestep in the sequence.

        Args:
            states: [batch, seq_len, state_dim] or [seq_len, state_dim]
            actions: [batch, seq_len, action_dim] or [seq_len, action_dim]
            returns_to_go: [batch, seq_len, 1] or [seq_len, 1]
            timesteps: [batch, seq_len] or [seq_len]
            deterministic: If True, return mode of distribution

        Returns:
            action: [batch, action_dim]
            log_prob: [batch]
            value: [batch]
        """
        # Add batch dimension if needed
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            returns_to_go = returns_to_go.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)

        alpha, beta, values = self.forward(states, actions, returns_to_go, timesteps)

        # Get last timestep predictions
        alpha_last = alpha[:, -1, :]
        beta_last = beta[:, -1, :]
        value_last = values[:, -1]

        if deterministic:
            # Mode of Beta distribution
            action = ((alpha_last - 1) / (alpha_last + beta_last - 2 + 1e-8)).clamp(0, 1)
            log_prob = torch.zeros(action.size(0), device=action.device)
        else:
            dist = torch.distributions.Beta(alpha_last, beta_last)
            action = dist.rsample().clamp(1e-7, 1 - 1e-7)
            log_prob = dist.log_prob(action).sum(dim=-1).clamp(min=-100.0)

        return action.squeeze(0), log_prob.squeeze(0), value_last.squeeze(0)


class StrategicMoE(nn.Module):
    """
    Mixture of Experts for Strategic Layer.

    4 specialized experts with learned routing:
    - Expert 0: Aggressive/Rush strategies
    - Expert 1: Defensive/Turtle strategies
    - Expert 2: Economy-focused strategies
    - Expert 3: Tech/Late-game strategies

    The router learns which expert(s) to use based on game state.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = TOTAL_ACTION_DIM,
        hidden_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        noise_std: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Router network
        self.router = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # Expert networks (each is a small policy network)
        self.experts = nn.ModuleList([
            self._make_expert(state_dim, action_dim, hidden_dim)
            for _ in range(num_experts)
        ])

        # Shared value head (experts don't need separate critics)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Expert usage tracking for load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))

        self._init_weights()

    def _make_expert(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """Create a single expert network."""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # alpha and beta for Beta dist
        )

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Initialize router for uniform expert selection initially
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse expert routing.

        Args:
            state: [batch_size, state_dim] or [state_dim]

        Returns:
            alpha: [batch_size, action_dim]
            beta: [batch_size, action_dim]
            value: [batch_size, 1]
            router_probs: [batch_size, num_experts] for auxiliary loss
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.size(0)

        # Get router logits and add noise during training
        router_logits = self.router(state)  # [batch, num_experts]
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Softmax for router probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute weighted expert outputs
        expert_outputs = torch.zeros(batch_size, self.action_dim * 2, device=state.device)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # [batch]
            expert_weight = top_k_probs[:, i:i+1]  # [batch, 1]

            # Gather expert outputs for this batch
            for b in range(batch_size):
                eidx = expert_idx[b].item()
                expert_out = self.experts[eidx](state[b:b+1])
                expert_outputs[b] += (expert_weight[b] * expert_out).squeeze(0)

                # Track expert usage
                if self.training:
                    self.expert_usage[eidx] += 1

        # Split into alpha and beta
        alpha = F.softplus(expert_outputs[:, :self.action_dim]) + 1.0
        beta = F.softplus(expert_outputs[:, self.action_dim:]) + 1.0

        # Value from shared head
        value = self.value_head(state)

        return alpha, beta, value, router_probs

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from MoE policy.

        Args:
            state: State tensor
            deterministic: If True, return mode of distribution

        Returns:
            action: [action_dim]
            log_prob: scalar
            value: scalar
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        alpha, beta, value, _ = self.forward(state)

        if deterministic:
            action = ((alpha - 1) / (alpha + beta - 2 + 1e-8)).clamp(0, 1)
            log_prob = torch.zeros(state.size(0), device=state.device)
        else:
            dist = torch.distributions.Beta(alpha, beta)
            action = dist.rsample().clamp(1e-7, 1 - 1e-7)
            log_prob = dist.log_prob(action).sum(dim=-1).clamp(min=-100.0)

        return action.view(-1), log_prob.view(1), value.view(1)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_probs, values, entropy, aux_loss (for load balancing)
        """
        alpha, beta, values, router_probs = self.forward(states)

        dist = torch.distributions.Beta(alpha, beta)
        actions_clamped = actions.clamp(1e-7, 1 - 1e-7)
        log_probs = dist.log_prob(actions_clamped).sum(dim=-1).clamp(min=-100.0)
        entropy = dist.entropy().sum(dim=-1)

        # Load balancing auxiliary loss (encourages uniform expert usage)
        # Based on Switch Transformer paper
        expert_fraction = router_probs.mean(dim=0)  # [num_experts]
        uniform = torch.ones_like(expert_fraction) / self.num_experts
        aux_loss = (expert_fraction * torch.log(expert_fraction / uniform + 1e-8)).sum()

        return log_probs, values.squeeze(-1), entropy, aux_loss

    def get_expert_usage_stats(self) -> Dict:
        """Get statistics on expert usage."""
        total = self.expert_usage.sum().item()
        if total == 0:
            return {f'expert_{i}': 0.0 for i in range(self.num_experts)}

        usage_pct = (self.expert_usage / total).tolist()
        return {
            f'expert_{i}_usage': pct
            for i, pct in enumerate(usage_pct)
        }

    def reset_usage_stats(self):
        """Reset expert usage counters."""
        self.expert_usage.zero_()

    def save(self, path: str):
        """Save model."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'num_experts': self.num_experts,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'StrategicMoE':
        """Load model."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(
            state_dim=checkpoint.get('state_dim', STATE_DIM),
            action_dim=checkpoint.get('action_dim', TOTAL_ACTION_DIM),
            num_experts=checkpoint.get('num_experts', 4),
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


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

    # Verify action bounds with Beta distribution
    print("\nVerifying action bounds...")
    for _ in range(100):
        s = torch.randn(STATE_DIM)
        a, _, _ = model.get_action(s)
        assert torch.all(a >= 0) and torch.all(a <= 1), f"Action out of bounds: {a}"
    print("  All 100 samples within [0,1] ✓")

    # Verify log_probs are finite
    print("Verifying log_prob computation...")
    states = torch.randn(100, STATE_DIM)
    for i in range(100):
        a, lp, _ = model.get_action(states[i])
        assert torch.isfinite(lp), f"Log prob not finite: {lp}"
    print("  All log_probs finite ✓")

    print("\nPolicyNetwork test passed!")

    # Test Strategic Transformer
    print("\n" + "="*50)
    print("Testing StrategicTransformer...")

    transformer = StrategicTransformer(state_dim=STATE_DIM, action_dim=TOTAL_ACTION_DIM)
    print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    # Create test sequence
    seq_len = 10
    batch_size = 4
    test_states = torch.randn(batch_size, seq_len, STATE_DIM)
    test_actions = torch.rand(batch_size, seq_len, TOTAL_ACTION_DIM)
    test_returns = torch.randn(batch_size, seq_len, 1)
    test_timesteps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Test forward
    alpha, beta, values = transformer(test_states, test_actions, test_returns, test_timesteps)
    print(f"Alpha shape: {alpha.shape}, Beta shape: {beta.shape}, Values shape: {values.shape}")

    # Test get_action
    action, log_prob, value = transformer.get_action(
        test_states[0], test_actions[0], test_returns[0], test_timesteps[0]
    )
    print(f"Action: {action.shape}, Log prob: {log_prob}, Value: {value}")

    print("StrategicTransformer test passed!")

    # Test Mixture of Experts
    print("\n" + "="*50)
    print("Testing StrategicMoE...")

    moe = StrategicMoE(state_dim=STATE_DIM, action_dim=TOTAL_ACTION_DIM, num_experts=4)
    print(f"MoE parameters: {sum(p.numel() for p in moe.parameters()):,}")

    # Test forward
    test_state = torch.randn(STATE_DIM)
    alpha, beta, value, router_probs = moe.forward(test_state.unsqueeze(0))
    print(f"Alpha shape: {alpha.shape}, Router probs: {router_probs.squeeze().tolist()}")

    # Test get_action
    action, log_prob, value = moe.get_action(test_state)
    print(f"Action: {action.shape}, Log prob: {log_prob}, Value: {value}")

    # Test batch with evaluate_actions
    test_states = torch.randn(32, STATE_DIM)
    test_actions = torch.rand(32, TOTAL_ACTION_DIM)
    log_probs, values, entropy, aux_loss = moe.evaluate_actions(test_states, test_actions)
    print(f"Batch - log_probs: {log_probs.shape}, aux_loss: {aux_loss.item():.4f}")

    # Test expert usage stats
    usage = moe.get_expert_usage_stats()
    print(f"Expert usage: {usage}")

    print("StrategicMoE test passed!")

    print("\n" + "="*50)
    print("All model tests passed!")
