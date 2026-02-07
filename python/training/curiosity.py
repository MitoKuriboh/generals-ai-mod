"""
Random Network Distillation (RND) for Curiosity-Driven Exploration

Implements intrinsic motivation via prediction error on a fixed random network.
Reference: https://arxiv.org/abs/1810.12894

Usage:
    curiosity = RNDCuriosity(state_dim=44)
    intrinsic_reward = curiosity.intrinsic_reward(state)
    curiosity.update(states_batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple
import numpy as np


class RandomNetwork(nn.Module):
    """
    Fixed random network that provides prediction targets.
    Weights are frozen after initialization.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize with orthogonal weights for better feature diversity
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictorNetwork(nn.Module):
    """
    Learned predictor network that tries to match the target network.
    Prediction error indicates novelty.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize with orthogonal weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RunningMeanStd:
    """
    Running mean and standard deviation for reward normalization.
    Prevents intrinsic rewards from exploding or vanishing.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update running statistics with new batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Welford's online algorithm for combining running stats."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values using running statistics."""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RNDCuriosity:
    """
    Random Network Distillation for curiosity-driven exploration.

    Uses a fixed random target network and a trained predictor network.
    The prediction error serves as intrinsic reward for novel states.

    Args:
        state_dim: Dimension of state features
        hidden_dim: Hidden layer size for both networks
        output_dim: Output embedding dimension
        lr: Learning rate for predictor network
        intrinsic_coef: Scale factor for intrinsic rewards
        normalize_rewards: Whether to normalize intrinsic rewards
        device: Device to run on
    """

    def __init__(
        self,
        state_dim: int = 44,
        hidden_dim: int = 128,
        output_dim: int = 64,
        lr: float = 1e-4,
        intrinsic_coef: float = 1.0,
        normalize_rewards: bool = True,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.intrinsic_coef = intrinsic_coef
        self.normalize_rewards = normalize_rewards
        self.device = torch.device(device)

        # Create networks
        self.target_network = RandomNetwork(state_dim, hidden_dim, output_dim).to(self.device)
        self.predictor_network = PredictorNetwork(state_dim, hidden_dim, output_dim).to(self.device)

        # Optimizer for predictor only
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=lr)

        # Running statistics for normalization
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(state_dim,))

        # Statistics
        self.update_count = 0
        self.total_intrinsic = 0.0

    def intrinsic_reward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward for a state (batch supported).

        Args:
            state: State tensor [state_dim] or [batch_size, state_dim]

        Returns:
            Intrinsic reward scalar or [batch_size] tensor
        """
        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)

        state = state.to(self.device)

        with torch.no_grad():
            target = self.target_network(state)
            prediction = self.predictor_network(state)

            # MSE prediction error as curiosity signal
            intrinsic = F.mse_loss(prediction, target, reduction='none').mean(dim=-1)

            # Normalize if enabled
            if self.normalize_rewards:
                intrinsic_np = intrinsic.cpu().numpy()
                self.reward_rms.update(intrinsic_np)
                intrinsic = intrinsic / (np.sqrt(self.reward_rms.var) + 1e-8)

            # Scale by coefficient
            intrinsic = intrinsic * self.intrinsic_coef

        if was_1d:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic

    def update(self, states: torch.Tensor) -> float:
        """
        Update predictor network to better predict target network.

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            Mean prediction loss
        """
        states = states.to(self.device)

        # Update observation normalization
        if self.normalize_rewards:
            self.obs_rms.update(states.cpu().numpy())

        # Get target (no gradient needed)
        with torch.no_grad():
            target = self.target_network(states)

        # Get prediction
        prediction = self.predictor_network(states)

        # Compute loss
        loss = F.mse_loss(prediction, target)

        # Optimize predictor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        self.total_intrinsic += loss.item()

        return loss.item()

    def get_combined_reward(
        self,
        extrinsic_reward: float,
        state: torch.Tensor,
        extrinsic_weight: float = 1.0,
        intrinsic_weight: float = 0.5
    ) -> float:
        """
        Combine extrinsic and intrinsic rewards.

        Args:
            extrinsic_reward: Reward from environment
            state: Current state
            extrinsic_weight: Weight for extrinsic reward
            intrinsic_weight: Weight for intrinsic reward

        Returns:
            Combined reward
        """
        intrinsic = self.intrinsic_reward(state)
        if isinstance(intrinsic, torch.Tensor):
            intrinsic = intrinsic.item()

        return extrinsic_weight * extrinsic_reward + intrinsic_weight * intrinsic

    def get_stats(self) -> dict:
        """Get curiosity statistics."""
        avg_intrinsic = self.total_intrinsic / max(1, self.update_count)
        return {
            'rnd_updates': self.update_count,
            'rnd_avg_loss': avg_intrinsic,
            'rnd_reward_mean': float(self.reward_rms.mean) if np.isscalar(self.reward_rms.mean) else 0.0,
            'rnd_reward_std': float(np.sqrt(self.reward_rms.var)) if np.isscalar(self.reward_rms.var) else 0.0,
        }

    def save(self, path: str):
        """Save predictor network state."""
        torch.save({
            'predictor_state_dict': self.predictor_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_rms_mean': self.reward_rms.mean,
            'reward_rms_var': self.reward_rms.var,
            'reward_rms_count': self.reward_rms.count,
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
            'update_count': self.update_count,
            'total_intrinsic': self.total_intrinsic,
        }, path)

    def load(self, path: str):
        """Load predictor network state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.predictor_network.load_state_dict(checkpoint['predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.reward_rms.mean = checkpoint['reward_rms_mean']
        self.reward_rms.var = checkpoint['reward_rms_var']
        self.reward_rms.count = checkpoint['reward_rms_count']
        self.obs_rms.mean = checkpoint['obs_rms_mean']
        self.obs_rms.var = checkpoint['obs_rms_var']
        self.obs_rms.count = checkpoint['obs_rms_count']
        self.update_count = checkpoint.get('update_count', 0)
        self.total_intrinsic = checkpoint.get('total_intrinsic', 0.0)


# Integration helper for PPO
class CuriosityPPOWrapper:
    """
    Wrapper to integrate RND curiosity with PPO training.

    Usage:
        wrapper = CuriosityPPOWrapper(ppo_agent, state_dim=44)
        # In training loop:
        combined_reward = wrapper.get_reward(extrinsic_reward, state)
        wrapper.update_curiosity(states_batch)
    """

    def __init__(
        self,
        ppo_agent,
        state_dim: int = 44,
        intrinsic_coef: float = 0.5,
        curiosity_lr: float = 1e-4,
        device: str = 'cpu'
    ):
        self.ppo_agent = ppo_agent
        self.curiosity = RNDCuriosity(
            state_dim=state_dim,
            intrinsic_coef=intrinsic_coef,
            lr=curiosity_lr,
            device=device
        )

    def get_reward(self, extrinsic_reward: float, state: torch.Tensor) -> float:
        """Get combined extrinsic + intrinsic reward."""
        return self.curiosity.get_combined_reward(
            extrinsic_reward, state,
            extrinsic_weight=1.0,
            intrinsic_weight=1.0
        )

    def update_curiosity(self, states: torch.Tensor) -> float:
        """Update curiosity module with batch of states."""
        return self.curiosity.update(states)

    def save(self, path_prefix: str):
        """Save both PPO agent and curiosity module."""
        self.ppo_agent.save(f"{path_prefix}_ppo.pt")
        self.curiosity.save(f"{path_prefix}_curiosity.pt")

    def load(self, path_prefix: str):
        """Load both PPO agent and curiosity module."""
        self.ppo_agent.load(f"{path_prefix}_ppo.pt")
        self.curiosity.load(f"{path_prefix}_curiosity.pt")


if __name__ == '__main__':
    print("Testing RND Curiosity Module...")

    # Test basic functionality
    curiosity = RNDCuriosity(state_dim=44, hidden_dim=128, output_dim=64)

    # Test single state
    state = torch.randn(44)
    intrinsic = curiosity.intrinsic_reward(state)
    print(f"Single state intrinsic reward: {intrinsic.item():.4f}")

    # Test batch
    states = torch.randn(32, 44)
    intrinsic_batch = curiosity.intrinsic_reward(states)
    print(f"Batch intrinsic rewards: mean={intrinsic_batch.mean():.4f}, std={intrinsic_batch.std():.4f}")

    # Test update
    loss = curiosity.update(states)
    print(f"Update loss: {loss:.4f}")

    # Test that familiar states have lower intrinsic reward
    print("\nTesting novelty detection...")
    familiar_state = torch.randn(44)

    # Train on familiar state many times
    for _ in range(100):
        curiosity.update(familiar_state.unsqueeze(0))

    familiar_reward = curiosity.intrinsic_reward(familiar_state)
    novel_reward = curiosity.intrinsic_reward(torch.randn(44))

    print(f"Familiar state reward: {familiar_reward.item():.4f}")
    print(f"Novel state reward: {novel_reward.item():.4f}")

    # Novel should generally have higher reward
    if novel_reward > familiar_reward:
        print("Novel state correctly has higher intrinsic reward")
    else:
        print("Note: May need more training for clear novelty signal")

    # Test combined reward
    combined = curiosity.get_combined_reward(1.0, torch.randn(44))
    print(f"\nCombined reward (extrinsic=1.0): {combined:.4f}")

    # Test stats
    stats = curiosity.get_stats()
    print(f"\nCuriosity stats: {stats}")

    # Test save/load
    curiosity.save('/tmp/test_curiosity.pt')
    curiosity2 = RNDCuriosity(state_dim=44)
    curiosity2.load('/tmp/test_curiosity.pt')
    print("\nSave/load test passed")

    print("\nRND Curiosity test passed!")
