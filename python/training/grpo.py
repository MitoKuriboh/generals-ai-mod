"""
Group Relative Policy Optimization (GRPO)

Critic-free policy optimization that uses group-relative baselines
instead of a learned value function.

Reference:
- https://cameronrwolfe.substack.com/p/grpo
- https://verl.readthedocs.io/en/latest/algo/grpo.html

Benefits:
- ~50% less memory (no critic network)
- Simpler training pipeline
- Works well for reasoning/planning tasks
- Robust to reward scale

Usage:
    trainer = GRPOTrainer(policy)
    for batch in data:
        loss = trainer.update(states, rewards)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .model import PolicyNetwork, STATE_DIM, TOTAL_ACTION_DIM


@dataclass
class GRPOConfig:
    """GRPO hyperparameters."""
    # Learning rate
    lr: float = 3e-4
    lr_decay: float = 0.999

    # GRPO specific
    group_size: int = 8          # Number of samples per state for relative baseline
    clip_epsilon: float = 0.2   # PPO-style clipping
    kl_coef: float = 0.1        # KL penalty coefficient
    entropy_coef: float = 0.01  # Entropy bonus
    max_grad_norm: float = 0.5  # Gradient clipping

    # Baseline options
    use_mean_baseline: bool = True   # Use mean reward as baseline
    use_std_normalization: bool = True  # Normalize advantages by std

    # Training
    num_epochs: int = 4
    batch_size: int = 64


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    Instead of learning a value function, GRPO samples multiple actions
    for each state and uses the mean reward across the group as the baseline.
    This eliminates the need for a separate critic network.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        config: Optional[GRPOConfig] = None,
        device: str = 'cpu'
    ):
        self.config = config or GRPOConfig()
        self.device = torch.device(device)

        self.policy = policy.to(self.device)

        # Optimizer (no critic to optimize)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay
        )

        # Statistics
        self.update_count = 0
        self.total_steps = 0

    def sample_group_actions(
        self,
        state: torch.Tensor,
        group_size: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a group of actions for a single state.

        Args:
            state: Single state [state_dim]
            group_size: Number of actions to sample (default: config.group_size)

        Returns:
            actions: [group_size, action_dim]
            log_probs: [group_size]
        """
        group_size = group_size or self.config.group_size
        state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Repeat state for group sampling
        states = state.expand(group_size, -1)

        with torch.no_grad():
            alpha, beta, _ = self.policy.forward(states)

        dist = torch.distributions.Beta(alpha, beta)
        actions = dist.rsample().clamp(1e-7, 1 - 1e-7)
        log_probs = dist.log_prob(actions).sum(dim=-1).clamp(min=-100.0)

        return actions, log_probs

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        use_mean: bool = True,
        use_std: bool = True
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.

        Args:
            rewards: [group_size] rewards for each action in the group
            use_mean: Subtract mean reward as baseline
            use_std: Normalize by standard deviation

        Returns:
            advantages: [group_size]
        """
        advantages = rewards.clone()

        if use_mean:
            baseline = rewards.mean()
            advantages = advantages - baseline

        if use_std and rewards.numel() > 1:
            std = rewards.std()
            if std > 1e-8:
                advantages = advantages / std

        return advantages

    def update_single_state(
        self,
        state: torch.Tensor,
        reward_fn,  # Callable[[Tensor], float] - evaluates an action
        old_action: Optional[torch.Tensor] = None,
        old_log_prob: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        GRPO update for a single state.

        This is the core GRPO algorithm:
        1. Sample group of actions for the state
        2. Evaluate rewards for each action
        3. Compute group-relative advantages
        4. Update policy with PPO-style clipped objective

        Args:
            state: Single state [state_dim]
            reward_fn: Function that takes action and returns reward
            old_action: Original action (for importance sampling)
            old_log_prob: Original log prob

        Returns:
            Dictionary of training statistics
        """
        state = state.to(self.device)
        group_size = self.config.group_size

        # Sample group of actions
        actions, log_probs = self.sample_group_actions(state, group_size)

        # Evaluate rewards for each action
        rewards = torch.tensor([
            reward_fn(actions[i])
            for i in range(group_size)
        ], device=self.device, dtype=torch.float32)

        # Compute group-relative advantages
        advantages = self.compute_group_advantages(
            rewards,
            use_mean=self.config.use_mean_baseline,
            use_std=self.config.use_std_normalization
        )

        # Get current log probs (with gradient)
        alpha, beta, _ = self.policy.forward(state.unsqueeze(0).expand(group_size, -1))
        dist = torch.distributions.Beta(alpha, beta)
        current_log_probs = dist.log_prob(actions).sum(dim=-1).clamp(min=-100.0)

        # PPO-style clipped objective
        with torch.no_grad():
            old_log_probs_detached = log_probs

        ratio = torch.exp(current_log_probs - old_log_probs_detached)
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        )

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_coef * entropy

        # KL penalty (optional, for stability)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        kl_loss = self.config.kl_coef * approx_kl

        # Total loss
        loss = policy_loss + entropy_loss + kl_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.update_count += 1

        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl.item(),
            'mean_reward': rewards.mean().item(),
            'std_reward': rewards.std().item(),
        }

    def update_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Dict[str, float]:
        """
        GRPO update for a batch of state-action-reward tuples.

        This is a simpler variant that uses provided rewards directly
        instead of sampling groups (useful when environment is expensive).

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size]
            old_log_probs: [batch_size]

        Returns:
            Dictionary of training statistics
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Compute advantages using batch as the "group"
        advantages = self.compute_group_advantages(
            rewards,
            use_mean=self.config.use_mean_baseline,
            use_std=self.config.use_std_normalization
        )

        stats = {
            'policy_loss': 0,
            'entropy': 0,
            'approx_kl': 0,
            'clip_fraction': 0,
        }
        num_updates = 0

        # Multiple epochs
        for epoch in range(self.config.num_epochs):
            # Shuffle
            indices = torch.randperm(states.size(0))

            for start in range(0, states.size(0), self.config.batch_size):
                end = min(start + self.config.batch_size, states.size(0))
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # Get current log probs
                alpha, beta, _ = self.policy.forward(batch_states)
                dist = torch.distributions.Beta(alpha, beta)
                actions_clamped = batch_actions.clamp(1e-7, 1 - 1e-7)
                current_log_probs = dist.log_prob(actions_clamped).sum(dim=-1).clamp(min=-100.0)

                # PPO-style clipped objective
                log_prob_diff = current_log_probs - batch_old_log_probs
                ratio = torch.exp(torch.clamp(log_prob_diff, -20, 20))
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                )

                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()

                # Entropy bonus
                entropy = dist.entropy().sum(dim=-1).mean()
                entropy_loss = -self.config.entropy_coef * entropy

                # Total loss
                loss = policy_loss + entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()

                stats['policy_loss'] += policy_loss.item()
                stats['entropy'] += entropy.item()
                stats['approx_kl'] += approx_kl.item()
                stats['clip_fraction'] += clip_fraction.item()
                num_updates += 1

        # Average statistics
        for key in stats:
            stats[key] /= max(num_updates, 1)

        # Learning rate decay
        self.scheduler.step()
        stats['learning_rate'] = self.scheduler.get_last_lr()[0]

        self.update_count += 1

        return stats

    def save(self, path: str):
        """Save trainer state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.update_count = checkpoint.get('update_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)


class GRPOBuffer:
    """
    Simple buffer for GRPO training.

    Unlike PPO, we don't need to store values since there's no critic.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def add(self, state: torch.Tensor, action: torch.Tensor,
            reward: float, log_prob: torch.Tensor):
        """Add a transition."""
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.log_probs.pop(0)

        self.states.append(state.clone().detach())
        self.actions.append(action.clone().detach())
        self.rewards.append(reward)
        self.log_probs.append(log_prob.clone().detach())

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all data as tensors."""
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.log_probs).squeeze()
        )

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def __len__(self):
        return len(self.states)


class GRPOAgent:
    """
    Complete GRPO agent with buffer management.

    Drop-in replacement for PPOAgent in many use cases.
    """

    def __init__(self, config: Optional[GRPOConfig] = None, device: str = 'cpu'):
        self.config = config or GRPOConfig()
        self.device = torch.device(device)

        self.policy = PolicyNetwork().to(self.device)
        self.trainer = GRPOTrainer(self.policy, self.config, device)
        self.buffer = GRPOBuffer()

        self.total_steps = 0
        self.total_episodes = 0

    def select_action(self, state: torch.Tensor, deterministic: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action given state.

        Returns:
            action: Action tensor
            log_prob: Log probability of action
        """
        state = state.to(self.device)
        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(state, deterministic)
        return action, log_prob

    def store_transition(self, state: torch.Tensor, action: torch.Tensor,
                         reward: float, log_prob: torch.Tensor, done: bool):
        """Store a transition."""
        self.buffer.add(state, action, reward, log_prob)
        self.total_steps += 1
        if done:
            self.total_episodes += 1

    def update(self) -> Dict[str, float]:
        """Perform GRPO update."""
        if len(self.buffer) == 0:
            return {}

        states, actions, rewards, log_probs = self.buffer.get_all()
        stats = self.trainer.update_batch(states, actions, rewards, log_probs)

        self.buffer.clear()
        return stats

    def save(self, path: str):
        """Save agent state."""
        self.trainer.save(path)

    def load(self, path: str):
        """Load agent state."""
        self.trainer.load(path)


if __name__ == '__main__':
    print("Testing GRPO Trainer...")

    # Create policy and trainer
    policy = PolicyNetwork()
    config = GRPOConfig(group_size=8, lr=1e-3)
    trainer = GRPOTrainer(policy, config)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Test group sampling
    state = torch.randn(STATE_DIM)
    actions, log_probs = trainer.sample_group_actions(state)
    print(f"\nGroup sampling:")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Log probs shape: {log_probs.shape}")

    # Test advantage computation
    rewards = torch.randn(8)
    advantages = trainer.compute_group_advantages(rewards)
    print(f"\nAdvantage computation:")
    print(f"  Rewards: {rewards.tolist()}")
    print(f"  Advantages: {advantages.tolist()}")

    # Test single state update with mock reward function
    def mock_reward_fn(action):
        # Reward actions that are close to 0.5 (balanced)
        return -((action - 0.5) ** 2).sum().item()

    stats = trainer.update_single_state(state, mock_reward_fn)
    print(f"\nSingle state update stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Test batch update
    states = torch.randn(100, STATE_DIM)
    actions = torch.rand(100, TOTAL_ACTION_DIM)
    rewards = torch.randn(100)
    log_probs = torch.randn(100)

    stats = trainer.update_batch(states, actions, rewards, log_probs)
    print(f"\nBatch update stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Test GRPOAgent
    print("\n" + "="*50)
    print("Testing GRPOAgent...")

    agent = GRPOAgent()

    # Simulate an episode
    for i in range(100):
        state = torch.randn(STATE_DIM)
        action, log_prob = agent.select_action(state)
        reward = float(np.random.randn())
        done = i == 99

        agent.store_transition(state, action, reward, log_prob, done)

    # Update
    stats = agent.update()
    print(f"\nAgent update stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nTotal steps: {agent.total_steps}")
    print(f"Total episodes: {agent.total_episodes}")

    # Test save/load
    agent.save('/tmp/test_grpo.pt')
    agent2 = GRPOAgent()
    agent2.load('/tmp/test_grpo.pt')
    print("\nSave/load test passed")

    print("\nGRPO test passed!")
