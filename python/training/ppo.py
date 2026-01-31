"""
Proximal Policy Optimization (PPO) for Generals Zero Hour Learning AI

Implements PPO-Clip algorithm for training the policy network.
Reference: https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .model import PolicyNetwork, STATE_DIM, TOTAL_ACTION_DIM


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Learning rates
    lr: float = 3e-4
    lr_decay: float = 0.999

    # PPO specific
    clip_epsilon: float = 0.2
    clip_value: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # GAE (Generalized Advantage Estimation)
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda

    # Training
    batch_size: int = 64
    num_epochs: int = 4          # PPO epochs per update
    num_minibatches: int = 4

    # Normalization
    normalize_advantages: bool = True
    normalize_returns: bool = False


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO updates.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        self.advantages = None
        self.returns = None

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float,
            value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """Add a transition to the buffer."""
        self.states.append(state.clone())
        self.actions.append(action.clone())
        self.rewards.append(reward)
        self.values.append(value.clone())
        self.log_probs.append(log_prob.clone())
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: torch.Tensor,
                                        gamma: float, gae_lambda: float):
        """
        Compute returns and advantages using GAE.
        """
        n = len(self.rewards)
        self.advantages = torch.zeros(n)
        self.returns = torch.zeros(n)

        # Convert to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values).squeeze()
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # GAE computation
        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value.item()
                next_non_terminal = 1.0 - dones[t].item()
            else:
                next_value = values[t + 1].item()
                next_non_terminal = 1.0 - dones[t].item()

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + values

    def get_batches(self, batch_size: int, num_minibatches: int):
        """
        Generate minibatches for training.
        """
        n = len(self.states)
        indices = np.random.permutation(n)

        # Stack tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        advantages = self.advantages
        returns = self.returns

        # Generate minibatches
        minibatch_size = n // num_minibatches
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_indices = indices[start:end]

            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'old_log_probs': old_log_probs[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices],
            }

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO Agent for training and inference.
    """

    def __init__(self, config: Optional[PPOConfig] = None, device: str = 'cpu'):
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        # Initialize policy network
        self.policy = PolicyNetwork().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0

    def select_action(self, state: torch.Tensor, deterministic: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given state.
        """
        state = state.to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state, deterministic)
        return action, log_prob, value

    def store_transition(self, state: torch.Tensor, action: torch.Tensor,
                         reward: float, value: torch.Tensor,
                         log_prob: torch.Tensor, done: bool):
        """Store a transition in the buffer."""
        self.buffer.add(state, action, reward, value, log_prob, done)
        self.total_steps += 1
        if done:
            self.total_episodes += 1

    def update(self, last_value: torch.Tensor) -> Dict[str, float]:
        """
        Perform PPO update.

        Returns:
            Dictionary of training statistics
        """
        if len(self.buffer) == 0:
            return {}

        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = self.buffer.advantages
            self.buffer.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training statistics
        stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'approx_kl': 0,
            'clip_fraction': 0,
        }
        num_updates = 0

        # PPO epochs
        for _ in range(self.config.num_epochs):
            for batch in self.buffer.get_batches(
                self.config.batch_size, self.config.num_minibatches
            ):
                # Move to device
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

                # Policy loss (PPO-Clip)
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages
                ).mean()

                # Value loss
                if self.config.clip_value > 0:
                    # Clipped value loss
                    values_clipped = batch['returns'].to(self.device) + torch.clamp(
                        values - batch['returns'].to(self.device),
                        -self.config.clip_value,
                        self.config.clip_value
                    )
                    value_loss = torch.max(
                        (values - returns) ** 2,
                        (values_clipped - returns) ** 2
                    ).mean()
                else:
                    value_loss = ((values - returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += (-entropy_loss).item()
                stats['approx_kl'] += approx_kl.item()
                stats['clip_fraction'] += clip_fraction.item()
                num_updates += 1

        # Average statistics
        for key in stats:
            stats[key] /= max(num_updates, 1)

        # Learning rate decay
        self.scheduler.step()
        stats['learning_rate'] = self.scheduler.get_last_lr()[0]

        # Clear buffer
        self.buffer.clear()
        self.update_count += 1

        return stats

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'update_count': self.update_count,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.update_count = checkpoint.get('update_count', 0)


if __name__ == '__main__':
    # Test PPO agent
    print("Testing PPO Agent...")

    config = PPOConfig()
    agent = PPOAgent(config)

    print(f"Config: {config}")

    # Simulate a rollout
    for i in range(100):
        state = torch.randn(STATE_DIM)
        action, log_prob, value = agent.select_action(state)

        # Simulate reward
        reward = np.random.randn() * 0.1
        done = i == 99

        agent.store_transition(state, action, reward, value, log_prob, done)

    # Perform update
    last_value = torch.tensor([0.0])
    stats = agent.update(last_value)

    print(f"\nUpdate stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nTotal steps: {agent.total_steps}")
    print(f"Total episodes: {agent.total_episodes}")

    print("\nPPO test passed!")
