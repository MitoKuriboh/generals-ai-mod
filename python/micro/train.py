"""
Micro Layer PPO Training

PPO training for unit-level micro control.
Can start from imitation-learned weights or from scratch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .model import MicroNetwork, MicroAction, MICRO_STATE_DIM, MICRO_ACTION_DIM
from .state import MicroState, build_micro_state
from .rewards import micro_reward, MicroRewardConfig, MicroRewardTracker


@dataclass
class MicroPPOConfig:
    """PPO hyperparameters for micro layer."""

    # Learning rates (lower for fine-tuning after imitation)
    lr: float = 1e-4
    lr_decay: float = 0.9995

    # PPO specific
    clip_epsilon: float = 0.2
    clip_value: float = 0.2
    entropy_coef: float = 0.01
    entropy_min: float = 0.001
    entropy_decay: float = 0.998
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # GAE
    gamma: float = 0.98  # Shorter horizon for micro (faster feedback)
    gae_lambda: float = 0.95

    # Training
    batch_size: int = 128
    num_epochs: int = 4
    num_minibatches: int = 4
    sequence_length: int = 8  # For LSTM training

    # Normalization
    normalize_advantages: bool = True


class MicroRolloutBuffer:
    """Buffer for storing micro rollout sequences."""

    def __init__(self, sequence_length: int = 8):
        self.sequence_length = sequence_length

        self.states = []
        self.actions = []
        self.moves = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.hidden_states = []  # LSTM states

        self.advantages = None
        self.returns = None

    def add(self, state: torch.Tensor, action: int, move: torch.Tensor,
            reward: float, value: torch.Tensor, log_prob: torch.Tensor,
            done: bool, hidden: Optional[Tuple] = None):
        """Add a transition."""
        self.states.append(state.clone())
        self.actions.append(action)
        self.moves.append(move.clone())
        self.rewards.append(reward)
        self.values.append(value.clone())
        self.log_probs.append(log_prob.clone())
        self.dones.append(done)
        if hidden is not None:
            self.hidden_states.append((hidden[0].clone(), hidden[1].clone()))

    def compute_returns_and_advantages(self, last_value: torch.Tensor,
                                       gamma: float, gae_lambda: float):
        """Compute returns and GAE advantages."""
        n = len(self.rewards)
        self.advantages = torch.zeros(n)
        self.returns = torch.zeros(n)

        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values).squeeze()
        dones = torch.tensor(self.dones, dtype=torch.float32)

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
        """Generate minibatches for training."""
        n = len(self.states)
        indices = np.random.permutation(n)

        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        moves = torch.stack(self.moves)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.stack(self.values).squeeze()
        advantages = self.advantages
        returns = self.returns

        minibatch_size = n // num_minibatches
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_indices = indices[start:end]

            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'moves': moves[batch_indices],
                'old_log_probs': old_log_probs[batch_indices],
                'old_values': old_values[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices],
            }

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.moves = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.hidden_states = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.states)


class MicroPPOAgent:
    """PPO agent for micro layer with LSTM support."""

    def __init__(self, config: Optional[MicroPPOConfig] = None,
                 device: str = 'cpu'):
        self.config = config or MicroPPOConfig()
        self.device = torch.device(device)

        self.policy = MicroNetwork().to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay
        )

        self.buffer = MicroRolloutBuffer(self.config.sequence_length)
        self.current_entropy_coef = self.config.entropy_coef

        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0

    def reset_hidden(self, batch_size: int = 1):
        """Reset LSTM hidden state."""
        self.policy.reset_hidden(batch_size, self.device)

    def select_action(self, state: torch.Tensor, deterministic: bool = False
                      ) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Select action given state."""
        state = state.to(self.device)
        with torch.no_grad():
            action_dict, log_prob, value = self.policy.get_action(state, deterministic)
        return action_dict, log_prob, value

    def store_transition(self, state: torch.Tensor, action_dict: Dict,
                         reward: float, value: torch.Tensor,
                         log_prob: torch.Tensor, done: bool):
        """Store a transition in the buffer."""
        # Normalize angle back to [0, 1] for storage
        angle_normalized = (action_dict['move_angle'] / np.pi + 1) / 2
        if hasattr(angle_normalized, 'item'):
            angle_normalized = angle_normalized.item()
        distance = action_dict['move_distance']
        if hasattr(distance, 'item'):
            distance = distance.item()

        move = torch.tensor([angle_normalized, distance], dtype=torch.float32)
        action = int(action_dict['action'].item()) if hasattr(action_dict['action'], 'item') else int(action_dict['action'])

        hidden = self.policy.hidden
        self.buffer.add(state, action, move, reward, value, log_prob, done, hidden)

        self.total_steps += 1
        if done:
            self.total_episodes += 1
            self.reset_hidden(1)

    def update(self, last_value: torch.Tensor) -> Dict[str, float]:
        """Perform PPO update."""
        if len(self.buffer) == 0:
            return {}

        self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        if self.config.normalize_advantages:
            advantages = self.buffer.advantages
            self.buffer.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'approx_kl': 0,
            'clip_fraction': 0,
        }
        num_updates = 0

        for _ in range(self.config.num_epochs):
            for batch in self.buffer.get_batches(
                self.config.batch_size, self.config.num_minibatches
            ):
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                moves = batch['moves'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)

                # Reset hidden state for batch processing
                self.policy.reset_hidden(states.size(0), self.device)

                log_probs, values, entropy = self.policy.evaluate_actions(
                    states, actions, moves
                )

                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * advantages, clipped_ratio * advantages
                ).mean()

                # Value loss
                if self.config.clip_value > 0:
                    old_values = batch['old_values'].to(self.device)
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
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
                    self.current_entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += (-entropy_loss).item()
                stats['approx_kl'] += approx_kl.item()
                stats['clip_fraction'] += clip_fraction.item()
                num_updates += 1

        for key in stats:
            stats[key] /= max(num_updates, 1)

        self.scheduler.step()
        stats['learning_rate'] = self.scheduler.get_last_lr()[0]

        self.current_entropy_coef = max(
            self.config.entropy_min,
            self.current_entropy_coef * self.config.entropy_decay
        )
        stats['entropy_coef'] = self.current_entropy_coef

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
            'current_entropy_coef': self.current_entropy_coef,
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
        self.current_entropy_coef = checkpoint.get('current_entropy_coef',
                                                    self.config.entropy_coef)


class SimulatedMicroEnv:
    """
    Simulated environment for micro layer training.

    Simulates 1v1 unit combat with kiting and ability mechanics.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> MicroState:
        """Reset to new combat scenario."""
        self.step_count = 0

        # Unit stats
        self.health = 1.0
        self.ammo = 1.0
        self.ability_cooldown = 0.0
        self.position = np.array([0.3, 0.5])

        # Enemy stats
        self.enemy_health = 1.0
        self.enemy_position = np.array([0.7, 0.5])
        self.enemy_range = 0.2
        self.enemy_dps = 0.15

        # Combat state
        self.under_fire = False
        self.time_in_combat = 0.0

        return self._get_state()

    def step(self, action_dict: Dict) -> Tuple[MicroState, float, bool, Dict]:
        """Execute action and return next state."""
        action = action_dict['action'] if isinstance(action_dict['action'], int) else action_dict['action'].item()
        move_angle = float(action_dict['move_angle'].item()) if hasattr(action_dict['move_angle'], 'item') else float(action_dict['move_angle'])
        move_dist = float(action_dict['move_distance'].item()) if hasattr(action_dict['move_distance'], 'item') else float(action_dict['move_distance'])

        self.step_count += 1
        damage_dealt = 0.0
        damage_taken = 0.0

        # Calculate distance to enemy
        dist_to_enemy = np.linalg.norm(self.position - self.enemy_position)

        # Process action
        if action in [MicroAction.ATTACK_CURRENT, MicroAction.ATTACK_NEAREST,
                      MicroAction.ATTACK_WEAKEST, MicroAction.ATTACK_PRIORITY]:
            # Attack
            if dist_to_enemy < 0.3 and self.ammo > 0:
                damage_dealt = self.rng.uniform(0.1, 0.2)
                self.enemy_health -= damage_dealt
                self.ammo -= 0.1

        elif action == MicroAction.MOVE_FORWARD:
            # Move toward enemy
            direction = self.enemy_position - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            self.position = np.clip(self.position + direction * move_dist * 0.1, 0, 1)

        elif action == MicroAction.MOVE_BACKWARD:
            # Move away from enemy
            direction = self.position - self.enemy_position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            self.position = np.clip(self.position + direction * move_dist * 0.1, 0, 1)

        elif action == MicroAction.MOVE_FLANK:
            # Move perpendicular
            direction = self.enemy_position - self.position
            perpendicular = np.array([-direction[1], direction[0]])
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
            self.position = np.clip(self.position + perpendicular * move_dist * 0.1, 0, 1)

        elif action == MicroAction.USE_ABILITY:
            # Special ability
            if self.ability_cooldown <= 0:
                damage_dealt = 0.3
                self.enemy_health -= damage_dealt
                self.ability_cooldown = 30

        elif action == MicroAction.RETREAT:
            # Full retreat toward spawn
            direction = np.array([0, 0.5]) - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            self.position = np.clip(self.position + direction * 0.15, 0, 1)

        # Enemy AI: simple approach and attack
        dist_to_enemy = np.linalg.norm(self.position - self.enemy_position)

        if dist_to_enemy > self.enemy_range:
            # Move toward player
            enemy_dir = self.position - self.enemy_position
            if np.linalg.norm(enemy_dir) > 0:
                enemy_dir = enemy_dir / np.linalg.norm(enemy_dir)
            self.enemy_position = np.clip(self.enemy_position + enemy_dir * 0.05, 0, 1)
            self.under_fire = False
        else:
            # Attack
            damage_taken = self.enemy_dps * self.rng.uniform(0.8, 1.2)
            self.health -= damage_taken
            self.under_fire = True

        # Update cooldowns
        if self.ability_cooldown > 0:
            self.ability_cooldown -= 1

        # Track combat time
        if self.under_fire:
            self.time_in_combat += 1

        # Clamp values
        self.health = np.clip(self.health, 0, 1)
        self.enemy_health = np.clip(self.enemy_health, 0, 1)
        self.ammo = np.clip(self.ammo, 0, 1)

        # Check done
        done = (
            self.health <= 0 or
            self.enemy_health <= 0 or
            self.step_count >= 100
        )

        # Calculate reward
        state_dict = self._state_to_dict()
        next_state_dict = self._state_to_dict()
        next_state_dict['damage_dealt'] = damage_dealt
        next_state_dict['damage_taken'] = damage_taken

        reward = micro_reward(state_dict, action, next_state_dict, {})

        # Terminal bonuses
        if self.enemy_health <= 0:
            reward += 5.0
        elif self.health <= 0:
            reward -= 5.0

        info = {
            'damage_dealt': damage_dealt,
            'damage_taken': damage_taken,
            'health': self.health,
            'enemy_health': self.enemy_health,
            'won': self.enemy_health <= 0,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> MicroState:
        """Build current micro state."""
        return build_micro_state(self._state_to_dict())

    def _state_to_dict(self) -> Dict:
        """Convert internal state to dict."""
        dist_to_enemy = np.linalg.norm(self.position - self.enemy_position)
        angle_to_enemy = np.arctan2(
            self.enemy_position[1] - self.position[1],
            self.enemy_position[0] - self.position[0]
        )

        return {
            'type': 'tank',
            'health': self.health,
            'ammunition': self.ammo,
            'speed': 50.0,
            'range': 200.0,
            'dps': 20.0,
            'situational': {
                'enemy_dist': dist_to_enemy * 500,
                'enemy_angle': angle_to_enemy,
                'enemy_health': self.enemy_health,
                'enemy_threat': 0.7 if self.under_fire else 0.3,
                'under_fire': self.under_fire,
                'ability_ready': self.ability_cooldown <= 0,
                'can_retreat': True,
            },
            'target': {
                'dist': dist_to_enemy * 500,
                'health': self.enemy_health,
                'type': 'tank',
                'dps': self.enemy_dps * 50,
            },
            'temporal': {
                'since_hit': 0.5 if self.under_fire else 30.0,
                'in_combat': self.time_in_combat,
            },
        }


def train_micro(
    num_episodes: int = 1000,
    checkpoint_interval: int = 100,
    checkpoint_dir: str = 'checkpoints/micro',
    pretrained_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Train micro layer with PPO.

    Can optionally start from pretrained (imitation-learned) weights.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    config = MicroPPOConfig()
    agent = MicroPPOAgent(config, device)

    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        agent.policy.load_state_dict(
            torch.load(pretrained_path, map_location=device, weights_only=False)['state_dict']
        )

    env = SimulatedMicroEnv()

    best_win_rate = 0.0
    recent_wins = []

    for episode in range(num_episodes):
        state = env.reset()
        state_tensor = state.to_tensor()
        agent.reset_hidden(1)

        episode_reward = 0.0
        done = False

        while not done:
            action_dict, log_prob, value = agent.select_action(state_tensor)

            next_state, reward, done, info = env.step(action_dict)
            next_state_tensor = next_state.to_tensor()

            agent.store_transition(
                state_tensor, action_dict, reward, value, log_prob, done
            )

            state_tensor = next_state_tensor
            episode_reward += reward

        # Update policy
        with torch.no_grad():
            _, _, last_value = agent.select_action(state_tensor)
        stats = agent.update(last_value)

        # Track wins
        recent_wins.append(1.0 if info.get('won', False) else 0.0)
        if len(recent_wins) > 100:
            recent_wins.pop(0)
        win_rate = sum(recent_wins) / len(recent_wins)

        # Logging
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: "
                  f"reward={episode_reward:.2f}, "
                  f"win_rate={win_rate:.1%}, "
                  f"health={info.get('health', 0):.2f}")

        # Checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            path = f"{checkpoint_dir}/micro_ep{episode + 1}.pt"
            agent.save(path)
            print(f"Saved checkpoint: {path}")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save(f"{checkpoint_dir}/micro_best.pt")
                print(f"New best win rate: {best_win_rate:.1%}")

    return agent


if __name__ == '__main__':
    print("Training micro layer...")
    agent = train_micro(num_episodes=500, checkpoint_interval=50)
    print("Training complete!")
