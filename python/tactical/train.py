"""
Tactical Layer Training

PPO training for team-level decision making.
Follows the same pattern as the strategic layer training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .model import TacticalNetwork, TacticalAction, TACTICAL_STATE_DIM, TACTICAL_ACTION_DIM
from .state import TacticalState, build_tactical_state
from .rewards import tactical_reward, TacticalRewardConfig, TacticalRewardTracker


@dataclass
class TacticalPPOConfig:
    """PPO hyperparameters for tactical layer."""

    # Learning rates
    lr: float = 3e-4
    lr_decay: float = 0.999

    # PPO specific
    clip_epsilon: float = 0.2
    clip_value: float = 0.2
    entropy_coef: float = 0.02  # Slightly higher for exploration
    entropy_min: float = 0.002
    entropy_decay: float = 0.995
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    batch_size: int = 64
    num_epochs: int = 4
    num_minibatches: int = 4

    # Normalization
    normalize_advantages: bool = True


class TacticalRolloutBuffer:
    """Buffer for storing tactical rollout data."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.positions = []
        self.attitudes = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        self.advantages = None
        self.returns = None

    def add(self, state: torch.Tensor, action: int, position: torch.Tensor,
            attitude: float, reward: float, value: torch.Tensor,
            log_prob: torch.Tensor, done: bool):
        """Add a transition."""
        self.states.append(state.clone())
        self.actions.append(action)
        self.positions.append(position.clone())
        self.attitudes.append(attitude)
        self.rewards.append(reward)
        self.values.append(value.clone())
        self.log_probs.append(log_prob.clone())
        self.dones.append(done)

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
        positions = torch.stack(self.positions)
        attitudes = torch.tensor(self.attitudes, dtype=torch.float32)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.stack(self.values).squeeze()
        advantages = self.advantages
        returns = self.returns

        # FIX P2: Ensure minibatch_size >= 1 to prevent infinite loop
        minibatch_size = max(1, n // num_minibatches)
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_indices = indices[start:end]

            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'positions': positions[batch_indices],
                'attitudes': attitudes[batch_indices],
                'old_log_probs': old_log_probs[batch_indices],
                'old_values': old_values[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices],
            }

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.positions = []
        self.attitudes = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.states)


class TacticalPPOAgent:
    """PPO agent for tactical layer."""

    def __init__(self, config: Optional[TacticalPPOConfig] = None,
                 device: str = 'cpu'):
        self.config = config or TacticalPPOConfig()
        self.device = torch.device(device)

        self.policy = TacticalNetwork().to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay
        )

        self.buffer = TacticalRolloutBuffer()
        self.current_entropy_coef = self.config.entropy_coef

        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0

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
        position = torch.tensor([action_dict['target_x'], action_dict['target_y']])
        attitude = float(action_dict['attitude'])
        action = int(action_dict['action'])

        self.buffer.add(state, action, position, attitude, reward, value, log_prob, done)
        self.total_steps += 1
        if done:
            self.total_episodes += 1

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
                positions = batch['positions'].to(self.device)
                attitudes = batch['attitudes'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)

                log_probs, values, entropy = self.policy.evaluate_actions(
                    states, actions, positions, attitudes
                )

                # Policy loss
                # FIX P1: Clamp exp input to prevent overflow
                ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -20, 20))
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


class SimulatedTacticalEnv:
    """
    Simulated environment for tactical layer training.

    Simulates team combat scenarios without needing the real game.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> TacticalState:
        """Reset to a new scenario."""
        self.step_count = 0
        self.team_health = 1.0
        self.enemy_health = 1.0
        self.objective_progress = 0.0
        self.position = np.array([0.5, 0.5])
        self.enemy_position = np.array([
            self.rng.uniform(0.3, 0.7),
            self.rng.uniform(0.3, 0.7)
        ])
        self.under_fire = False
        self.strategic_aggression = self.rng.uniform(0.2, 0.8)

        return self._get_state()

    def step(self, action_dict: Dict) -> Tuple[TacticalState, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.
        """
        action = action_dict['action'] if isinstance(action_dict['action'], int) else action_dict['action'].item()
        target_x = float(action_dict['target_x'].item()) if hasattr(action_dict['target_x'], 'item') else float(action_dict['target_x'])
        target_y = float(action_dict['target_y'].item()) if hasattr(action_dict['target_y'], 'item') else float(action_dict['target_y'])
        attitude = float(action_dict['attitude'].item()) if hasattr(action_dict['attitude'], 'item') else float(action_dict['attitude'])

        self.step_count += 1
        damage_dealt = 0.0
        damage_taken = 0.0

        # Action effects
        if action == TacticalAction.ATTACK_MOVE:
            self.position = np.array([target_x, target_y])
            dist = np.linalg.norm(self.position - self.enemy_position)
            if dist < 0.2:
                damage_dealt = self.rng.uniform(0.1, 0.2) * (1 + attitude * 0.5)
                damage_taken = self.rng.uniform(0.05, 0.15)
                self.enemy_health -= damage_dealt
                self.team_health -= damage_taken
                self.under_fire = True
                self.objective_progress += 0.05
            else:
                self.under_fire = False

        elif action == TacticalAction.ATTACK_TARGET:
            dist = np.linalg.norm(self.position - self.enemy_position)
            if dist < 0.3:
                damage_dealt = self.rng.uniform(0.15, 0.25) * (1 + attitude * 0.5)
                damage_taken = self.rng.uniform(0.1, 0.2)
                self.enemy_health -= damage_dealt
                self.team_health -= damage_taken
                self.under_fire = True
                self.objective_progress += 0.1

        elif action == TacticalAction.DEFEND_POSITION:
            damage_taken *= 0.5  # Reduced damage when defending
            if self.rng.random() < 0.3:
                damage_dealt = self.rng.uniform(0.05, 0.1)
                damage_taken = self.rng.uniform(0.02, 0.05)
                self.enemy_health -= damage_dealt
                self.team_health -= damage_taken
            self.under_fire = self.rng.random() < 0.2

        elif action == TacticalAction.RETREAT:
            # Move toward base
            self.position = self.position * 0.8  # Move toward (0,0)
            self.under_fire = False
            self.team_health = min(1.0, self.team_health + 0.02)  # Slight heal

        elif action == TacticalAction.HOLD:
            if self.rng.random() < 0.2:
                damage_taken = self.rng.uniform(0.02, 0.08)
                self.team_health -= damage_taken
                self.under_fire = True
            else:
                self.under_fire = False

        elif action == TacticalAction.HUNT:
            # Move toward enemy
            direction = self.enemy_position - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            self.position = np.clip(self.position + direction * 0.1, 0, 1)

            dist = np.linalg.norm(self.position - self.enemy_position)
            if dist < 0.15:
                damage_dealt = self.rng.uniform(0.2, 0.3) * (1 + attitude * 0.5)
                damage_taken = self.rng.uniform(0.15, 0.25)
                self.enemy_health -= damage_dealt
                self.team_health -= damage_taken
                self.under_fire = True
                self.objective_progress += 0.15

        # Clamp values
        self.team_health = np.clip(self.team_health, 0, 1)
        self.enemy_health = np.clip(self.enemy_health, 0, 1)
        self.objective_progress = np.clip(self.objective_progress, 0, 1)

        # Determine done
        done = (
            self.team_health <= 0 or
            self.enemy_health <= 0 or
            self.objective_progress >= 1.0 or
            self.step_count >= 100
        )

        # Calculate reward
        state_dict = self._state_to_dict()
        next_state_dict = self._state_to_dict()
        next_state_dict['damage_dealt'] = damage_dealt
        next_state_dict['damage_taken'] = damage_taken

        strategic_goals = {'aggression': self.strategic_aggression}
        reward = tactical_reward(state_dict, action, next_state_dict, strategic_goals)

        # Terminal bonuses
        if self.enemy_health <= 0:
            reward += 10.0  # Victory
        elif self.team_health <= 0:
            reward -= 10.0  # Defeat

        info = {
            'damage_dealt': damage_dealt,
            'damage_taken': damage_taken,
            'team_health': self.team_health,
            'enemy_health': self.enemy_health,
            'objective_progress': self.objective_progress,
            'won': self.enemy_health <= 0,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> TacticalState:
        """Build current tactical state."""
        game_data = {
            'teams': {
                '0': self._state_to_dict()
            }
        }
        strategic = np.array([0.25, 0.1, 0.4, 0.25, 0.33, 0.33, 0.34, self.strategic_aggression])
        return build_tactical_state(game_data, 0, strategic)

    def _state_to_dict(self) -> Dict:
        """Convert internal state to dict format."""
        dist_to_enemy = np.linalg.norm(self.position - self.enemy_position)
        return {
            'composition': {
                'infantry_count': 5,
                'infantry_health': self.team_health,
                'vehicle_count': 3,
                'vehicle_health': self.team_health,
            },
            'status': {
                'health': self.team_health,
                'under_fire': self.under_fire,
                'cohesion': 0.8,
                'dist_to_objective': 1.0 - self.objective_progress,
            },
            'situational': {
                'nearby_enemies': [
                    0.5 if dist_to_enemy < 0.2 else 0.0,
                    0.0, 0.0, 0.0
                ],
                'threat_level': 0.5 if self.under_fire else 0.2,
                'target_value': 0.8,
            },
            'objective': {
                'type': 'attack',
                'x': self.enemy_position[0],
                'y': self.enemy_position[1],
                'progress': self.objective_progress,
            },
            'temporal': {
                'since_engagement': 5.0 if self.under_fire else 30.0,
            },
        }


def train_tactical(
    num_episodes: int = 1000,
    checkpoint_interval: int = 100,
    checkpoint_dir: str = 'checkpoints/tactical',
    device: str = 'cpu',
):
    """
    Train tactical layer in simulated environment.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    config = TacticalPPOConfig()
    agent = TacticalPPOAgent(config, device)
    env = SimulatedTacticalEnv()

    best_win_rate = 0.0
    recent_wins = []

    for episode in range(num_episodes):
        state = env.reset()
        state_tensor = state.to_tensor()

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
                  f"entropy={stats.get('entropy', 0):.3f}")

        # Checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            path = f"{checkpoint_dir}/tactical_ep{episode + 1}.pt"
            agent.save(path)
            print(f"Saved checkpoint: {path}")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save(f"{checkpoint_dir}/tactical_best.pt")
                print(f"New best win rate: {best_win_rate:.1%}")

    return agent


if __name__ == '__main__':
    print("Training tactical layer...")
    agent = train_tactical(num_episodes=500, checkpoint_interval=50)
    print("Training complete!")
