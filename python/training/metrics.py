"""
Metrics and Monitoring for Generals Zero Hour Learning AI Training

Provides utilities for:
- Tracking training statistics
- Plotting learning curves
- Exporting metrics to various formats
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from collections import deque
import numpy as np


@dataclass
class RunningStats:
    """Track running statistics efficiently."""
    window_size: int = 100
    values: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        self.values = deque(maxlen=self.window_size)

    def add(self, value: float):
        self.values.append(value)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return np.mean(self.values)

    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return np.std(self.values)

    def min(self) -> float:
        if not self.values:
            return 0.0
        return np.min(self.values)

    def max(self) -> float:
        if not self.values:
            return 0.0
        return np.max(self.values)


class TrainingMetrics:
    """Track and aggregate training metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Episode metrics
        self.episode_rewards = RunningStats(window_size)
        self.episode_lengths = RunningStats(window_size)
        self.win_rate = RunningStats(window_size)

        # PPO metrics
        self.policy_losses = RunningStats(window_size)
        self.value_losses = RunningStats(window_size)
        self.entropies = RunningStats(window_size)
        self.kl_divs = RunningStats(window_size)
        self.clip_fractions = RunningStats(window_size)

        # Game-specific metrics
        self.units_killed = RunningStats(window_size)
        self.units_lost = RunningStats(window_size)
        self.army_strength = RunningStats(window_size)
        self.game_times = RunningStats(window_size)

        # Totals
        self.total_episodes = 0
        self.total_steps = 0
        self.total_wins = 0
        self.total_losses = 0

        # History (for plotting)
        self.history: List[Dict] = []

    def log_episode(self, reward: float, length: int, won: Optional[bool] = None,
                    units_killed: int = 0, units_lost: int = 0,
                    army_strength: float = 0.0, game_time: float = 0.0):
        """Log metrics for completed episode."""
        self.total_episodes += 1
        self.total_steps += length

        self.episode_rewards.add(reward)
        self.episode_lengths.add(length)
        self.units_killed.add(units_killed)
        self.units_lost.add(units_lost)
        self.army_strength.add(army_strength)
        self.game_times.add(game_time)

        if won is not None:
            self.win_rate.add(1.0 if won else 0.0)
            if won:
                self.total_wins += 1
            else:
                self.total_losses += 1

    def log_ppo_update(self, stats: Dict[str, float]):
        """Log metrics from PPO update."""
        if 'policy_loss' in stats:
            self.policy_losses.add(stats['policy_loss'])
        if 'value_loss' in stats:
            self.value_losses.add(stats['value_loss'])
        if 'entropy' in stats:
            self.entropies.add(stats['entropy'])
        if 'approx_kl' in stats:
            self.kl_divs.add(stats['approx_kl'])
        if 'clip_fraction' in stats:
            self.clip_fractions.add(stats['clip_fraction'])

    def snapshot(self) -> Dict:
        """Get current metrics snapshot."""
        return {
            'episode': self.total_episodes,
            'total_steps': self.total_steps,
            'reward_mean': self.episode_rewards.mean(),
            'reward_std': self.episode_rewards.std(),
            'episode_length': self.episode_lengths.mean(),
            'win_rate': self.win_rate.mean(),
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'policy_loss': self.policy_losses.mean(),
            'value_loss': self.value_losses.mean(),
            'entropy': self.entropies.mean(),
            'kl_div': self.kl_divs.mean(),
            'clip_fraction': self.clip_fractions.mean(),
            'units_killed': self.units_killed.mean(),
            'units_lost': self.units_lost.mean(),
            'army_strength': self.army_strength.mean(),
            'game_time': self.game_times.mean(),
        }

    def save_snapshot(self):
        """Save current snapshot to history."""
        self.history.append(self.snapshot())

    def print_summary(self):
        """Print current training summary."""
        snap = self.snapshot()
        print(f"\n{'='*60}")
        print(f"Training Summary - Episode {snap['episode']}")
        print(f"{'='*60}")
        print(f"\nPerformance:")
        print(f"  Reward:     {snap['reward_mean']:7.2f} ± {snap['reward_std']:.2f}")
        print(f"  Win Rate:   {snap['win_rate']:7.1%} ({snap['total_wins']}W/{snap['total_losses']}L)")
        print(f"  Ep Length:  {snap['episode_length']:7.1f} steps")
        print(f"  Game Time:  {snap['game_time']:7.1f} min")
        print(f"\nCombat:")
        print(f"  Units Killed: {snap['units_killed']:5.1f}")
        print(f"  Units Lost:   {snap['units_lost']:5.1f}")
        print(f"  Army Ratio:   {snap['army_strength']:5.2f}x")
        print(f"\nPPO:")
        print(f"  Policy Loss:  {snap['policy_loss']:8.4f}")
        print(f"  Value Loss:   {snap['value_loss']:8.4f}")
        print(f"  Entropy:      {snap['entropy']:8.4f}")
        print(f"  KL Div:       {snap['kl_div']:8.4f}")
        print(f"{'='*60}\n")

    def export_history(self, path: str):
        """Export training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot training learning curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if not self.history:
            print("No history to plot")
            return

        episodes = [h['episode'] for h in self.history]
        rewards = [h['reward_mean'] for h in self.history]
        win_rates = [h['win_rate'] for h in self.history]
        policy_losses = [h['policy_loss'] for h in self.history]
        value_losses = [h['value_loss'] for h in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress')

        # Reward
        axes[0, 0].plot(episodes, rewards, 'b-')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3)

        # Win rate
        axes[0, 1].plot(episodes, win_rates, 'g-')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_title('Win Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        # Policy loss
        axes[1, 0].plot(episodes, policy_losses, 'r-')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Value loss
        axes[1, 1].plot(episodes, value_losses, 'm-')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Value Loss')
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def load_training_logs(log_path: str) -> List[Dict]:
    """Load training logs from JSONL file."""
    logs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def analyze_training_run(log_path: str):
    """Analyze and summarize a training run from logs."""
    logs = load_training_logs(log_path)

    if not logs:
        print("No logs found")
        return

    rewards = [l['total_reward'] for l in logs if 'total_reward' in l]
    wins = [l['won'] for l in logs if 'won' in l and l['won'] is not None]

    print(f"\nTraining Run Analysis: {log_path}")
    print(f"{'='*50}")
    print(f"Total episodes: {len(logs)}")

    if rewards:
        print(f"\nRewards:")
        print(f"  Mean:  {np.mean(rewards):.2f}")
        print(f"  Std:   {np.std(rewards):.2f}")
        print(f"  Min:   {np.min(rewards):.2f}")
        print(f"  Max:   {np.max(rewards):.2f}")

    if wins:
        win_count = sum(wins)
        print(f"\nWin Rate: {win_count}/{len(wins)} = {win_count/len(wins):.1%}")

        # Win rate over time (last 100 episodes)
        if len(wins) >= 100:
            recent_wins = wins[-100:]
            print(f"Recent (last 100): {sum(recent_wins)/len(recent_wins):.1%}")


if __name__ == '__main__':
    # Test metrics
    print("Testing TrainingMetrics...")

    metrics = TrainingMetrics(window_size=10)

    # Simulate some episodes
    for i in range(50):
        reward = np.random.randn() * 2 + i * 0.1  # Improving reward
        won = np.random.random() > (0.7 - i * 0.01)  # Improving win rate

        metrics.log_episode(
            reward=reward,
            length=100 + np.random.randint(-20, 20),
            won=won,
            units_killed=np.random.randint(0, 20),
            units_lost=np.random.randint(0, 15),
            army_strength=0.8 + np.random.random() * 0.6,
            game_time=5 + np.random.random() * 10,
        )

        # Log PPO update every 5 episodes
        if i % 5 == 0:
            metrics.log_ppo_update({
                'policy_loss': 0.5 - i * 0.005,
                'value_loss': 1.0 - i * 0.01,
                'entropy': 2.0 - i * 0.02,
                'approx_kl': 0.01 + np.random.random() * 0.01,
                'clip_fraction': 0.1 + np.random.random() * 0.1,
            })
            metrics.save_snapshot()

    metrics.print_summary()
    print("\nMetrics test passed!")
