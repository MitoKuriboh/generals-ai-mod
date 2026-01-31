#!/usr/bin/env python3
"""
Training Script for Generals Zero Hour Learning AI

Main training loop using PPO with the game environment.
Supports both real game training and simulated environment testing.

Usage:
    # Simulated training (for testing)
    python -m training.train --simulated --episodes 100

    # Real game training (requires game running)
    python -m training.train --episodes 1000 --checkpoint-dir checkpoints

    # Resume from checkpoint
    python -m training.train --resume checkpoints/agent_ep500.pt
"""

import argparse
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np

from .ppo import PPOAgent, PPOConfig
from .env import GeneralsEnv, SimulatedEnv
from .rewards import RewardConfig, get_config as get_reward_config
from .model import STATE_DIM


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training parameters
    total_episodes: int = 1000
    steps_per_update: int = 512      # Collect this many steps before PPO update
    eval_interval: int = 50          # Evaluate every N episodes
    checkpoint_interval: int = 100   # Save checkpoint every N episodes
    log_interval: int = 10           # Log stats every N episodes

    # Environment
    simulated: bool = False          # Use simulated env for testing
    sim_episode_length: int = 100    # Steps per simulated episode

    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    # Reward config preset
    reward_preset: str = 'balanced'


@dataclass
class EpisodeLog:
    """Log entry for a single episode."""
    episode: int
    total_reward: float
    steps: int
    won: Optional[bool]
    game_time: float
    units_killed: int
    units_lost: int
    buildings_destroyed: int
    buildings_lost: int
    final_army_strength: float
    timestamp: str


class Trainer:
    """Main training class."""

    def __init__(self, config: TrainingConfig, ppo_config: Optional[PPOConfig] = None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Trainer] Using device: {self.device}")

        # Initialize PPO agent
        ppo_config = ppo_config or PPOConfig()
        self.agent = PPOAgent(ppo_config, device=self.device)

        # Initialize environment
        if config.simulated:
            print("[Trainer] Using simulated environment")
            self.env = SimulatedEnv(episode_length=config.sim_episode_length)
        else:
            print("[Trainer] Using real game environment")
            self.env = GeneralsEnv()

        # Reward configuration
        self.reward_config = get_reward_config(config.reward_preset)

        # Tracking
        self.episode_logs: List[EpisodeLog] = []
        self.best_avg_reward = float('-inf')
        self.wins = 0
        self.losses = 0

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train(self, resume_path: Optional[str] = None):
        """Main training loop."""
        start_episode = 1

        # Resume from checkpoint if specified
        if resume_path:
            start_episode = self._load_checkpoint(resume_path)
            print(f"[Trainer] Resumed from episode {start_episode}")

        print(f"\n{'='*60}")
        print(f"Training: {self.config.total_episodes} episodes")
        print(f"Steps per update: {self.config.steps_per_update}")
        print(f"Reward preset: {self.config.reward_preset}")
        print(f"{'='*60}\n")

        # Setup environment connection (for real env)
        if not self.config.simulated:
            if not self._setup_connection():
                print("[Trainer] Failed to establish game connection")
                return

        total_steps = 0
        episode_rewards = []

        for episode in range(start_episode, self.config.total_episodes + 1):
            episode_reward = self._run_episode()
            episode_rewards.append(episode_reward)
            total_steps += self.agent.total_steps

            # Logging
            if episode % self.config.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.config.log_interval:])
                win_rate = self.wins / max(self.wins + self.losses, 1)
                print(f"[Ep {episode:4d}] "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | "
                      f"Steps: {self.agent.total_steps:6d} | "
                      f"WinRate: {win_rate:5.1%} | "
                      f"LR: {self.agent.scheduler.get_last_lr()[0]:.2e}")

            # Checkpoint
            if episode % self.config.checkpoint_interval == 0:
                self._save_checkpoint(episode)

            # Evaluation
            if episode % self.config.eval_interval == 0:
                self._evaluate(episode)

        # Final save
        self._save_checkpoint(self.config.total_episodes, final=True)
        self._save_logs()

        print(f"\n[Trainer] Training complete!")
        print(f"Total steps: {self.agent.total_steps}")
        print(f"Final win rate: {self.wins / max(self.wins + self.losses, 1):.1%}")

    def _run_episode(self) -> float:
        """Run a single episode and collect experience."""
        state, info = self.env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Select action
            action, log_prob, value = self.agent.select_action(state)

            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            episode_reward += reward
            state = next_state

            # PPO update when enough steps collected
            if len(self.agent.buffer) >= self.config.steps_per_update:
                # Get value estimate for last state
                with torch.no_grad():
                    _, _, last_value = self.agent.select_action(state)
                stats = self.agent.update(last_value)

        # Track win/loss
        episode_stats = info.get('episode_stats')
        if episode_stats and episode_stats.won is not None:
            if episode_stats.won:
                self.wins += 1
            else:
                self.losses += 1

            # Log episode
            self.episode_logs.append(EpisodeLog(
                episode=len(self.episode_logs) + 1,
                total_reward=episode_stats.total_reward,
                steps=episode_stats.steps,
                won=episode_stats.won,
                game_time=episode_stats.game_time,
                units_killed=episode_stats.units_killed,
                units_lost=episode_stats.units_lost,
                buildings_destroyed=episode_stats.buildings_destroyed,
                buildings_lost=episode_stats.buildings_lost,
                final_army_strength=episode_stats.final_army_strength,
                timestamp=datetime.now().isoformat(),
            ))

        return episode_reward

    def _setup_connection(self) -> bool:
        """Setup connection to game (for real environment)."""
        if self.config.simulated:
            return True

        print("[Trainer] Creating named pipe...")
        if not self.env.create_pipe():
            return False

        print("[Trainer] Waiting for game to connect...")
        print("  Please start a skirmish game with Learning AI")
        return self.env.wait_for_connection(timeout=120.0)

    def _evaluate(self, episode: int):
        """Run evaluation episodes."""
        print(f"\n[Eval] Episode {episode}")
        recent_rewards = [log.total_reward for log in self.episode_logs[-50:]]
        if recent_rewards:
            avg = np.mean(recent_rewards)
            std = np.std(recent_rewards)
            print(f"  Avg reward (last 50): {avg:.2f} ± {std:.2f}")

            if avg > self.best_avg_reward:
                self.best_avg_reward = avg
                self._save_checkpoint(episode, best=True)
                print(f"  New best! Saved.")

        recent_wins = [log.won for log in self.episode_logs[-50:] if log.won is not None]
        if recent_wins:
            win_rate = sum(recent_wins) / len(recent_wins)
            print(f"  Win rate (last 50): {win_rate:.1%}")

        print()

    def _save_checkpoint(self, episode: int, best: bool = False, final: bool = False):
        """Save training checkpoint."""
        if best:
            path = os.path.join(self.config.checkpoint_dir, 'best_agent.pt')
        elif final:
            path = os.path.join(self.config.checkpoint_dir, 'final_agent.pt')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'agent_ep{episode}.pt')

        # Save agent
        self.agent.save(path)

        # Save training state
        state_path = path.replace('.pt', '_state.json')
        state = {
            'episode': episode,
            'wins': self.wins,
            'losses': self.losses,
            'best_avg_reward': self.best_avg_reward,
            'config': asdict(self.config),
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return start episode."""
        self.agent.load(path)

        # Try to load training state
        state_path = path.replace('.pt', '_state.json')
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
                self.wins = state.get('wins', 0)
                self.losses = state.get('losses', 0)
                self.best_avg_reward = state.get('best_avg_reward', float('-inf'))
                return state.get('episode', 0) + 1

        return 1

    def _save_logs(self):
        """Save episode logs to file."""
        log_path = os.path.join(
            self.config.log_dir,
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(log_path, 'w') as f:
            for log in self.episode_logs:
                f.write(json.dumps(asdict(log)) + '\n')
        print(f"[Trainer] Logs saved to {log_path}")

    def close(self):
        """Clean up resources."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description='Train Generals Zero Hour Learning AI')

    # Training args
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Total episodes to train')
    parser.add_argument('--steps-per-update', type=int, default=512,
                        help='Steps to collect before PPO update')

    # Environment args
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated environment for testing')
    parser.add_argument('--sim-length', type=int, default=100,
                        help='Steps per simulated episode')

    # Checkpoint args
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    # Reward args
    parser.add_argument('--reward-preset', type=str, default='balanced',
                        choices=['exploration', 'balanced', 'sparse', 'aggressive'],
                        help='Reward configuration preset')

    # PPO args
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clip epsilon')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')

    args = parser.parse_args()

    # Build configs
    training_config = TrainingConfig(
        total_episodes=args.episodes,
        steps_per_update=args.steps_per_update,
        simulated=args.simulated,
        sim_episode_length=args.sim_length,
        checkpoint_dir=args.checkpoint_dir,
        reward_preset=args.reward_preset,
    )

    ppo_config = PPOConfig(
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        gamma=args.gamma,
    )

    # Create trainer and run
    trainer = Trainer(training_config, ppo_config)
    try:
        trainer.train(resume_path=args.resume)
    except KeyboardInterrupt:
        print("\n[Trainer] Interrupted by user")
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
