#!/usr/bin/env python3
"""
Base trainer class with common PPO training logic.

All training modes inherit from this class and implement
the mode-specific run_episode() method.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

import torch

from ..ppo import PPOAgent, PPOConfig
from ..model import state_dict_to_tensor, action_tensor_to_dict
from ..rewards import calculate_step_reward, RewardConfig
from ..config import WIN_REWARD, LOSS_REWARD, PROTOCOL_VERSION


@dataclass
class TrainingStats:
    """Training statistics across all episodes."""
    episodes_completed: int = 0
    episodes_failed: int = 0
    wins: int = 0
    losses: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    best_win_rate: float = 0.0


@dataclass
class EpisodeResult:
    """Result of a single training episode."""
    victory: bool
    game_time: float
    final_army_strength: float
    steps: int
    reward: float
    states: List[Dict] = None
    actions: List[Dict] = None


def wrap_recommendation_with_capabilities(recommendation: Dict) -> Dict:
    """
    Wrap a strategic recommendation with capabilities declaration.

    Strategic-only servers declare hierarchical=false so the game
    doesn't try to use tactical/micro layers.
    """
    return {
        'version': PROTOCOL_VERSION,
        'capabilities': {
            'hierarchical': False,
            'tactical': False,
            'micro': False,
        },
        **recommendation
    }


def validate_protocol_version(state: Dict) -> bool:
    """Validate protocol version from game state. Returns True if valid."""
    version = state.get('version', 1)
    if version != PROTOCOL_VERSION:
        logging.warning(f"Protocol mismatch: expected {PROTOCOL_VERSION}, got {version}")
        return False
    return True


class BaseTrainer(ABC):
    """
    Base class for all training modes.

    Provides common functionality:
    - PPO agent management
    - Checkpoint save/load
    - Stats tracking
    - Logging

    Subclasses must implement:
    - run_episode(): Mode-specific episode execution
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        learning_rate: float = 3e-4,
        verbose: bool = False,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.verbose = verbose

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # PPO agent
        ppo_config = PPOConfig(lr=learning_rate)
        self.agent = PPOAgent(ppo_config, device=self.device)

        # Reward configuration
        self.reward_config = RewardConfig()

        # Training state
        self.stats = TrainingStats()
        self.episode_logs: List[Dict] = []

        # PPO state for current step
        self._current_log_prob = None
        self._current_value = None
        self._current_action = None
        self._prev_state = None

    @abstractmethod
    def run_episode(self) -> Optional[EpisodeResult]:
        """
        Run a single training episode.

        Returns:
            EpisodeResult if episode completed successfully, None if failed/disconnected.
        """
        pass

    @abstractmethod
    def setup(self) -> bool:
        """
        Perform any setup needed before training.

        Returns:
            True if setup successful, False otherwise.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources after training."""
        pass

    def get_recommendation(self, state: Dict) -> Dict:
        """Get recommendation from PPO agent."""
        state_tensor = state_dict_to_tensor(state)

        action, log_prob, value = self.agent.select_action(state_tensor)

        self._current_log_prob = log_prob
        self._current_value = value
        self._current_action = action

        return action_tensor_to_dict(action)

    def calculate_reward(self, prev_state: Optional[Dict], state: Dict) -> float:
        """Calculate step reward using unified rewards module."""
        return calculate_step_reward(prev_state, state, self.reward_config)

    def store_transition(self, state: Dict, reward: float, done: bool = False):
        """Store transition in PPO buffer."""
        state_tensor = state_dict_to_tensor(state)
        self.agent.store_transition(
            state_tensor,
            self._current_action,
            reward,
            self._current_value,
            self._current_log_prob,
            done=done
        )

    def store_terminal_transition(self, state: Dict, victory: bool):
        """Store terminal transition with win/loss reward."""
        terminal_reward = WIN_REWARD if victory else LOSS_REWARD
        state_tensor = state_dict_to_tensor(state)
        self.agent.store_transition(
            state_tensor,
            self._current_action,
            terminal_reward,
            torch.tensor([0.0]),  # Terminal value = 0
            self._current_log_prob,
            done=True
        )

    def maybe_update(self, state: Dict, threshold: int = 256) -> Optional[float]:
        """
        Perform PPO update if buffer has enough transitions.

        Returns:
            Loss value if update performed, None otherwise.
        """
        if len(self.agent.buffer) >= threshold:
            state_tensor = state_dict_to_tensor(state)
            with torch.no_grad():
                _, _, last_value = self.agent.select_action(state_tensor)
            try:
                return self.agent.update(last_value)
            except Exception as e:
                # Save emergency checkpoint on training error
                logging.error(f"PPO update error: {e}")
                emergency_path = os.path.join(
                    self.checkpoint_dir, 'emergency_checkpoint.pt'
                )
                self.agent.save(emergency_path)
                logging.info(f"Emergency checkpoint saved to {emergency_path}")
                raise
        return None

    def final_update(self):
        """Perform final PPO update with terminal state."""
        if len(self.agent.buffer) > 0:
            try:
                self.agent.update(torch.tensor([0.0]))
            except Exception as e:
                logging.error(f"Final PPO update error: {e}")
                emergency_path = os.path.join(
                    self.checkpoint_dir, 'emergency_checkpoint.pt'
                )
                self.agent.save(emergency_path)
                raise

    def train(
        self,
        num_episodes: int,
        checkpoint_interval: int = 10,
        resume_path: str = None,
    ):
        """
        Main training loop.

        Args:
            num_episodes: Target number of episodes
            checkpoint_interval: Save every N episodes
            resume_path: Optional checkpoint to resume from
        """
        # Resume from checkpoint
        if resume_path:
            start_episode = self._load_checkpoint(resume_path)
            print(f"[Training] Resumed from episode {start_episode}")

        print(f"\n{'='*60}")
        print(f"  {self.__class__.__name__}")
        print(f"  Device: {self.device}")
        print(f"  Target Episodes: {num_episodes}")
        print(f"  Checkpoint Interval: {checkpoint_interval}")
        print(f"{'='*60}\n")

        # Mode-specific setup
        if not self.setup():
            print("[Training] Setup failed")
            return

        try:
            while self.stats.episodes_completed < num_episodes:
                remaining = num_episodes - self.stats.episodes_completed
                if self.verbose:
                    print(f"\n[Training] {remaining} episodes remaining")

                # Run episode
                result = self.run_episode()

                if result is None:
                    self.stats.episodes_failed += 1
                    continue

                # Update stats
                self.stats.episodes_completed += 1
                if result.victory:
                    self.stats.wins += 1
                else:
                    self.stats.losses += 1

                self.stats.total_reward += result.reward
                self.stats.total_steps += result.steps

                # Log episode
                win_rate = self.stats.wins / max(self.stats.wins + self.stats.losses, 1)
                status = "WIN" if result.victory else "LOSS"

                print(f"Episode {self.stats.episodes_completed}: {status}")
                print(f"  Time: {result.game_time:.1f}m | "
                      f"Army: {result.final_army_strength:.2f}x | "
                      f"Reward: {result.reward:.2f}")
                print(f"  Overall: {self.stats.wins}W-{self.stats.losses}L "
                      f"({win_rate:.1%} win rate)")

                self.episode_logs.append({
                    'episode': self.stats.episodes_completed,
                    'victory': result.victory,
                    'game_time': result.game_time,
                    'army_strength': result.final_army_strength,
                    'reward': result.reward,
                    'steps': result.steps,
                    'timestamp': datetime.now().isoformat(),
                })

                # Checkpoint
                if self.stats.episodes_completed % checkpoint_interval == 0:
                    self._save_checkpoint(self.stats.episodes_completed)

                # Track best
                if win_rate > self.stats.best_win_rate and self.stats.episodes_completed >= 5:
                    self.stats.best_win_rate = win_rate
                    self._save_checkpoint(self.stats.episodes_completed, best=True)
                    print(f"  New best win rate!")

        except KeyboardInterrupt:
            print("\n\n[Training] Interrupted by user")

        finally:
            self.cleanup()
            self._save_checkpoint(self.stats.episodes_completed, final=True)
            self._save_logs()

        # Summary
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Episodes: {self.stats.episodes_completed}/{num_episodes}")
        win_rate = self.stats.wins / max(self.stats.wins + self.stats.losses, 1)
        print(f"  Wins: {self.stats.wins} | Losses: {self.stats.losses}")
        print(f"  Final Win Rate: {win_rate:.1%}")
        print(f"  Best Win Rate: {self.stats.best_win_rate:.1%}")
        print(f"{'='*60}\n")

    def _save_checkpoint(self, episode: int, best: bool = False, final: bool = False):
        """Save checkpoint."""
        if best:
            path = os.path.join(self.checkpoint_dir, 'best_agent.pt')
        elif final:
            path = os.path.join(self.checkpoint_dir, 'final_agent.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'agent_ep{episode}.pt')

        self.agent.save(path)

        # Save training state
        state_path = path.replace('.pt', '_state.json')
        with open(state_path, 'w') as f:
            json.dump({
                'episode': episode,
                'stats': asdict(self.stats),
            }, f, indent=2)

        print(f"[Checkpoint] Saved: {path}")

    def _load_checkpoint(self, path: str) -> int:
        """Load checkpoint, return next episode number."""
        self.agent.load(path)

        state_path = path.replace('.pt', '_state.json')
        if os.path.exists(state_path):
            with open(state_path) as f:
                data = json.load(f)
                stats_data = data.get('stats', {})
                self.stats.wins = stats_data.get('wins', 0)
                self.stats.losses = stats_data.get('losses', 0)
                self.stats.episodes_completed = stats_data.get('episodes_completed', 0)
                self.stats.best_win_rate = stats_data.get('best_win_rate', 0.0)
                return data.get('episode', 0)
        return 0

    def _save_logs(self):
        """Save training logs."""
        if not self.episode_logs:
            return

        log_path = os.path.join(
            self.log_dir,
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(log_path, 'w') as f:
            for log in self.episode_logs:
                f.write(json.dumps(log) + '\n')
        print(f"[Logs] Saved: {log_path}")
