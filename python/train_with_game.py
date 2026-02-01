#!/usr/bin/env python3
"""
Unified Training Launcher for C&C Generals AI

Combines game launching with PPO training in a single script.
This is the main entry point for training the AI against the real game.

Usage:
    # Visual training (see what's happening)
    python train_with_game.py --episodes 100

    # Headless training (faster, no graphics)
    python train_with_game.py --episodes 500 --headless

    # Resume from checkpoint
    python train_with_game.py --episodes 500 --resume checkpoints/best_agent.pt

    # Different map/difficulty
    python train_with_game.py --map "Tournament Desert" --ai 2  # vs Hard AI

Requirements:
    - Game built with auto-skirmish support (see STATE.md)
    - pywin32: pip install pywin32
    - PyTorch: pip install torch
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.ppo import PPOAgent, PPOConfig
from training.model import state_dict_to_tensor, action_tensor_to_dict, STATE_DIM
from training.rewards import calculate_step_reward, RewardConfig
from training.config import WIN_REWARD, LOSS_REWARD, PROTOCOL_VERSION
from game_launcher import GameLauncher, Episode


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
        print(f"[Warning] Protocol mismatch: expected {PROTOCOL_VERSION}, got {version}")
        return False
    return True


@dataclass
class TrainingStats:
    """Training statistics."""
    episodes_completed: int = 0
    episodes_failed: int = 0
    wins: int = 0
    losses: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    best_win_rate: float = 0.0


class UnifiedTrainer:
    """
    Unified trainer that handles both game launching and PPO training.

    This is the recommended way to train the AI against the real game.
    """

    def __init__(
        self,
        game_path: str = None,
        map_name: str = "Alpine Assault",
        ai_difficulty: int = 0,  # Start with Easy
        headless: bool = False,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        learning_rate: float = 3e-4,
        seed: int = None,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Trainer] Device: {self.device}")

        # Game launcher setup
        self.launcher = GameLauncher(
            game_path=game_path,
            ai_difficulty=ai_difficulty,
            map_name=map_name,
            headless=headless,
            seed=seed,
        )

        # PPO agent
        ppo_config = PPOConfig(lr=learning_rate)
        self.agent = PPOAgent(ppo_config, device=self.device)

        # Reward configuration
        self.reward_config = RewardConfig()

        # Paths
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Stats
        self.stats = TrainingStats()
        self.episode_logs: List[Dict] = []

    def policy_fn(self, state: Dict) -> Dict:
        """
        Policy function that uses PPO agent.

        Takes raw game state dict, returns recommendation dict.
        """
        # Convert state dict to tensor
        state_tensor = state_dict_to_tensor(state)

        # Get action from PPO agent
        action, log_prob, value = self.agent.select_action(state_tensor)

        # Store for later PPO update
        self._current_log_prob = log_prob
        self._current_value = value
        self._current_action = action

        # Convert action tensor to recommendation dict
        return action_tensor_to_dict(action)

    def train(
        self,
        num_episodes: int,
        checkpoint_interval: int = 50,
        resume_path: str = None,
    ):
        """
        Main training loop.

        Args:
            num_episodes: Number of episodes to train
            checkpoint_interval: Save checkpoint every N episodes
            resume_path: Path to checkpoint to resume from
        """
        start_episode = 1

        # Resume from checkpoint
        if resume_path:
            start_episode = self._load_checkpoint(resume_path)
            print(f"[Trainer] Resumed from episode {start_episode}")

        print(f"\n{'='*60}")
        print(f"  UNIFIED TRAINING LAUNCHER")
        print(f"  Episodes: {num_episodes}")
        print(f"  Map: {self.launcher.map_name}")
        print(f"  AI: {['Easy', 'Medium', 'Hard', 'Learning'][self.launcher.ai_difficulty]}")
        print(f"  Headless: {self.launcher.headless}")
        print(f"{'='*60}\n")

        for ep_num in range(start_episode, num_episodes + 1):
            print(f"\n=== Episode {ep_num}/{num_episodes} ===")

            # Run episode with PPO policy
            episode = self._run_training_episode()

            if episode is None:
                self.stats.episodes_failed += 1
                print(f"Episode {ep_num}: FAILED (game crashed or timed out)")
                time.sleep(5)  # Wait before retry
                continue

            self.stats.episodes_completed += 1

            # Track win/loss
            if episode.victory:
                self.stats.wins += 1
                result = "WIN"
            else:
                self.stats.losses += 1
                result = "LOSS"

            episode_reward = sum(episode.rewards)
            self.stats.total_reward += episode_reward
            self.stats.total_steps += len(episode.states)

            # Log
            win_rate = self.stats.wins / max(self.stats.wins + self.stats.losses, 1)
            print(f"Result: {result} | "
                  f"Time: {episode.game_time:.1f}m | "
                  f"Army: {episode.final_army_strength:.2f}x | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Win Rate: {win_rate:.1%}")

            self.episode_logs.append({
                'episode': ep_num,
                'victory': episode.victory,
                'game_time': episode.game_time,
                'army_strength': episode.final_army_strength,
                'reward': episode_reward,
                'steps': len(episode.states),
                'timestamp': datetime.now().isoformat(),
            })

            # Checkpoint
            if ep_num % checkpoint_interval == 0:
                self._save_checkpoint(ep_num)

            # Track best win rate
            if win_rate > self.stats.best_win_rate and ep_num >= 10:
                self.stats.best_win_rate = win_rate
                self._save_checkpoint(ep_num, best=True)
                print(f"  New best win rate: {win_rate:.1%}")

            # Brief pause between episodes
            time.sleep(2)

        # Final save
        self._save_checkpoint(num_episodes, final=True)
        self._save_logs()

        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Episodes: {self.stats.episodes_completed}/{num_episodes}")
        print(f"  Wins: {self.stats.wins} | Losses: {self.stats.losses}")
        print(f"  Final Win Rate: {self.stats.wins / max(self.stats.wins + self.stats.losses, 1):.1%}")
        print(f"  Best Win Rate: {self.stats.best_win_rate:.1%}")
        print(f"{'='*60}\n")

    def _run_training_episode(self) -> Optional[Episode]:
        """
        Run a single training episode with PPO updates.
        """
        # Start game
        if not self.launcher.start_game():
            return None

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []

        prev_state = None
        game_ended = False
        victory = False
        game_time = 0.0
        final_army = 0.0

        try:
            while self.launcher.is_running() and not game_ended:
                # Get state from game
                state = self.launcher.get_state()

                if state is None:
                    time.sleep(0.1)
                    continue

                # Validate protocol version on first state
                if len(states) == 0:
                    validate_protocol_version(state)

                # Check for game end
                if state.get('type') == 'game_end':
                    game_ended = True
                    victory = state.get('victory', False)
                    game_time = state.get('game_time', 0.0)
                    final_army = state.get('army_strength', 0.0)
                    break

                states.append(state)

                # Calculate reward
                reward = self._calculate_reward(prev_state, state)
                rewards.append(reward)
                prev_state = state

                # Get action from PPO policy
                recommendation = self.policy_fn(state)
                actions.append(recommendation)
                log_probs.append(self._current_log_prob)
                values.append(self._current_value)

                # Store transition in PPO buffer
                state_tensor = state_dict_to_tensor(state)
                self.agent.store_transition(
                    state_tensor,
                    self._current_action,
                    reward,
                    self._current_value,
                    self._current_log_prob,
                    done=False
                )

                # Send to game (wrapped with capabilities declaration)
                wrapped = wrap_recommendation_with_capabilities(recommendation)
                self.launcher.send_recommendation(wrapped)

                # PPO update every 256 steps
                if len(self.agent.buffer) >= 256:
                    with torch.no_grad():
                        _, _, last_value = self.agent.select_action(state_tensor)
                    self.agent.update(last_value)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.launcher.stop_game()

        if not game_ended:
            return None

        # Terminal reward from config (±100.0)
        terminal_reward = WIN_REWARD if victory else LOSS_REWARD

        # CRITICAL FIX: Store terminal transition in PPO buffer
        # Previously terminal reward was only added to local list, never to PPO buffer
        if states and hasattr(self, '_current_action') and self._current_action is not None:
            last_state_tensor = state_dict_to_tensor(states[-1])
            self.agent.store_transition(
                last_state_tensor,
                self._current_action,
                terminal_reward,
                torch.tensor(0.0),  # Terminal state value estimate is 0
                self._current_log_prob,
                done=True
            )

        # Update local rewards list for stats (kept for compatibility)
        if rewards:
            rewards[-1] += terminal_reward
        else:
            rewards.append(terminal_reward)

        # Final PPO update with terminal state
        if len(self.agent.buffer) > 0:
            self.agent.update(torch.tensor(0.0))  # Terminal value = 0

        return Episode(
            states=states,
            actions=actions,
            rewards=rewards,
            victory=victory,
            game_time=game_time,
            final_army_strength=final_army,
        )

    def _calculate_reward(self, prev_state: Optional[Dict], state: Dict) -> float:
        """Calculate step reward using unified rewards module."""
        return calculate_step_reward(prev_state, state, self.reward_config)

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
                'config': {
                    'map': self.launcher.map_name,
                    'ai_difficulty': self.launcher.ai_difficulty,
                    'headless': self.launcher.headless,
                },
            }, f, indent=2)

        print(f"  Checkpoint saved: {path}")

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
                self.stats.best_win_rate = stats_data.get('best_win_rate', 0.0)
                return data.get('episode', 0) + 1
        return 1

    def _save_logs(self):
        """Save training logs."""
        log_path = os.path.join(
            self.log_dir,
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(log_path, 'w') as f:
            for log in self.episode_logs:
                f.write(json.dumps(log) + '\n')
        print(f"Logs saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified C&C Generals AI Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 100 episodes against Easy AI (visual)
  python train_with_game.py --episodes 100

  # Train 500 episodes headless (faster)
  python train_with_game.py --episodes 500 --headless

  # Train against Hard AI
  python train_with_game.py --episodes 200 --ai 2

  # Resume from checkpoint
  python train_with_game.py --episodes 500 --resume checkpoints/best_agent.pt
        """
    )

    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train (default: 100)')
    parser.add_argument('--game-path', type=str, default=None,
                        help='Path to generals.exe')
    parser.add_argument('--map', type=str, default='Alpine Assault',
                        help='Map name (default: Alpine Assault)')
    parser.add_argument('--ai', type=int, default=0, choices=[0, 1, 2],
                        help='Enemy AI difficulty: 0=Easy, 1=Medium, 2=Hard (default: 0)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without graphics (faster training)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Save checkpoint every N episodes (default: 50)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    trainer = UnifiedTrainer(
        game_path=args.game_path,
        map_name=args.map,
        ai_difficulty=args.ai,
        headless=args.headless,
        learning_rate=args.lr,
        seed=args.seed,
    )

    try:
        trainer.train(
            num_episodes=args.episodes,
            checkpoint_interval=args.checkpoint_interval,
            resume_path=args.resume,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")


if __name__ == '__main__':
    main()
