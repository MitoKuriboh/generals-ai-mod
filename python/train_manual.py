#!/usr/bin/env python3
"""
Manual Training Script for C&C Generals AI

This script enables training when the game doesn't support auto-skirmish.
User manually starts skirmish games; this script handles the ML training.

Workflow:
1. Run this script (creates named pipe, waits for game)
2. Launch game normally, start skirmish with Learning AI opponent
3. Script trains during gameplay
4. When game ends, start another skirmish (or exit)
5. Script handles multiple episodes across manual game starts

Can also be auto-launched by the game when Learning AI is selected.

Usage:
    # Basic training
    python train_manual.py --episodes 10

    # Resume from checkpoint
    python train_manual.py --episodes 50 --resume checkpoints/best_agent.pt

    # Verbose mode
    python train_manual.py --episodes 20 --verbose
"""

import os
import sys
import argparse
import json
import struct
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging (works even when launched without console)
def setup_logging(log_dir: str = "logs"):
    """Configure logging to file and optionally console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # File handler (always)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Console handler only if we have a console
    try:
        # Check if we have a valid stdout
        if sys.stdout is not None and hasattr(sys.stdout, 'write'):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    except:
        pass  # No console available (pythonw.exe)

    return log_file

from training.ppo import PPOAgent, PPOConfig
from training.model import state_dict_to_tensor, action_tensor_to_dict, STATE_DIM

# Windows named pipe support
if sys.platform == 'win32':
    import win32pipe
    import win32file
    import pywintypes
    HAS_WIN32 = True
else:
    HAS_WIN32 = False


PIPE_NAME = r'\\.\pipe\generals_ml_bridge'


@dataclass
class TrainingStats:
    """Training statistics across all episodes."""
    episodes_completed: int = 0
    wins: int = 0
    losses: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    best_win_rate: float = 0.0


class ManualTrainer:
    """
    Training handler for manual game starts.

    Creates a named pipe server and waits for the game to connect.
    User starts games manually; training happens during gameplay.
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

        # Named pipe
        self.pipe = None
        self.connected = False

        # Training state
        self.stats = TrainingStats()
        self.episode_logs: List[Dict] = []
        self.current_episode_states: List[Dict] = []
        self.current_episode_actions: List[Dict] = []
        self.current_episode_rewards: List[float] = []

        # PPO state for current step
        self._current_log_prob = None
        self._current_value = None
        self._current_action = None
        self._prev_state = None

    def _create_pipe(self) -> bool:
        """Create the named pipe server."""
        if not HAS_WIN32:
            print("[Error] Named pipes require Windows")
            return False

        try:
            self.pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1,  # Max instances
                4096,  # Out buffer
                4096,  # In buffer
                0,  # Default timeout
                None  # Security attributes
            )
            print(f"[Pipe] Created: {PIPE_NAME}")
            return True
        except pywintypes.error as e:
            print(f"[Error] Failed to create pipe: {e}")
            return False

    def _wait_for_connection(self, timeout: float = None) -> bool:
        """Wait for game to connect."""
        if not self.pipe:
            return False

        print("[Pipe] Waiting for game to connect...")
        print("       Start a skirmish with Learning AI opponent")

        try:
            overlapped = pywintypes.OVERLAPPED()
            overlapped.hEvent = win32file.CreateEvent(None, True, False, None)

            try:
                win32pipe.ConnectNamedPipe(self.pipe, overlapped)
            except pywintypes.error as e:
                if e.args[0] != 997:  # ERROR_IO_PENDING
                    raise

            # Wait (indefinitely if timeout is None)
            wait_time = int(timeout * 1000) if timeout else 0xFFFFFFFF  # INFINITE
            result = win32file.WaitForSingleObject(overlapped.hEvent, wait_time)

            if result == 0:  # WAIT_OBJECT_0
                self.connected = True
                print("[Pipe] Game connected!")
                return True
            else:
                print("[Pipe] Connection timeout")
                return False

        except pywintypes.error as e:
            print(f"[Error] Connection failed: {e}")
            return False

    def _disconnect_pipe(self):
        """Disconnect and prepare for next connection."""
        if self.pipe:
            try:
                win32pipe.DisconnectNamedPipe(self.pipe)
            except:
                pass
        self.connected = False

    def _close_pipe(self):
        """Close the pipe completely."""
        if self.pipe:
            try:
                win32pipe.DisconnectNamedPipe(self.pipe)
                win32file.CloseHandle(self.pipe)
            except:
                pass
            self.pipe = None
        self.connected = False

    def _read_message(self) -> Optional[Dict]:
        """Read a message from the pipe (non-blocking check, blocking read)."""
        if not self.connected:
            return None

        try:
            # Check if data available
            _, bytes_avail, _ = win32pipe.PeekNamedPipe(self.pipe, 0)

            if bytes_avail < 4:
                return None

            # Read length prefix
            result, length_data = win32file.ReadFile(self.pipe, 4)
            if len(length_data) < 4:
                return None

            msg_length = struct.unpack('<I', length_data)[0]

            if msg_length > 65536:
                print(f"[Error] Message too large: {msg_length}")
                return None

            # Read message data
            result, msg_data = win32file.ReadFile(self.pipe, msg_length)

            return json.loads(msg_data.decode('utf-8'))

        except pywintypes.error as e:
            if e.args[0] in (109, 232):  # Broken pipe, pipe being closed
                self.connected = False
                print("[Pipe] Disconnected")
            return None
        except json.JSONDecodeError as e:
            print(f"[Error] JSON parse error: {e}")
            return None

    def _write_message(self, data: Dict) -> bool:
        """Write a message to the pipe."""
        if not self.connected:
            return False

        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            # Write length prefix + data
            length_bytes = struct.pack('<I', len(json_bytes))
            win32file.WriteFile(self.pipe, length_bytes)
            win32file.WriteFile(self.pipe, json_bytes)

            return True
        except pywintypes.error as e:
            print(f"[Error] Write failed: {e}")
            self.connected = False
            return False

    def _get_recommendation(self, state: Dict) -> Dict:
        """Get recommendation from PPO agent."""
        state_tensor = state_dict_to_tensor(state)

        action, log_prob, value = self.agent.select_action(state_tensor)

        self._current_log_prob = log_prob
        self._current_value = value
        self._current_action = action

        return action_tensor_to_dict(action)

    def _calculate_reward(self, prev_state: Optional[Dict], state: Dict) -> float:
        """Calculate step reward."""
        if prev_state is None:
            return 0.0

        reward = 0.0

        # Army strength increase
        prev_army = prev_state.get('army_strength', 1.0)
        curr_army = state.get('army_strength', 1.0)
        reward += (curr_army - prev_army) * 0.5

        # Tech level increase
        prev_tech = prev_state.get('tech_level', 0)
        curr_tech = state.get('tech_level', 0)
        reward += (curr_tech - prev_tech) * 1.0

        # Penalty for being attacked
        if state.get('under_attack', 0) > 0.5:
            reward -= 0.02

        # Small reward for staying alive
        reward += 0.01

        return reward

    def _run_single_episode(self) -> Optional[Dict]:
        """
        Run a single training episode.

        Returns episode results or None if disconnected/failed.
        """
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self._prev_state = None

        victory = False
        game_time = 0.0
        final_army = 0.0
        game_ended = False

        print(f"\n--- Episode {self.stats.episodes_completed + 1} Started ---")

        while self.connected and not game_ended:
            # Read state from game
            state = self._read_message()

            if state is None:
                time.sleep(0.05)
                continue

            # Check for game end message
            if state.get('type') == 'game_end':
                game_ended = True
                victory = state.get('victory', False)
                game_time = state.get('game_time', 0.0)
                final_army = state.get('army_strength', 0.0)
                break

            # Store state
            self.current_episode_states.append(state)

            # Calculate reward
            reward = self._calculate_reward(self._prev_state, state)
            self.current_episode_rewards.append(reward)

            # Get recommendation
            recommendation = self._get_recommendation(state)
            self.current_episode_actions.append(recommendation)

            if self.verbose and len(self.current_episode_states) % 50 == 0:
                print(f"  Step {len(self.current_episode_states)}: "
                      f"army={state.get('army_strength', 1.0):.2f}x, "
                      f"time={state.get('game_time', 0):.1f}m")

            # Store transition for PPO
            state_tensor = state_dict_to_tensor(state)
            self.agent.store_transition(
                state_tensor,
                self._current_action,
                reward,
                self._current_value,
                self._current_log_prob,
                done=False
            )

            # Send recommendation to game
            self._write_message(recommendation)

            # PPO update every 256 steps
            if len(self.agent.buffer) >= 256:
                with torch.no_grad():
                    _, _, last_value = self.agent.select_action(state_tensor)
                loss = self.agent.update(last_value)
                if self.verbose:
                    print(f"  PPO update: loss={loss:.4f}")

            self._prev_state = state

        if not game_ended:
            # Disconnected mid-game
            return None

        # Terminal reward
        terminal_reward = 10.0 if victory else -10.0
        if self.current_episode_rewards:
            self.current_episode_rewards[-1] += terminal_reward

        # Final PPO update
        if len(self.agent.buffer) > 0:
            self.agent.update(torch.tensor(0.0))

        # Update stats
        self.stats.episodes_completed += 1
        if victory:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        episode_reward = sum(self.current_episode_rewards)
        self.stats.total_reward += episode_reward
        self.stats.total_steps += len(self.current_episode_states)

        result = {
            'episode': self.stats.episodes_completed,
            'victory': victory,
            'game_time': game_time,
            'final_army_strength': final_army,
            'steps': len(self.current_episode_states),
            'reward': episode_reward,
            'timestamp': datetime.now().isoformat(),
        }

        self.episode_logs.append(result)

        return result

    def train(
        self,
        num_episodes: int,
        checkpoint_interval: int = 10,
        resume_path: str = None,
    ):
        """
        Main training loop for manual game starts.

        Args:
            num_episodes: Target number of episodes
            checkpoint_interval: Save every N episodes
            resume_path: Optional checkpoint to resume from
        """
        start_episode = 0

        # Resume from checkpoint
        if resume_path:
            start_episode = self._load_checkpoint(resume_path)
            print(f"[Training] Resumed from episode {start_episode}")

        print(f"\n{'='*60}")
        print(f"  MANUAL TRAINING MODE")
        print(f"  Device: {self.device}")
        print(f"  Target Episodes: {num_episodes}")
        print(f"  Checkpoint Interval: {checkpoint_interval}")
        print(f"{'='*60}")
        print(f"\nInstructions:")
        print(f"  1. Launch C&C Generals Zero Hour")
        print(f"  2. Start Skirmish → Select Learning AI opponent")
        print(f"  3. Play the game (training happens automatically)")
        print(f"  4. When game ends, start another skirmish")
        print(f"  5. Repeat until {num_episodes} episodes complete")
        print(f"  6. Press Ctrl+C to stop early\n")

        # Create pipe
        if not self._create_pipe():
            return

        try:
            while self.stats.episodes_completed < num_episodes:
                remaining = num_episodes - self.stats.episodes_completed
                print(f"\n[Training] {remaining} episodes remaining")

                # Wait for game connection
                if not self._wait_for_connection():
                    break

                # Run episode
                result = self._run_single_episode()

                if result:
                    win_rate = self.stats.wins / max(self.stats.wins + self.stats.losses, 1)
                    status = "WIN" if result['victory'] else "LOSS"

                    print(f"\nEpisode {result['episode']}: {status}")
                    print(f"  Time: {result['game_time']:.1f}m | "
                          f"Army: {result['final_army_strength']:.2f}x | "
                          f"Reward: {result['reward']:.2f}")
                    print(f"  Overall: {self.stats.wins}W-{self.stats.losses}L "
                          f"({win_rate:.1%} win rate)")

                    # Checkpoint
                    if result['episode'] % checkpoint_interval == 0:
                        self._save_checkpoint(result['episode'])

                    # Track best
                    if win_rate > self.stats.best_win_rate and result['episode'] >= 5:
                        self.stats.best_win_rate = win_rate
                        self._save_checkpoint(result['episode'], best=True)
                        print(f"  New best win rate!")

                # Disconnect and prepare for next game
                self._disconnect_pipe()

                if self.stats.episodes_completed < num_episodes:
                    print("\n[Training] Ready for next game. Start a new skirmish...")

        except KeyboardInterrupt:
            print("\n\n[Training] Interrupted by user")

        finally:
            self._close_pipe()
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
            f"manual_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(log_path, 'w') as f:
            for log in self.episode_logs:
                f.write(json.dumps(log) + '\n')
        print(f"[Logs] Saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Manual Training for C&C Generals AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script handles training when the game is started manually.
Can also be auto-launched by the game when Learning AI is selected.

Workflow:
  1. Run this script (or let the game launch it)
  2. Start skirmish with Learning AI
  3. Training happens during gameplay
  4. When game ends, start another skirmish
  5. Repeat until target episodes reached

Examples:
  python train_manual.py --episodes 10
  python train_manual.py --episodes 50 --resume checkpoints/best_agent.pt
  python train_manual.py --episodes 20 --verbose
        """
    )

    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to train (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N episodes (default: 10)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output during training')

    args = parser.parse_args()

    # Set up logging (works even when launched by game without console)
    log_file = setup_logging()
    logging.info(f"Trainer starting, log file: {log_file}")

    if not HAS_WIN32:
        logging.error("This script requires Windows (for named pipes)")
        sys.exit(1)

    try:
        trainer = ManualTrainer(
            learning_rate=args.lr,
            verbose=args.verbose,
        )

        trainer.train(
            num_episodes=args.episodes,
            checkpoint_interval=args.checkpoint_interval,
            resume_path=args.resume,
        )
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
