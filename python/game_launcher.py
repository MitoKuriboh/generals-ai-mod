"""
Game Launcher for ML Training

Automates launching C&C Generals Zero Hour for training runs.
Handles game process management, named pipe communication, and episode collection.

Author: Mito, 2025
"""

import subprocess
import time
import os
import sys
import json
import struct
import signal
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Windows-specific imports for named pipes
if sys.platform == 'win32':
    import win32pipe
    import win32file
    import pywintypes


class GameState(Enum):
    """Game process states."""
    NOT_STARTED = "not_started"
    STARTING = "starting"
    RUNNING = "running"
    ENDED = "ended"
    CRASHED = "crashed"


@dataclass
class Episode:
    """Training episode data."""
    states: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    victory: bool
    game_time: float
    final_army_strength: float


class NamedPipeServer:
    """Named pipe server for communication with game."""

    PIPE_NAME = r'\\.\pipe\generals_ml_bridge'
    BUFFER_SIZE = 4096

    def __init__(self):
        self.pipe = None
        self.connected = False

    def create(self) -> bool:
        """Create the named pipe server."""
        if sys.platform != 'win32':
            print("Named pipes only supported on Windows")
            return False

        try:
            self.pipe = win32pipe.CreateNamedPipe(
                self.PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1,  # Max instances
                self.BUFFER_SIZE,
                self.BUFFER_SIZE,
                0,  # Default timeout
                None  # Security attributes
            )
            print(f"Created named pipe: {self.PIPE_NAME}")
            return True
        except pywintypes.error as e:
            print(f"Failed to create pipe: {e}")
            return False

    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """Wait for game to connect to the pipe."""
        if not self.pipe:
            return False

        try:
            print("Waiting for game to connect...")
            # Use overlapped I/O for timeout support
            overlapped = pywintypes.OVERLAPPED()
            overlapped.hEvent = win32file.CreateEvent(None, True, False, None)

            try:
                win32pipe.ConnectNamedPipe(self.pipe, overlapped)
            except pywintypes.error as e:
                if e.args[0] != 997:  # ERROR_IO_PENDING
                    raise

            # Wait with timeout
            result = win32file.WaitForSingleObject(overlapped.hEvent, int(timeout * 1000))

            if result == 0:  # WAIT_OBJECT_0
                self.connected = True
                print("Game connected!")
                return True
            else:
                print(f"Connection timeout (waited {timeout}s)")
                return False

        except pywintypes.error as e:
            print(f"Connection error: {e}")
            return False

    def read_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Read a message from the pipe (non-blocking with timeout)."""
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

            if msg_length > self.BUFFER_SIZE:
                print(f"Message too large: {msg_length}")
                return None

            # Read message data
            result, msg_data = win32file.ReadFile(self.pipe, msg_length)

            # Parse JSON
            return json.loads(msg_data.decode('utf-8'))

        except pywintypes.error as e:
            if e.args[0] in (109, 232):  # Broken pipe, pipe being closed
                self.connected = False
                print("Pipe disconnected")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None

    def write_message(self, data: Dict[str, Any]) -> bool:
        """Write a message to the pipe."""
        if not self.connected:
            return False

        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            # Write length prefix
            length_bytes = struct.pack('<I', len(json_bytes))
            win32file.WriteFile(self.pipe, length_bytes)

            # Write data
            win32file.WriteFile(self.pipe, json_bytes)

            return True

        except pywintypes.error as e:
            print(f"Write error: {e}")
            self.connected = False
            return False

    def close(self):
        """Close the pipe."""
        if self.pipe:
            try:
                win32pipe.DisconnectNamedPipe(self.pipe)
                win32file.CloseHandle(self.pipe)
            except:
                pass
            self.pipe = None
            self.connected = False


class GameLauncher:
    """Manages game process and training episodes."""

    # Default game paths (adjust for your system)
    DEFAULT_GAME_PATH = r"C:\Games\Command and Conquer Generals Zero Hour\generals.exe"
    DEFAULT_MAP = "Alpine Assault"

    def __init__(
        self,
        game_path: str = None,
        ai_difficulty: int = 3,  # 0=Easy, 1=Medium, 2=Hard, 3=Learning
        map_name: str = None,
        headless: bool = True,
        seed: int = None
    ):
        self.game_path = game_path or self.DEFAULT_GAME_PATH
        self.ai_difficulty = ai_difficulty
        self.map_name = map_name or self.DEFAULT_MAP
        self.headless = headless
        self.seed = seed

        self.process: Optional[subprocess.Popen] = None
        self.pipe = NamedPipeServer()
        self.state = GameState.NOT_STARTED

        self.current_episode: Optional[Episode] = None
        self.episode_states: List[Dict[str, Any]] = []
        self.episode_actions: List[Dict[str, Any]] = []

    def build_command_line(self) -> List[str]:
        """Build command line arguments for the game."""
        args = [self.game_path]

        # Auto-skirmish mode
        args.append("-autoSkirmish")

        # AI difficulty
        args.extend(["-aiDifficulty", str(self.ai_difficulty)])

        # Map
        if self.map_name:
            args.extend(["-skirmishMap", self.map_name])

        # Headless options
        if self.headless:
            args.append("-noDraw")
            args.append("-noAudio")
            args.append("-noFPSLimit")

        # Quick start (skip intro movies)
        args.append("-quickstart")

        # Random seed for reproducibility
        if self.seed is not None:
            args.extend(["-seed", str(self.seed)])

        return args

    def start_game(self) -> bool:
        """Start the game process."""
        if self.state == GameState.RUNNING:
            print("Game already running")
            return False

        # Create pipe first
        if not self.pipe.create():
            return False

        # Build and start game
        cmd = self.build_command_line()
        print(f"Starting game: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            self.state = GameState.STARTING
            print(f"Game started (PID: {self.process.pid})")

        except Exception as e:
            print(f"Failed to start game: {e}")
            self.pipe.close()
            return False

        # Wait for game to connect
        if not self.pipe.wait_for_connection(timeout=60.0):
            print("Game failed to connect to pipe")
            self.stop_game()
            return False

        self.state = GameState.RUNNING
        self.episode_states = []
        self.episode_actions = []

        return True

    def stop_game(self):
        """Stop the game process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        self.pipe.close()
        self.state = GameState.NOT_STARTED

    def is_running(self) -> bool:
        """Check if game is still running."""
        if not self.process:
            return False
        return self.process.poll() is None

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get latest game state from pipe."""
        return self.pipe.read_message()

    def send_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Send ML recommendation to game."""
        return self.pipe.write_message(recommendation)

    def run_episode(self, policy_fn=None) -> Optional[Episode]:
        """
        Run a single training episode.

        Args:
            policy_fn: Function that takes game state and returns recommendation.
                      If None, uses default balanced recommendations.

        Returns:
            Episode data if game completed, None if error.
        """
        if not self.start_game():
            return None

        states = []
        actions = []
        rewards = []

        prev_state = None
        game_ended = False
        victory = False
        game_time = 0.0
        final_army = 0.0

        try:
            while self.is_running() and not game_ended:
                # Get state from game
                state = self.get_state()

                if state is None:
                    time.sleep(0.1)
                    continue

                # Check for game end message
                if state.get('type') == 'game_end':
                    game_ended = True
                    victory = state.get('victory', False)
                    game_time = state.get('game_time', 0.0)
                    final_army = state.get('army_strength', 0.0)
                    print(f"Game ended: {'Victory' if victory else 'Defeat'} at {game_time:.1f} min")
                    break

                # Store state
                states.append(state)

                # Calculate reward from state change
                reward = self._calculate_reward(prev_state, state)
                rewards.append(reward)
                prev_state = state

                # Get recommendation from policy
                if policy_fn:
                    action = policy_fn(state)
                else:
                    action = self._default_policy(state)

                actions.append(action)

                # Send to game
                self.send_recommendation(action)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.stop_game()

        if not game_ended:
            # Game crashed or was interrupted
            return None

        # Add terminal reward
        terminal_reward = 1.0 if victory else -1.0
        if rewards:
            rewards[-1] += terminal_reward
        else:
            rewards.append(terminal_reward)

        return Episode(
            states=states,
            actions=actions,
            rewards=rewards,
            victory=victory,
            game_time=game_time,
            final_army_strength=final_army
        )

    def _calculate_reward(self, prev_state: Optional[Dict], state: Dict) -> float:
        """Calculate step reward from state change."""
        if prev_state is None:
            return 0.0

        reward = 0.0

        # Reward for army strength increase
        prev_army = prev_state.get('army_strength', 1.0)
        curr_army = state.get('army_strength', 1.0)
        reward += (curr_army - prev_army) * 0.1

        # Penalty for being under attack
        if state.get('under_attack', 0) > 0.5:
            reward -= 0.01

        # Reward for tech advancement
        prev_tech = prev_state.get('tech_level', 0)
        curr_tech = state.get('tech_level', 0)
        reward += (curr_tech - prev_tech) * 0.5

        return reward

    def _default_policy(self, state: Dict) -> Dict[str, Any]:
        """Default balanced policy when no ML model provided."""
        return {
            'priority_economy': 0.25,
            'priority_defense': 0.25,
            'priority_military': 0.30,
            'priority_tech': 0.20,
            'prefer_infantry': 0.33,
            'prefer_vehicles': 0.40,
            'prefer_aircraft': 0.27,
            'aggression': 0.5,
            'target_player': -1
        }

    def run_training_loop(
        self,
        num_episodes: int,
        policy_fn=None,
        save_dir: str = None
    ) -> List[Episode]:
        """
        Run multiple training episodes.

        Args:
            num_episodes: Number of episodes to run.
            policy_fn: Policy function for generating actions.
            save_dir: Directory to save episode data.

        Returns:
            List of completed episodes.
        """
        episodes = []
        wins = 0

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i in range(num_episodes):
            print(f"\n=== Episode {i+1}/{num_episodes} ===")

            episode = self.run_episode(policy_fn)

            if episode:
                episodes.append(episode)
                if episode.victory:
                    wins += 1

                print(f"Episode {i+1}: {'Win' if episode.victory else 'Loss'}, "
                      f"time={episode.game_time:.1f}min, "
                      f"army={episode.final_army_strength:.2f}")

                # Save episode data
                if save_dir:
                    ep_path = os.path.join(save_dir, f"episode_{i+1:04d}.json")
                    self._save_episode(episode, ep_path)
            else:
                print(f"Episode {i+1}: Failed/Crashed")

            # Short delay between episodes
            time.sleep(2.0)

        print(f"\n=== Training Complete ===")
        print(f"Episodes: {len(episodes)}/{num_episodes}")
        print(f"Win rate: {wins}/{len(episodes)} ({100*wins/len(episodes):.1f}%)" if episodes else "No episodes completed")

        return episodes

    def _save_episode(self, episode: Episode, path: str):
        """Save episode data to JSON file."""
        data = {
            'victory': episode.victory,
            'game_time': episode.game_time,
            'final_army_strength': episode.final_army_strength,
            'num_steps': len(episode.states),
            'total_reward': sum(episode.rewards),
            'states': episode.states,
            'actions': episode.actions,
            'rewards': episode.rewards
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='C&C Generals ML Training Launcher')

    parser.add_argument('--game-path', type=str, default=None,
                       help='Path to generals.exe')
    parser.add_argument('--map', type=str, default='Alpine Assault',
                       help='Map name for skirmish')
    parser.add_argument('--ai', type=int, default=3, choices=[0, 1, 2, 3],
                       help='AI difficulty (0=Easy, 1=Medium, 2=Hard, 3=Learning)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run')
    parser.add_argument('--headless', action='store_true',
                       help='Run without graphics (faster)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='./episodes',
                       help='Directory to save episode data')

    args = parser.parse_args()

    launcher = GameLauncher(
        game_path=args.game_path,
        ai_difficulty=args.ai,
        map_name=args.map,
        headless=args.headless,
        seed=args.seed
    )

    try:
        episodes = launcher.run_training_loop(
            num_episodes=args.episodes,
            save_dir=args.save_dir
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        launcher.stop_game()


if __name__ == '__main__':
    main()
