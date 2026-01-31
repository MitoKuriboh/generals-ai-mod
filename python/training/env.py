"""
Environment Wrapper for Generals Zero Hour Learning AI

Provides a gym-like interface for the game, handling:
- Named pipe communication with the game
- State preprocessing
- Action postprocessing
- Episode management
"""

import struct
import json
import time
import sys
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch

from .model import state_dict_to_tensor, action_tensor_to_dict, STATE_DIM, TOTAL_ACTION_DIM

# Windows named pipe support
if sys.platform == 'win32':
    import win32pipe
    import win32file
    import pywintypes
    HAS_WIN32 = True
else:
    HAS_WIN32 = False


PIPE_NAME = r'\\.\pipe\generals_ml_bridge'
MAX_EPISODE_STEPS = 3000  # ~100 minutes of game time, prevents infinite episodes


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    total_reward: float = 0.0
    steps: int = 0
    units_killed: int = 0
    units_lost: int = 0
    buildings_destroyed: int = 0
    buildings_lost: int = 0
    final_army_strength: float = 0.0
    final_money: float = 0.0
    game_time: float = 0.0
    won: Optional[bool] = None


class GeneralsEnv:
    """
    Environment wrapper for Generals Zero Hour.

    Provides gym-like interface: reset(), step(), close()
    """

    def __init__(self, pipe_name: str = PIPE_NAME, timeout_ms: int = 100):
        self.pipe_name = pipe_name
        self.timeout_ms = timeout_ms
        self.pipe = None
        self.connected = False

        # State tracking
        self.current_state: Optional[Dict] = None
        self.prev_state: Optional[Dict] = None
        self.episode_stats = EpisodeStats()
        self.step_count = 0

        # For reward shaping
        self.prev_own_units = 0
        self.prev_enemy_units = 0
        self.prev_own_buildings = 0
        self.prev_enemy_buildings = 0
        self.prev_money = 0

    def create_pipe(self) -> bool:
        """Create the named pipe server."""
        if not HAS_WIN32:
            print("[Env] Named pipes require Windows")
            return False

        try:
            self.pipe = win32pipe.CreateNamedPipe(
                self.pipe_name,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1,  # Max instances
                4096,  # Out buffer
                4096,  # In buffer
                self.timeout_ms,
                None
            )
            return True
        except Exception as e:
            print(f"[Env] Failed to create pipe: {e}")
            return False

    def wait_for_connection(self, timeout: float = 60.0) -> bool:
        """Wait for game to connect."""
        if not HAS_WIN32:
            return False

        print(f"[Env] Waiting for game connection (timeout: {timeout}s)...")
        try:
            win32pipe.ConnectNamedPipe(self.pipe, None)
            self.connected = True
            print("[Env] Game connected!")
            return True
        except Exception as e:
            print(f"[Env] Connection failed: {e}")
            return False

    def _read_message(self) -> Optional[str]:
        """Read a length-prefixed message from pipe."""
        if not HAS_WIN32 or not self.connected:
            return None

        try:
            # Read 4-byte length prefix
            result, length_bytes = win32file.ReadFile(self.pipe, 4)
            if len(length_bytes) < 4:
                return None

            length = struct.unpack('<I', length_bytes)[0]

            # Read message data
            result, data = win32file.ReadFile(self.pipe, length)
            return data.decode('utf-8')
        except Exception as e:
            if hasattr(e, 'args') and e.args[0] == 109:  # ERROR_BROKEN_PIPE
                self.connected = False
            return None

    def _write_message(self, data: str) -> bool:
        """Write a length-prefixed message to pipe."""
        if not HAS_WIN32 or not self.connected:
            return False

        try:
            encoded = data.encode('utf-8')
            length = struct.pack('<I', len(encoded))
            win32file.WriteFile(self.pipe, length + encoded)
            return True
        except Exception as e:
            self.connected = False
            return False

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """
        Reset for a new episode.

        Note: This doesn't restart the game - that must be done externally.
        This just resets episode tracking and waits for first state.

        Returns:
            state: Initial state tensor
            info: Episode info dict
        """
        self.current_state = None
        self.prev_state = None
        self.episode_stats = EpisodeStats()
        self.step_count = 0

        self.prev_own_units = 0
        self.prev_enemy_units = 0
        self.prev_own_buildings = 0
        self.prev_enemy_buildings = 0
        self.prev_money = 0

        # Wait for first state from game
        state = self._wait_for_state()
        if state is None:
            raise RuntimeError("Failed to receive initial state from game")

        self.current_state = state
        self._update_tracking(state)

        return state_dict_to_tensor(state), {'raw_state': state}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """
        Execute action and return next state.

        Args:
            action: Action tensor from policy

        Returns:
            next_state: Next state tensor
            reward: Reward for this transition
            terminated: Whether episode ended (win/loss)
            truncated: Whether episode was cut short
            info: Additional info
        """
        # Convert action to recommendation dict
        recommendation = action_tensor_to_dict(action)

        # Send recommendation to game
        rec_json = json.dumps(recommendation)
        if not self._write_message(rec_json):
            # Connection lost
            return self._handle_disconnect()

        # Wait for next state
        state = self._wait_for_state()
        if state is None:
            return self._handle_disconnect()

        self.prev_state = self.current_state
        self.current_state = state
        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward(self.prev_state, state, recommendation)
        self.episode_stats.total_reward += reward
        self.episode_stats.steps = self.step_count

        # Check for episode end
        terminated, won = self._check_terminated(state)
        truncated = self.step_count >= MAX_EPISODE_STEPS

        if truncated and not terminated:
            # Truncated but not terminated - determine winner by army strength
            won = state.get('army_strength', 1.0) > 1.0
            self.episode_stats.won = won

        if terminated or truncated:
            self.episode_stats.won = won
            self.episode_stats.game_time = state.get('game_time', 0)
            self.episode_stats.final_army_strength = state.get('army_strength', 0)
            self.episode_stats.final_money = state.get('money', 0)

        # Update tracking
        self._update_tracking(state)

        done = terminated or truncated
        info = {
            'raw_state': state,
            'recommendation': recommendation,
            'episode_stats': self.episode_stats if done else None,
            'truncated': truncated,
        }

        return state_dict_to_tensor(state), reward, terminated, truncated, info

    def _wait_for_state(self, timeout: float = 5.0) -> Optional[Dict]:
        """Wait for next state message from game."""
        start = time.time()
        while time.time() - start < timeout:
            msg = self._read_message()
            if msg:
                try:
                    return json.loads(msg)
                except json.JSONDecodeError:
                    continue
            time.sleep(0.01)
        return None

    def _calculate_reward(self, prev_state: Optional[Dict], state: Dict,
                          action: Dict) -> float:
        """Calculate reward for state transition."""
        from .rewards import calculate_reward
        return calculate_reward(prev_state, state, action, self)

    def _check_terminated(self, state: Dict) -> Tuple[bool, Optional[bool]]:
        """
        Check if episode has terminated.

        Returns:
            (terminated, won): Whether episode ended and if we won
        """
        # Check if we lost (no structures)
        own_structures = state.get('own_structures', [0, 0, 0])
        if own_structures[0] < 0.3:  # Less than 1 structure (log scale)
            return True, False

        # Check if enemy lost (no visible structures for a while)
        enemy_structures = state.get('enemy_structures', [0, 0, 0])
        if enemy_structures[0] < 0.3 and state.get('game_time', 0) > 5.0:
            # No enemy structures visible and game has been running
            # This is a heuristic - actual win detection would need game events
            return True, True

        # Game timeout (30 minutes)
        if state.get('game_time', 0) > 30.0:
            # Determine winner by army strength
            army_strength = state.get('army_strength', 1.0)
            return True, army_strength > 1.0

        return False, None

    def _update_tracking(self, state: Dict):
        """Update tracking variables for reward calculation."""
        # Extract unit counts from log-scale
        self.prev_own_units = self._count_from_state(state, 'own')
        self.prev_enemy_units = self._count_from_state(state, 'enemy')

        own_struct = state.get('own_structures', [0, 0, 0])
        enemy_struct = state.get('enemy_structures', [0, 0, 0])
        self.prev_own_buildings = int(10 ** own_struct[0] - 1) if own_struct[0] > 0 else 0
        self.prev_enemy_buildings = int(10 ** enemy_struct[0] - 1) if enemy_struct[0] > 0 else 0

        self.prev_money = state.get('money', 0)

    def _count_from_state(self, state: Dict, prefix: str) -> int:
        """Count total units from state."""
        total = 0
        for category in ['infantry', 'vehicles', 'aircraft']:
            key = f'{prefix}_{category}'
            arr = state.get(key, [0, 0, 0])
            if arr[0] > 0:
                total += int(10 ** arr[0] - 1)
        return total

    def _handle_disconnect(self) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """Handle pipe disconnection."""
        self.connected = False
        # Return terminal state with loss
        dummy_state = torch.zeros(STATE_DIM)
        return dummy_state, -1.0, True, True, {'disconnected': True}

    def close(self):
        """Close the environment."""
        if self.pipe and HAS_WIN32:
            try:
                win32file.CloseHandle(self.pipe)
            except:
                pass
            self.pipe = None
        self.connected = False

    def render(self):
        """Render current state (for debugging)."""
        if self.current_state is None:
            print("[Env] No state to render")
            return

        state = self.current_state
        print(f"\n{'='*50}")
        print(f"Step: {self.step_count} | Time: {state.get('game_time', 0):.1f}m")
        print(f"Money: ${10**state.get('money', 0):.0f} | Power: {state.get('power', 0):.0f}")
        print(f"Army Strength: {state.get('army_strength', 1):.2f}x | Under Attack: {state.get('under_attack', 0) > 0.5}")
        print(f"Episode Reward: {self.episode_stats.total_reward:.2f}")
        print(f"{'='*50}")


class SimulatedEnv:
    """
    Simulated environment for testing without the actual game.

    Generates synthetic states that roughly mimic game progression.
    """

    def __init__(self, episode_length: int = 100):
        self.episode_length = episode_length
        self.step_count = 0
        self.current_state = None
        self.total_reward = 0.0

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset to initial state."""
        self.step_count = 0
        self.total_reward = 0.0
        self.current_state = self._generate_initial_state()
        return state_dict_to_tensor(self.current_state), {'raw_state': self.current_state}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.step_count += 1

        # Update state based on action
        recommendation = action_tensor_to_dict(action)
        self.current_state = self._update_state(self.current_state, recommendation)

        # Simple reward: higher army strength is better
        reward = (self.current_state['army_strength'] - 1.0) * 0.1
        self.total_reward += reward

        # Episode terminates after episode_length steps
        terminated = self.step_count >= self.episode_length
        truncated = False

        info = {'raw_state': self.current_state}

        # Add episode stats when terminated
        if terminated:
            won = self.current_state['army_strength'] > 1.2
            info['episode_stats'] = EpisodeStats(
                total_reward=self.total_reward,
                steps=self.step_count,
                won=won,
                game_time=self.current_state.get('game_time', 0),
                units_killed=0,
                units_lost=0,
                buildings_destroyed=0,
                buildings_lost=0,
                final_army_strength=self.current_state.get('army_strength', 1.0),
                final_money=self.current_state.get('money', 0),
            )

        return state_dict_to_tensor(self.current_state), reward, terminated, truncated, info

    def _generate_initial_state(self) -> Dict:
        """Generate initial game state."""
        return {
            'player': 1,
            'money': 3.0,  # $1000
            'power': 10.0,
            'income': 2.0,
            'supply': 0.9,
            'own_infantry': [0.5, 0.9, 0.0],
            'own_vehicles': [0.3, 0.9, 0.0],
            'own_aircraft': [0.0, 0.0, 0.0],
            'own_structures': [0.7, 0.95, 0.0],
            'enemy_infantry': [0.3, 0.8, 0.0],
            'enemy_vehicles': [0.3, 0.8, 0.0],
            'enemy_aircraft': [0.0, 0.0, 0.0],
            'enemy_structures': [0.6, 0.9, 0.0],
            'game_time': 0.0,
            'tech_level': 0.2,
            'base_threat': 0.0,
            'army_strength': 1.0,
            'under_attack': 0.0,
            'distance_to_enemy': 0.5,
        }

    def _update_state(self, state: Dict, action: Dict) -> Dict:
        """Update state based on action (simplified simulation)."""
        new_state = state.copy()

        # Advance game time
        new_state['game_time'] = state['game_time'] + 0.5  # 30 seconds per step

        # Economy grows if prioritized
        if action['priority_economy'] > 0.3:
            new_state['money'] = min(4.0, state['money'] + 0.05)
            new_state['income'] = min(5.0, state['income'] + 0.1)

        # Military grows if prioritized
        if action['priority_military'] > 0.3:
            growth = 0.05
            if action['prefer_infantry'] > 0.4:
                new_state['own_infantry'] = [min(1.5, state['own_infantry'][0] + growth), 0.9, 0.0]
            if action['prefer_vehicles'] > 0.4:
                new_state['own_vehicles'] = [min(1.5, state['own_vehicles'][0] + growth), 0.9, 0.0]
            if action['prefer_aircraft'] > 0.4:
                new_state['own_aircraft'] = [min(1.5, state['own_aircraft'][0] + growth), 0.9, 0.0]

        # Tech advances if prioritized
        if action['priority_tech'] > 0.3:
            new_state['tech_level'] = min(1.0, state['tech_level'] + 0.02)

        # Combat simulation
        if action['aggression'] > 0.5:
            # Attacking: may kill enemy units, may lose own
            new_state['enemy_infantry'][0] = max(0, state['enemy_infantry'][0] - 0.03)
            new_state['own_infantry'][0] = max(0, state['own_infantry'][0] - 0.01)

        # Recalculate army strength
        own_power = (10 ** new_state['own_infantry'][0] +
                     10 ** new_state['own_vehicles'][0] * 2 +
                     10 ** new_state['own_aircraft'][0] * 3)
        enemy_power = (10 ** new_state['enemy_infantry'][0] +
                       10 ** new_state['enemy_vehicles'][0] * 2 +
                       10 ** new_state['enemy_aircraft'][0] * 3)
        new_state['army_strength'] = own_power / max(enemy_power, 1.0)

        return new_state

    def close(self):
        """Close environment."""
        pass

    def render(self):
        """Render current state."""
        if self.current_state:
            print(f"Step {self.step_count}: army_strength={self.current_state['army_strength']:.2f}")


if __name__ == '__main__':
    # Test with simulated environment
    print("Testing SimulatedEnv...")

    env = SimulatedEnv(episode_length=20)
    state, info = env.reset()

    print(f"Initial state shape: {state.shape}")

    total_reward = 0
    for i in range(20):
        # Random action
        action = torch.rand(TOTAL_ACTION_DIM)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode terminated at step {i+1}")
            break

    print(f"Total reward: {total_reward:.2f}")
    print("SimulatedEnv test passed!")
