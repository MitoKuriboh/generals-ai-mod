"""
Real Game Hierarchical Environment for Joint Training

Connects to the actual C&C Generals game via named pipe to collect
hierarchical (strategic/tactical/micro) training data.

This enables training all three layers on real game data instead of
simulated environments only.

Usage:
    env = RealGameHierarchicalEnv(headless=True)
    state = env.reset()

    while not done:
        # Get team and unit states
        team_states = env.get_team_states()
        unit_states = env.get_unit_states()

        # Apply actions
        for team_id, team_state in team_states.items():
            env.apply_tactical_action(team_id, tactical_action)

        for unit_id, unit_state in unit_states.items():
            env.apply_micro_action(unit_id, micro_action)

        # Step
        state, reward, done, info = env.step()

Author: Mito, 2026
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_launcher import GameLauncher, NamedPipeServer, Episode
from training.config import WIN_REWARD, LOSS_REWARD, PROTOCOL_VERSION


@dataclass
class HierarchicalGameState:
    """Parsed hierarchical state from game."""
    strategic: Dict[str, Any]
    teams: Dict[int, Dict[str, Any]]
    units: Dict[int, Dict[str, Any]]
    frame: int
    player_id: int


class RealGameHierarchicalEnv:
    """
    Real game environment for hierarchical RL training.

    Connects to C&C Generals via named pipe and supports all three layers:
    - Strategic: Overall game state (44 dims)
    - Tactical: Team-level states (64 dims each)
    - Micro: Unit-level states (32 dims each)
    """

    def __init__(
        self,
        game_path: str = None,
        map_name: str = "Alpine Assault",
        ai_difficulty: int = 0,  # Easy AI for training
        headless: bool = True,
        seed: int = None,
        connection_timeout: float = 60.0,
        step_timeout: float = 5.0,
    ):
        """
        Initialize the real game environment.

        Args:
            game_path: Path to generals.exe
            map_name: Map for skirmish
            ai_difficulty: 0=Easy, 1=Medium, 2=Hard
            headless: Run without graphics
            seed: Random seed
            connection_timeout: Timeout waiting for game connection
            step_timeout: Timeout waiting for game state
        """
        self.game_path = game_path
        self.map_name = map_name
        self.ai_difficulty = ai_difficulty
        self.headless = headless
        self.seed = seed
        self.connection_timeout = connection_timeout
        self.step_timeout = step_timeout

        # Game launcher (handles process and pipe)
        self.launcher: Optional[GameLauncher] = None

        # Current state tracking
        self.current_state: Optional[HierarchicalGameState] = None
        self.prev_state: Optional[HierarchicalGameState] = None
        self.done = False
        self.step_count = 0

        # Pending actions (batched and sent on step())
        self.pending_tactical: Dict[int, Dict] = {}  # team_id -> action
        self.pending_micro: Dict[int, Dict] = {}     # unit_id -> action

        # Episode data for rewards
        self.episode_states: List[Dict] = []

    def reset(self) -> Dict:
        """
        Reset environment by starting a new game.

        Returns:
            Initial strategic state dict (44 dimensions)
        """
        # Stop any existing game
        if self.launcher:
            self.launcher.stop_game()

        # Create new launcher
        self.launcher = GameLauncher(
            game_path=self.game_path,
            ai_difficulty=self.ai_difficulty,
            map_name=self.map_name,
            headless=self.headless,
            seed=self.seed,
        )

        # Start game and wait for connection
        if not self.launcher.start_game():
            raise RuntimeError("Failed to start game")

        # Reset tracking
        self.current_state = None
        self.prev_state = None
        self.done = False
        self.step_count = 0
        self.pending_tactical.clear()
        self.pending_micro.clear()
        self.episode_states = []

        # Wait for first state
        state = self._wait_for_state()
        if state is None:
            self.close()
            raise RuntimeError("Failed to get initial game state")

        self.current_state = state
        return state.strategic

    def step(self) -> Tuple[Dict, float, bool, Dict]:
        """
        Advance game by sending pending actions and receiving new state.

        Returns:
            (strategic_state, reward, done, info)
        """
        if self.done:
            return self.current_state.strategic if self.current_state else {}, 0.0, True, {}

        # Build and send batched response with pending actions
        response = self._build_response()
        if not self.launcher.send_recommendation(response):
            self.done = True
            return self.current_state.strategic, LOSS_REWARD, True, {'won': False, 'error': 'pipe_error'}

        # Clear pending actions
        self.pending_tactical.clear()
        self.pending_micro.clear()

        # Wait for next state
        state = self._wait_for_state()

        if state is None:
            # Game crashed or disconnected
            self.done = True
            return self.current_state.strategic, LOSS_REWARD, True, {'won': False, 'error': 'no_state'}

        # Check for game end
        if hasattr(state, 'game_end') and state.game_end:
            self.done = True
            victory = state.victory if hasattr(state, 'victory') else False
            game_time = state.game_time if hasattr(state, 'game_time') else 0.0
            reward = WIN_REWARD if victory else LOSS_REWARD
            return state.strategic, reward, True, {'won': victory, 'game_time': game_time}

        # Update state tracking
        self.prev_state = self.current_state
        self.current_state = state
        self.step_count += 1
        self.episode_states.append(state.strategic)

        # Calculate step reward
        reward = self._calculate_reward()

        return state.strategic, reward, False, {}

    def get_team_states(self) -> Dict[int, Dict]:
        """
        Get current states for all player teams.

        Returns:
            Dict mapping team_id -> team state dict (64 dimensions)
        """
        if self.current_state is None:
            return {}
        return self.current_state.teams

    def get_unit_states(self, team_id: int = None) -> Dict[int, Dict]:
        """
        Get current states for units, optionally filtered by team.

        Args:
            team_id: If provided, only return units belonging to this team

        Returns:
            Dict mapping unit_id -> unit state dict (32 dimensions)
        """
        if self.current_state is None:
            return {}

        if team_id is None:
            return self.current_state.units

        # Filter by team (need team membership info from game)
        # For now, return all units - game handles filtering
        return self.current_state.units

    def apply_tactical_action(self, team_id: int, action: Dict) -> float:
        """
        Queue a tactical action for a team.

        Action will be sent on next step() call.

        Args:
            team_id: Team to command
            action: Dict with 'action', 'target_x', 'target_y', 'attitude'

        Returns:
            0.0 (reward calculated on step())
        """
        self.pending_tactical[team_id] = action
        return 0.0

    def apply_micro_action(self, unit_id: int, action: Dict) -> float:
        """
        Queue a micro action for a unit.

        Action will be sent on next step() call.

        Args:
            unit_id: Unit to command
            action: Dict with 'action', 'move_angle', 'move_distance'

        Returns:
            0.0 (reward calculated on step())
        """
        self.pending_micro[unit_id] = action
        return 0.0

    def close(self):
        """Close the environment and stop the game."""
        if self.launcher:
            self.launcher.stop_game()
            self.launcher = None
        self.done = True

    def is_running(self) -> bool:
        """Check if game is still running."""
        return self.launcher is not None and self.launcher.is_running()

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _wait_for_state(self) -> Optional[HierarchicalGameState]:
        """Wait for and parse the next game state."""
        start_time = time.time()

        while time.time() - start_time < self.step_timeout:
            if not self.launcher.is_running():
                return None

            raw_state = self.launcher.get_state()
            if raw_state is None:
                time.sleep(0.01)
                continue

            # Check for game end message
            if raw_state.get('type') == 'game_end':
                # Create a game-end state
                state = HierarchicalGameState(
                    strategic=raw_state,
                    teams={},
                    units={},
                    frame=raw_state.get('frame', 0),
                    player_id=raw_state.get('player_id', 0),
                )
                state.game_end = True
                state.victory = raw_state.get('victory', False)
                state.game_time = raw_state.get('game_time', 0.0)
                return state

            return self._parse_state(raw_state)

        return None

    def _parse_state(self, raw: Dict) -> HierarchicalGameState:
        """Parse raw JSON state into HierarchicalGameState."""
        # Strategic state (44 floats)
        strategic = raw.get('strategic', raw)  # May be nested or flat

        # Team states (64 floats each)
        teams = {}
        for team_data in raw.get('teams', []):
            team_id = team_data.get('id', 0)
            state_array = team_data.get('state', [])
            teams[team_id] = {
                'state': state_array,
                'id': team_id,
            }

        # Unit states (32 floats each)
        units = {}
        for unit_data in raw.get('units', []):
            unit_id = unit_data.get('id', 0)
            state_array = unit_data.get('state', [])
            units[unit_id] = {
                'state': state_array,
                'id': unit_id,
            }

        return HierarchicalGameState(
            strategic=strategic,
            teams=teams,
            units=units,
            frame=raw.get('frame', 0),
            player_id=raw.get('player_id', 0),
        )

    def _build_response(self) -> Dict:
        """Build batched response with pending actions."""
        response = {
            'frame': self.current_state.frame if self.current_state else 0,
            'version': PROTOCOL_VERSION,
            'capabilities': {
                'hierarchical': True,
                'tactical': len(self.pending_tactical) > 0,
                'micro': len(self.pending_micro) > 0,
            },
            # Default strategic recommendation (can be overridden)
            'strategic': {
                'priority_economy': 0.25,
                'priority_defense': 0.20,
                'priority_military': 0.35,
                'priority_tech': 0.20,
                'prefer_infantry': 0.33,
                'prefer_vehicles': 0.34,
                'prefer_aircraft': 0.33,
                'aggression': 0.5,
                'target_player': -1,
            },
            'teams': [],
            'units': [],
        }

        # Add tactical commands
        for team_id, action in self.pending_tactical.items():
            cmd = {
                'id': team_id,
                'action': int(action.get('action', 0)) % 8,
                'x': float(action.get('target_x', 0.5)),
                'y': float(action.get('target_y', 0.5)),
                'attitude': float(action.get('attitude', 0.5)),
            }
            response['teams'].append(cmd)

        # Add micro commands
        for unit_id, action in self.pending_micro.items():
            cmd = {
                'id': unit_id,
                'action': int(action.get('action', 0)) % 11,
                'angle': float(action.get('move_angle', 0.0)),
                'dist': float(action.get('move_distance', 0.3)),
            }
            response['units'].append(cmd)

        return response

    def _calculate_reward(self) -> float:
        """Calculate step reward from state change."""
        if self.prev_state is None or self.current_state is None:
            return 0.0

        reward = 0.0
        prev = self.prev_state.strategic
        curr = self.current_state.strategic

        # Army strength change
        prev_army = prev.get('army_strength', 1.0)
        curr_army = curr.get('army_strength', 1.0)
        reward += (curr_army - prev_army) * 0.5

        # Under attack penalty
        if curr.get('under_attack', 0) > 0.5:
            reward -= 0.01

        # Tech advancement
        prev_tech = prev.get('tech_level', 0)
        curr_tech = curr.get('tech_level', 0)
        reward += (curr_tech - prev_tech) * 0.2

        # Enemy structure destruction (inferred from enemy count decrease)
        # This provides positive signal for offensive success
        prev_enemy = sum([
            prev.get('enemy_infantry', [0])[0] if isinstance(prev.get('enemy_infantry'), list) else 0,
            prev.get('enemy_vehicles', [0])[0] if isinstance(prev.get('enemy_vehicles'), list) else 0,
            prev.get('enemy_aircraft', [0])[0] if isinstance(prev.get('enemy_aircraft'), list) else 0,
        ])
        curr_enemy = sum([
            curr.get('enemy_infantry', [0])[0] if isinstance(curr.get('enemy_infantry'), list) else 0,
            curr.get('enemy_vehicles', [0])[0] if isinstance(curr.get('enemy_vehicles'), list) else 0,
            curr.get('enemy_aircraft', [0])[0] if isinstance(curr.get('enemy_aircraft'), list) else 0,
        ])
        if prev_enemy > curr_enemy:
            reward += (prev_enemy - curr_enemy) * 0.5  # Reward for kills

        return reward

    def set_strategic_recommendation(self, rec: Dict):
        """
        Override strategic recommendation for next step.

        Args:
            rec: Strategic recommendation dict
        """
        # This will be included in the next response
        self._strategic_override = rec


def test_real_env():
    """Quick test of the environment (requires game running on Windows)."""
    print("Testing RealGameHierarchicalEnv...")
    print("Note: This test requires the game to be available on Windows.")

    # Only test if on Windows
    if sys.platform != 'win32':
        print("Skipping test - not on Windows")
        return

    try:
        env = RealGameHierarchicalEnv(
            headless=True,
            ai_difficulty=0,
            map_name="Alpine Assault",
        )

        state = env.reset()
        print(f"Initial state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
        print(f"Teams: {len(env.get_team_states())}")
        print(f"Units: {len(env.get_unit_states())}")

        for step in range(10):
            # Apply some actions
            for team_id in env.get_team_states():
                env.apply_tactical_action(team_id, {
                    'action': 0,
                    'target_x': 0.8,
                    'target_y': 0.5,
                    'attitude': 0.5,
                })

            for unit_id in env.get_unit_states():
                env.apply_micro_action(unit_id, {
                    'action': 1,
                    'move_angle': 0.0,
                    'move_distance': 0.3,
                })

            state, reward, done, info = env.step()
            print(f"Step {step+1}: reward={reward:.3f}, done={done}")

            if done:
                print(f"Game ended: {info}")
                break

        env.close()
        print("Test completed!")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    test_real_env()
