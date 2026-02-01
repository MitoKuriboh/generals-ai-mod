"""
Hierarchical Coordinator

Orchestrates inference across three layers with proper latency budgeting:
- Strategic: ~0.1ms (once per second)
- Tactical: ~0.5ms x teams (every 5 seconds per team)
- Micro: ~0.02ms x units (every 0.5 seconds per unit in combat)

Total budget: <10ms per frame for all ML inference
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import math
import logging

logger = logging.getLogger(__name__)

# Import layer modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model import PolicyNetwork, state_dict_to_tensor, action_tensor_to_dict
from tactical.model import TacticalNetwork, TacticalAction, action_dict_to_command as tactical_to_command
from tactical.state import build_tactical_state, TacticalState
from micro.model import MicroNetwork, MicroAction, action_dict_to_command as micro_to_command
from micro.state import build_micro_state, MicroState


@dataclass
class InferenceConfig:
    """Configuration for hierarchical inference."""

    # Decision intervals (in game frames, 30 fps)
    strategic_interval: int = 30      # 1 second
    tactical_interval: int = 150      # 5 seconds per team
    micro_interval: int = 15          # 0.5 seconds per unit

    # Maximum entities per frame
    max_teams_per_frame: int = 10
    max_units_per_frame: int = 50

    # Enable/disable layers
    tactical_enabled: bool = True
    micro_enabled: bool = True

    # Device
    device: str = 'cpu'


class HierarchicalCoordinator:
    """
    Coordinates inference across strategic, tactical, and micro layers.

    Manages latency budgets and batches requests for efficiency.
    """

    def __init__(self,
                 strategic_model: Optional[PolicyNetwork] = None,
                 tactical_model: Optional[TacticalNetwork] = None,
                 micro_model: Optional[MicroNetwork] = None,
                 config: Optional[InferenceConfig] = None):

        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)

        # Initialize models
        self.strategic = strategic_model or PolicyNetwork()
        self.strategic = self.strategic.to(self.device).eval()

        if self.config.tactical_enabled:
            self.tactical = tactical_model or TacticalNetwork()
            self.tactical = self.tactical.to(self.device).eval()
        else:
            self.tactical = None

        if self.config.micro_enabled:
            self.micro = micro_model or MicroNetwork()
            self.micro = self.micro.to(self.device).eval()
            # Track per-unit hidden states for LSTM
            self.micro_hidden_states: Dict[int, Tuple] = {}
        else:
            self.micro = None

        # Caching for strategic output
        self.last_strategic_output: Optional[np.ndarray] = None
        self.last_strategic_frame: int = 0

        # Per-team tracking for tactical
        self.last_tactical_frame: Dict[int, int] = {}

        # Per-unit tracking for micro
        self.last_micro_frame: Dict[int, int] = {}

        # Latency tracking
        self.latency_stats = {
            'strategic': [],
            'tactical': [],
            'micro': [],
        }

    def process_frame(self, game_state: Dict, current_frame: int) -> Dict:
        """
        Process a game frame and return all recommendations.

        Args:
            game_state: Full game state from C++
            current_frame: Current game frame number

        Returns:
            Dict with strategic, tactical, and micro recommendations
        """
        result = {
            'strategic': None,
            'teams': [],
            'units': [],
        }

        # Strategic layer (runs every strategic_interval frames)
        if current_frame - self.last_strategic_frame >= self.config.strategic_interval:
            try:
                t0 = time.perf_counter()
                result['strategic'] = self._process_strategic(game_state)
                self.latency_stats['strategic'].append(time.perf_counter() - t0)
                self.last_strategic_frame = current_frame
                self.last_strategic_output = np.array([
                    result['strategic'][k] for k in [
                        'priority_economy', 'priority_defense', 'priority_military',
                        'priority_tech', 'prefer_infantry', 'prefer_vehicles',
                        'prefer_aircraft', 'aggression'
                    ]
                ])
            except Exception as e:
                logger.error(f"Strategic inference failed: {e}")
                result['strategic'] = self._default_strategic()

        # Tactical layer (runs per team when due)
        if self.config.tactical_enabled and self.tactical is not None:
            try:
                t0 = time.perf_counter()
                teams_to_process = self._get_teams_due(game_state, current_frame)
                if teams_to_process:
                    result['teams'] = self._process_tactical_batch(
                        game_state, teams_to_process, current_frame
                    )
                if result['teams']:
                    self.latency_stats['tactical'].append(time.perf_counter() - t0)
            except Exception as e:
                logger.error(f"Tactical inference failed: {e}")
                result['teams'] = []

        # Micro layer (runs per unit in combat when due)
        # FIX: Check self.micro is not None before processing
        if self.config.micro_enabled and self.micro is not None:
            try:
                t0 = time.perf_counter()
                units_to_process = self._get_units_due(game_state, current_frame)
                if units_to_process:
                    result['units'] = self._process_micro_batch(
                        game_state, units_to_process, current_frame
                    )
                if result['units']:
                    self.latency_stats['micro'].append(time.perf_counter() - t0)
            except Exception as e:
                logger.error(f"Micro inference failed: {e}")
                result['units'] = []

        return result

    def _default_strategic(self) -> Dict:
        """Return default strategic recommendation when inference fails."""
        return {
            'priority_economy': 0.25,
            'priority_defense': 0.15,
            'priority_military': 0.45,
            'priority_tech': 0.15,
            'prefer_infantry': 0.33,
            'prefer_vehicles': 0.34,
            'prefer_aircraft': 0.33,
            'aggression': 0.5,
            'target_player': -1,
        }

    def _process_strategic(self, game_state: Dict) -> Dict:
        """Process strategic layer inference."""
        state_tensor = state_dict_to_tensor(game_state).to(self.device)

        with torch.no_grad():
            action, _, _ = self.strategic.get_action(state_tensor, deterministic=True)

        return action_tensor_to_dict(action)

    def _get_teams_due(self, game_state: Dict, current_frame: int) -> List[int]:
        """Get list of team IDs due for tactical processing."""
        teams = game_state.get('teams', {})
        due_teams = []

        for team_id_str, team_data in teams.items():
            # FIX: Safe int conversion to handle malformed data
            try:
                team_id = int(team_id_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid team ID: {team_id_str}")
                continue

            last_frame = self.last_tactical_frame.get(team_id, 0)

            if current_frame - last_frame >= self.config.tactical_interval:
                due_teams.append(team_id)

                if len(due_teams) >= self.config.max_teams_per_frame:
                    break

        return due_teams

    def _process_tactical_batch(self, game_state: Dict, team_ids: List[int],
                                current_frame: int) -> List[Dict]:
        """Process tactical inference for a batch of teams."""
        results = []

        # Build states for all teams
        states = []
        for team_id in team_ids:
            team_data = game_state.get('teams', {}).get(str(team_id))

            # Handle raw array from C++ (list of 64 floats) vs structured dict
            if isinstance(team_data, list):
                # Raw 64-float array from C++ - use directly as tensor
                state_tensor = torch.tensor(team_data, dtype=torch.float32)
            elif isinstance(team_data, dict):
                # Structured dict (from simulation) - parse via build_tactical_state
                team_state = build_tactical_state(
                    game_state, team_id, self.last_strategic_output
                )
                state_tensor = team_state.to_tensor()
            else:
                # Missing data - use zeros
                logger.warning(f"No team data for team {team_id}")
                state_tensor = torch.zeros(64)

            states.append(state_tensor)
            self.last_tactical_frame[team_id] = current_frame

        if not states:
            return results

        # Batch inference
        batch_states = torch.stack(states).to(self.device)

        with torch.no_grad():
            # Process each state (tactical network doesn't batch well due to varied hidden states)
            for i, team_id in enumerate(team_ids):
                action_dict, _, _ = self.tactical.get_action(
                    batch_states[i], deterministic=True
                )
                cmd = tactical_to_command(action_dict)
                cmd['team_id'] = team_id
                results.append(cmd)

        return results

    def _get_units_due(self, game_state: Dict, current_frame: int) -> List[int]:
        """Get list of unit IDs due for micro processing."""
        units = game_state.get('units', {})
        due_units = []

        for unit_id_str, unit_data in units.items():
            # FIX: Safe int conversion to handle malformed data
            try:
                unit_id = int(unit_id_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid unit ID: {unit_id_str}")
                continue

            # Only process units in combat
            if not self._should_micro_unit(unit_data):
                continue

            last_frame = self.last_micro_frame.get(unit_id, 0)
            if current_frame - last_frame >= self.config.micro_interval:
                due_units.append(unit_id)

                if len(due_units) >= self.config.max_units_per_frame:
                    break

        return due_units

    def _should_micro_unit(self, unit_data) -> bool:
        """Determine if a unit should receive micro control."""
        # Handle raw array from C++ (list of 32 floats)
        if isinstance(unit_data, list):
            # MicroState layout:
            # Index 12: nearestEnemyDist (normalized)
            # Index 18: underFire (1.0 if taking damage)
            # Index 19: abilityReady (1.0 if ready)
            if len(unit_data) >= 32:
                under_fire = unit_data[18] > 0.5
                nearby_enemies = unit_data[12] < 0.5  # Normalized distance
                ability_ready = unit_data[19] > 0.5
                return under_fire or nearby_enemies or ability_ready
            return False

        # Handle structured dict (from simulation)
        if isinstance(unit_data, dict):
            situational = unit_data.get('situational', {})
            in_combat = (
                situational.get('under_fire', False) or
                situational.get('is_attacking', False)
            )
            nearby_enemies = situational.get('enemy_dist', 1000) < 500
            high_value = unit_data.get('cost', 0) > 1000
            has_ability = situational.get('ability_ready', False)
            return in_combat or (nearby_enemies and (high_value or has_ability))

        return False

    def _process_micro_batch(self, game_state: Dict, unit_ids: List[int],
                             current_frame: int) -> List[Dict]:
        """Process micro inference for a batch of units."""
        results = []

        for unit_id in unit_ids:
            unit_data = game_state.get('units', {}).get(str(unit_id))

            # Handle raw array from C++ (list of 32 floats) vs structured dict
            if isinstance(unit_data, list):
                # Raw 32-float array from C++ - use directly as tensor
                state_tensor = torch.tensor(unit_data, dtype=torch.float32).to(self.device)
            elif isinstance(unit_data, dict):
                # Structured dict (from simulation) - parse via build_micro_state
                team_objective = self._get_team_objective_for_unit(game_state, unit_data)
                unit_state = build_micro_state(unit_data, team_objective)
                state_tensor = unit_state.to_tensor().to(self.device)
            else:
                # Missing data - use zeros
                logger.warning(f"No unit data for unit {unit_id}")
                state_tensor = torch.zeros(32).to(self.device)

            # Get or create hidden state for this unit
            if unit_id not in self.micro_hidden_states:
                self.micro.reset_hidden(1, self.device)
            else:
                self.micro.hidden = self.micro_hidden_states[unit_id]

            # Inference
            with torch.no_grad():
                action_dict, _, _ = self.micro.get_action(state_tensor, deterministic=True)

            # Store hidden state
            self.micro_hidden_states[unit_id] = self.micro.hidden

            # Build command
            cmd = micro_to_command(action_dict)
            cmd['unit_id'] = unit_id
            results.append(cmd)

            self.last_micro_frame[unit_id] = current_frame

        return results

    def _get_team_objective_for_unit(self, game_state: Dict, unit_data) -> Dict:
        """Get team objective context for a unit."""
        # Handle raw array from C++ - no team association available
        if isinstance(unit_data, list):
            return {}

        # Handle structured dict
        if isinstance(unit_data, dict):
            team_id = unit_data.get('team_id')
            if team_id is None:
                return {}
            team_data = game_state.get('teams', {}).get(str(team_id), {})
            return team_data.get('objective', {})

        return {}

    def cleanup_stale_units(self, current_frame: int, max_age: int = 300):
        """Remove hidden states for units not seen recently."""
        stale_units = [
            uid for uid, frame in self.last_micro_frame.items()
            if current_frame - frame > max_age
        ]
        for uid in stale_units:
            self.last_micro_frame.pop(uid, None)
            self.micro_hidden_states.pop(uid, None)

        # FIX: Also limit total hidden states to prevent unbounded growth
        max_hidden_states = 256
        if len(self.micro_hidden_states) > max_hidden_states:
            # Remove oldest entries (those with lowest last_micro_frame values)
            sorted_units = sorted(
                self.last_micro_frame.items(),
                key=lambda x: x[1]
            )
            units_to_remove = sorted_units[:len(self.micro_hidden_states) - max_hidden_states]
            for uid, _ in units_to_remove:
                self.micro_hidden_states.pop(uid, None)
                self.last_micro_frame.pop(uid, None)

    def get_latency_stats(self) -> Dict[str, Dict]:
        """Get latency statistics for each layer."""
        stats = {}
        for layer, times in self.latency_stats.items():
            if times:
                times_ms = [t * 1000 for t in times[-100:]]  # Last 100 samples
                stats[layer] = {
                    'mean_ms': np.mean(times_ms),
                    'max_ms': np.max(times_ms),
                    'p99_ms': np.percentile(times_ms, 99),
                    'samples': len(times),
                }
            else:
                stats[layer] = {'mean_ms': 0, 'samples': 0}
        return stats

    def load_models(self,
                    strategic_path: Optional[str] = None,
                    tactical_path: Optional[str] = None,
                    micro_path: Optional[str] = None):
        """Load model weights from checkpoints."""
        if strategic_path:
            self.strategic = PolicyNetwork.load(strategic_path).to(self.device).eval()

        if tactical_path and self.tactical is not None:
            self.tactical = TacticalNetwork.load(tactical_path).to(self.device).eval()

        if micro_path and self.micro is not None:
            self.micro = MicroNetwork.load(micro_path).to(self.device).eval()


if __name__ == '__main__':
    print("Testing HierarchicalCoordinator...")

    config = InferenceConfig(
        tactical_enabled=True,
        micro_enabled=True,
    )

    coordinator = HierarchicalCoordinator(config=config)

    # Simulate game state
    game_state = {
        'money': 3.5,  # log10(3000)
        'power': 50,
        'income': 5,
        'supply': 0.6,
        'own_infantry': [1.0, 0.8, 0],
        'own_vehicles': [0.7, 0.9, 0],
        'own_aircraft': [0.3, 1.0, 0],
        'own_structures': [1.0, 0.95, 0],
        'enemy_infantry': [0.8, 0.7, 0],
        'enemy_vehicles': [0.5, 0.8, 0],
        'enemy_aircraft': [0.2, 1.0, 0],
        'enemy_structures': [0.9, 0.9, 0],
        'game_time': 5.0,
        'tech_level': 0.7,
        'base_threat': 0.3,
        'army_strength': 1.2,
        'under_attack': 0,
        'distance_to_enemy': 0.6,
        'is_usa': 1, 'is_china': 0, 'is_gla': 0,
        'teams': {
            '1': {
                'composition': {'infantry_count': 10, 'vehicle_count': 5},
                'status': {'health': 0.8},
                'objective': {'type': 'attack', 'x': 0.7, 'y': 0.5},
            },
        },
        'units': {
            '101': {
                'type': 'tank',
                'health': 0.75,
                'cost': 1500,
                'situational': {
                    'under_fire': True,
                    'enemy_dist': 200,
                    'enemy_angle': 0.3,
                },
                'team_id': 1,
            },
        },
    }

    # Process several frames
    for frame in range(0, 200, 10):
        result = coordinator.process_frame(game_state, frame)

        if result['strategic']:
            print(f"\nFrame {frame}: Strategic")
            print(f"  Aggression: {result['strategic']['aggression']:.2f}")

        if result['teams']:
            print(f"\nFrame {frame}: Tactical ({len(result['teams'])} teams)")
            for cmd in result['teams']:
                print(f"  Team {cmd['team_id']}: {TacticalAction.name(cmd['action'])}")

        if result['units']:
            print(f"\nFrame {frame}: Micro ({len(result['units'])} units)")
            for cmd in result['units']:
                print(f"  Unit {cmd['unit_id']}: {MicroAction.name(cmd['action'])}")

    # Print latency stats
    print("\nLatency stats:")
    for layer, stats in coordinator.get_latency_stats().items():
        if stats['samples'] > 0:
            print(f"  {layer}: {stats['mean_ms']:.3f}ms avg, {stats['p99_ms']:.3f}ms p99")

    print("\nHierarchicalCoordinator test passed!")
