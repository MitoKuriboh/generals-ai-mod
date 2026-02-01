"""
Tactical State Builder

Converts game state into 64-dimensional tactical state vector for team-level decisions.

State Layout (64 dimensions):
- Strategic embedding (8): From strategic layer output
- Team composition (12): Infantry/vehicles/aircraft/mixed counts and health
- Team status (8): Health, ammo, cohesion, experience, distances, flags
- Situational (16): Nearby enemies/allies, terrain, threat, target value
- Current objective (8): Type, position, priority, progress, time
- Temporal (4): Time since engagement, command, spawn
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
import torch


@dataclass
class TacticalState:
    """Structured tactical state for a team."""

    # From strategic layer (8 dims)
    strategy_embedding: np.ndarray  # [8]

    # Team composition (12 dims)
    team_infantry: np.ndarray   # [count, health, ready]
    team_vehicles: np.ndarray   # [count, health, ready]
    team_aircraft: np.ndarray   # [count, health, ready]
    team_mixed: np.ndarray      # [count, health, ready]

    # Team status (8 dims)
    team_health: float          # Average health ratio
    ammunition: float           # Average ammo ratio
    cohesion: float             # How spread out (0=scattered, 1=tight)
    experience: float           # Veterancy level
    dist_to_objective: float    # Normalized distance
    dist_to_base: float         # Normalized distance
    under_fire: float           # 1 if taking damage
    has_transport: float        # 1 if has transport units

    # Situational (16 dims)
    nearby_enemies: np.ndarray  # [4 quadrants: N, E, S, W]
    nearby_allies: np.ndarray   # [4 quadrants: N, E, S, W]
    terrain_advantage: float    # -1 to 1 (low ground to high ground)
    threat_level: float         # 0-1 immediate danger
    target_value: float         # Value of current target
    supply_dist: float          # Distance to supplies
    retreat_path: float         # 0-1 viability of retreat
    reinforce_possible: float   # 1 if reinforcements available
    special_ready: float        # 1 if special ability ready
    padding_1: float

    # Current objective (8 dims)
    objective_type: float       # Encoded type
    objective_x: float          # Normalized position
    objective_y: float          # Normalized position
    priority: float             # 0-1 priority level
    progress: float             # 0-1 objective progress
    time_on_objective: float    # Normalized time
    reserved_1: float
    reserved_2: float

    # Temporal (4 dims)
    time_since_engagement: float  # Normalized time
    time_since_command: float     # Normalized time
    frames_since_spawn: float     # Normalized time
    reserved_3: float

    # Additional padding for 64-dim alignment (8 dims)
    padding_2: float = 0.0
    padding_3: float = 0.0
    padding_4: float = 0.0
    padding_5: float = 0.0
    padding_6: float = 0.0
    padding_7: float = 0.0
    padding_8: float = 0.0
    padding_9: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert to 64-dim tensor."""
        features = []

        # Strategic embedding (8)
        features.extend(self.strategy_embedding)

        # Team composition (12)
        features.extend(self.team_infantry)
        features.extend(self.team_vehicles)
        features.extend(self.team_aircraft)
        features.extend(self.team_mixed)

        # Team status (8)
        features.extend([
            self.team_health, self.ammunition, self.cohesion, self.experience,
            self.dist_to_objective, self.dist_to_base, self.under_fire, self.has_transport
        ])

        # Situational (16)
        features.extend(self.nearby_enemies)
        features.extend(self.nearby_allies)
        features.extend([
            self.terrain_advantage, self.threat_level, self.target_value,
            self.supply_dist, self.retreat_path, self.reinforce_possible,
            self.special_ready, self.padding_1
        ])

        # Current objective (8)
        features.extend([
            self.objective_type, self.objective_x, self.objective_y,
            self.priority, self.progress, self.time_on_objective,
            self.reserved_1, self.reserved_2
        ])

        # Temporal (4)
        features.extend([
            self.time_since_engagement, self.time_since_command,
            self.frames_since_spawn, self.reserved_3
        ])

        # Additional padding (8)
        features.extend([
            self.padding_2, self.padding_3, self.padding_4, self.padding_5,
            self.padding_6, self.padding_7, self.padding_8, self.padding_9
        ])

        return torch.tensor(features, dtype=torch.float32)

    @classmethod
    def zeros(cls) -> 'TacticalState':
        """Create a zero-initialized state."""
        return cls(
            strategy_embedding=np.zeros(8),
            team_infantry=np.zeros(3),
            team_vehicles=np.zeros(3),
            team_aircraft=np.zeros(3),
            team_mixed=np.zeros(3),
            team_health=0.0,
            ammunition=1.0,
            cohesion=1.0,
            experience=0.0,
            dist_to_objective=1.0,
            dist_to_base=0.0,
            under_fire=0.0,
            has_transport=0.0,
            nearby_enemies=np.zeros(4),
            nearby_allies=np.zeros(4),
            terrain_advantage=0.0,
            threat_level=0.0,
            target_value=0.0,
            supply_dist=1.0,
            retreat_path=1.0,
            reinforce_possible=0.0,
            special_ready=0.0,
            padding_1=0.0,
            objective_type=0.0,
            objective_x=0.5,
            objective_y=0.5,
            priority=0.5,
            progress=0.0,
            time_on_objective=0.0,
            reserved_1=0.0,
            reserved_2=0.0,
            time_since_engagement=1.0,
            time_since_command=0.0,
            frames_since_spawn=0.0,
            reserved_3=0.0,
            padding_2=0.0,
            padding_3=0.0,
            padding_4=0.0,
            padding_5=0.0,
            padding_6=0.0,
            padding_7=0.0,
            padding_8=0.0,
            padding_9=0.0,
        )


def build_tactical_state(game_data: Dict, team_id: int,
                         strategic_output: Optional[np.ndarray] = None) -> TacticalState:
    """
    Build tactical state from game data.

    Args:
        game_data: Raw game state from C++
        team_id: ID of the team
        strategic_output: Output from strategic layer (8 floats)

    Returns:
        TacticalState ready for neural network
    """
    team_data = game_data.get('teams', {}).get(str(team_id), {})

    # Strategic embedding (8)
    if strategic_output is not None:
        strategy_embedding = np.array(strategic_output[:8])
    else:
        # Default balanced strategy
        strategy_embedding = np.array([0.25, 0.1, 0.4, 0.25, 0.33, 0.33, 0.34, 0.5])

    # Team composition (12)
    composition = team_data.get('composition', {})
    team_infantry = np.array([
        _safe_log_count(composition.get('infantry_count', 0)),
        composition.get('infantry_health', 1.0),
        composition.get('infantry_ready', 0.0),
    ])
    team_vehicles = np.array([
        _safe_log_count(composition.get('vehicle_count', 0)),
        composition.get('vehicle_health', 1.0),
        composition.get('vehicle_ready', 0.0),
    ])
    team_aircraft = np.array([
        _safe_log_count(composition.get('aircraft_count', 0)),
        composition.get('aircraft_health', 1.0),
        composition.get('aircraft_ready', 0.0),
    ])
    team_mixed = np.array([
        _safe_log_count(composition.get('mixed_count', 0)),
        composition.get('mixed_health', 1.0),
        composition.get('mixed_ready', 0.0),
    ])

    # Team status (8)
    status = team_data.get('status', {})
    team_health = np.clip(status.get('health', 1.0), 0.0, 1.0)
    ammunition = np.clip(status.get('ammunition', 1.0), 0.0, 1.0)
    cohesion = np.clip(status.get('cohesion', 1.0), 0.0, 1.0)
    experience = np.clip(status.get('experience', 0.0), 0.0, 1.0)
    dist_to_objective = np.clip(status.get('dist_to_objective', 1.0), 0.0, 1.0)
    dist_to_base = np.clip(status.get('dist_to_base', 0.0), 0.0, 1.0)
    under_fire = 1.0 if status.get('under_fire', False) else 0.0
    has_transport = 1.0 if status.get('has_transport', False) else 0.0

    # Situational (16)
    situational = team_data.get('situational', {})
    nearby_enemies = np.clip(np.array(situational.get('nearby_enemies', [0, 0, 0, 0])), 0.0, 1.0)
    nearby_allies = np.clip(np.array(situational.get('nearby_allies', [0, 0, 0, 0])), 0.0, 1.0)
    terrain_advantage = np.clip(situational.get('terrain_advantage', 0.0), -1.0, 1.0)
    threat_level = np.clip(situational.get('threat_level', 0.0), 0.0, 1.0)
    target_value = np.clip(situational.get('target_value', 0.0), 0.0, 1.0)
    supply_dist = np.clip(situational.get('supply_dist', 1.0), 0.0, 1.0)
    retreat_path = np.clip(situational.get('retreat_path', 1.0), 0.0, 1.0)
    reinforce_possible = 1.0 if situational.get('reinforce_possible', False) else 0.0
    special_ready = 1.0 if situational.get('special_ready', False) else 0.0

    # Current objective (8)
    objective = team_data.get('objective', {})
    objective_type = _encode_objective_type(objective.get('type', 'none'))
    objective_x = np.clip(objective.get('x', 0.5), 0.0, 1.0)
    objective_y = np.clip(objective.get('y', 0.5), 0.0, 1.0)
    priority = np.clip(objective.get('priority', 0.5), 0.0, 1.0)
    progress = np.clip(objective.get('progress', 0.0), 0.0, 1.0)
    time_on_objective = np.clip(objective.get('time', 0.0) / 60.0, 0.0, 1.0)  # Normalize to minutes

    # Temporal (4)
    temporal = team_data.get('temporal', {})
    time_since_engagement = np.clip(temporal.get('since_engagement', 60.0) / 60.0, 0.0, 1.0)
    time_since_command = np.clip(temporal.get('since_command', 0.0) / 30.0, 0.0, 1.0)
    frames_since_spawn = np.clip(temporal.get('since_spawn', 0.0) / 3600.0, 0.0, 1.0)

    return TacticalState(
        strategy_embedding=strategy_embedding,
        team_infantry=team_infantry,
        team_vehicles=team_vehicles,
        team_aircraft=team_aircraft,
        team_mixed=team_mixed,
        team_health=team_health,
        ammunition=ammunition,
        cohesion=cohesion,
        experience=experience,
        dist_to_objective=dist_to_objective,
        dist_to_base=dist_to_base,
        under_fire=under_fire,
        has_transport=has_transport,
        nearby_enemies=nearby_enemies,
        nearby_allies=nearby_allies,
        terrain_advantage=terrain_advantage,
        threat_level=threat_level,
        target_value=target_value,
        supply_dist=supply_dist,
        retreat_path=retreat_path,
        reinforce_possible=reinforce_possible,
        special_ready=special_ready,
        padding_1=0.0,
        objective_type=objective_type,
        objective_x=objective_x,
        objective_y=objective_y,
        priority=priority,
        progress=progress,
        time_on_objective=time_on_objective,
        reserved_1=0.0,
        reserved_2=0.0,
        time_since_engagement=time_since_engagement,
        time_since_command=time_since_command,
        frames_since_spawn=frames_since_spawn,
        reserved_3=0.0,
    )


def _safe_log_count(count: int) -> float:
    """Convert count to log-scaled value."""
    return np.log10(count + 1)


def _encode_objective_type(obj_type: str) -> float:
    """Encode objective type as normalized float."""
    types = {
        'none': 0.0,
        'attack': 0.2,
        'defend': 0.4,
        'scout': 0.6,
        'capture': 0.8,
        'retreat': 1.0,
    }
    return types.get(obj_type.lower(), 0.0)


def state_tensor_from_dict(team_data: Dict, strategic_output: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Quick conversion from raw team data dict to tensor.

    For use in inference when you don't need the full TacticalState object.
    """
    state = build_tactical_state({'teams': {'0': team_data}}, 0, strategic_output)
    return state.to_tensor()


if __name__ == '__main__':
    print("Testing TacticalState...")

    # Test zero state
    state = TacticalState.zeros()
    tensor = state.to_tensor()
    print(f"Zero state tensor shape: {tensor.shape}")
    assert tensor.shape == (64,), f"Expected (64,), got {tensor.shape}"

    # Test building from game data
    game_data = {
        'teams': {
            '1': {
                'composition': {
                    'infantry_count': 10,
                    'infantry_health': 0.8,
                    'vehicle_count': 5,
                    'vehicle_health': 0.9,
                },
                'status': {
                    'health': 0.85,
                    'under_fire': True,
                    'cohesion': 0.7,
                },
                'situational': {
                    'nearby_enemies': [0.2, 0.0, 0.5, 0.1],
                    'threat_level': 0.6,
                },
                'objective': {
                    'type': 'attack',
                    'x': 0.7,
                    'y': 0.3,
                    'priority': 0.8,
                },
                'temporal': {
                    'since_engagement': 10.0,
                },
            }
        }
    }

    strategic = np.array([0.2, 0.1, 0.5, 0.2, 0.3, 0.4, 0.3, 0.8])
    state = build_tactical_state(game_data, 1, strategic)
    tensor = state.to_tensor()

    print(f"\nBuilt state tensor shape: {tensor.shape}")
    print(f"Strategic embedding: {tensor[:8].numpy()}")
    print(f"Under fire: {state.under_fire}")
    print(f"Threat level: {state.threat_level}")

    print("\nTacticalState test passed!")
