"""
Micro State Builder

Converts game state into 32-dimensional micro state vector for unit-level decisions.

State Layout (32 dimensions):
- Unit identity (4): Type, hero flag, veterancy, has ability
- Unit status (8): Health, shield, ammo, cooldown, speed, range, dps, armor
- Situational (12): Nearest enemy info, nearest ally, cover, under fire, etc.
- Team context (4): Objective type, direction, role, priority
- Temporal (4): Time since hit, shot, combat, movement history
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import torch


@dataclass
class MicroState:
    """Structured micro state for a unit."""

    # Unit identity (4 dims)
    unit_type: float          # Encoded unit type
    is_hero: float            # 1 if hero unit
    veterancy: float          # 0-1 veterancy level
    has_ability: float        # 1 if has special ability

    # Unit status (8 dims)
    health: float             # 0-1 current health ratio
    shield: float             # 0-1 shield ratio (or 0 if none)
    ammunition: float         # 0-1 ammo ratio
    cooldown: float           # 0-1 weapon cooldown
    speed: float              # Normalized movement speed
    attack_range: float       # Normalized attack range
    dps: float                # Normalized damage per second
    armor: float              # Normalized armor value

    # Situational (12 dims)
    nearest_enemy_dist: float   # 0-1 normalized distance
    nearest_enemy_angle: float  # -1 to 1 (normalized from -pi to pi)
    nearest_enemy_health: float # 0-1 enemy health ratio
    nearest_enemy_threat: float # 0-1 threat level
    nearest_ally_dist: float    # 0-1 normalized distance
    in_cover: float             # 1 if in cover
    under_fire: float           # 1 if taking damage
    ability_ready: float        # 1 if ability cooldown complete
    target_dist: float          # 0-1 distance to current target
    target_health: float        # 0-1 target health ratio
    target_type: float          # Encoded target type
    can_retreat: float          # 1 if retreat path available

    # Team context (4 dims)
    objective_type: float       # Encoded team objective
    objective_dir: float        # -1 to 1 direction to objective
    team_role: float            # Encoded role (tank, dps, support)
    priority: float             # 0-1 micro priority from team

    # Temporal (4 dims)
    time_since_hit: float       # Normalized time since damage taken
    time_since_shot: float      # Normalized time since last shot
    time_in_combat: float       # Normalized time in combat
    movement_history: float     # Encoded recent movement (stationary, forward, back)

    def to_tensor(self) -> torch.Tensor:
        """Convert to 32-dim tensor."""
        features = [
            # Unit identity (4)
            self.unit_type, self.is_hero, self.veterancy, self.has_ability,
            # Unit status (8)
            self.health, self.shield, self.ammunition, self.cooldown,
            self.speed, self.attack_range, self.dps, self.armor,
            # Situational (12)
            self.nearest_enemy_dist, self.nearest_enemy_angle,
            self.nearest_enemy_health, self.nearest_enemy_threat,
            self.nearest_ally_dist, self.in_cover, self.under_fire,
            self.ability_ready, self.target_dist, self.target_health,
            self.target_type, self.can_retreat,
            # Team context (4)
            self.objective_type, self.objective_dir, self.team_role, self.priority,
            # Temporal (4)
            self.time_since_hit, self.time_since_shot,
            self.time_in_combat, self.movement_history,
        ]
        return torch.tensor(features, dtype=torch.float32)

    @classmethod
    def zeros(cls) -> 'MicroState':
        """Create a zero-initialized state."""
        return cls(
            unit_type=0.0, is_hero=0.0, veterancy=0.0, has_ability=0.0,
            health=1.0, shield=0.0, ammunition=1.0, cooldown=0.0,
            speed=0.5, attack_range=0.5, dps=0.5, armor=0.5,
            nearest_enemy_dist=1.0, nearest_enemy_angle=0.0,
            nearest_enemy_health=1.0, nearest_enemy_threat=0.0,
            nearest_ally_dist=1.0, in_cover=0.0, under_fire=0.0,
            ability_ready=0.0, target_dist=1.0, target_health=1.0,
            target_type=0.0, can_retreat=1.0,
            objective_type=0.0, objective_dir=0.0, team_role=0.0, priority=0.5,
            time_since_hit=1.0, time_since_shot=1.0,
            time_in_combat=0.0, movement_history=0.0,
        )


def build_micro_state(unit_data: Dict, team_objective: Optional[Dict] = None) -> MicroState:
    """
    Build micro state from unit data.

    Args:
        unit_data: Raw unit state from C++
        team_objective: Current team objective context

    Returns:
        MicroState ready for neural network
    """
    # Unit identity (4)
    unit_type = _encode_unit_type(unit_data.get('type', 'infantry'))
    is_hero = 1.0 if unit_data.get('is_hero', False) else 0.0
    veterancy = np.clip(unit_data.get('veterancy', 0) / 3.0, 0.0, 1.0)  # Max vet level 3
    has_ability = 1.0 if unit_data.get('has_ability', False) else 0.0

    # Unit status (8)
    health = np.clip(unit_data.get('health', 1.0), 0.0, 1.0)
    shield = np.clip(unit_data.get('shield', 0.0), 0.0, 1.0)
    ammunition = np.clip(unit_data.get('ammunition', 1.0), 0.0, 1.0)
    cooldown = np.clip(unit_data.get('cooldown', 0.0), 0.0, 1.0)
    speed = np.clip(unit_data.get('speed', 50.0) / 100.0, 0.0, 1.0)
    attack_range = np.clip(unit_data.get('range', 200.0) / 500.0, 0.0, 1.0)
    dps = np.clip(unit_data.get('dps', 10.0) / 50.0, 0.0, 1.0)
    armor = np.clip(unit_data.get('armor', 0.0) / 100.0, 0.0, 1.0)

    # Situational (12)
    situational = unit_data.get('situational', {})
    nearest_enemy_dist = np.clip(situational.get('enemy_dist', 500.0) / 500.0, 0.0, 1.0)
    nearest_enemy_angle = np.clip(situational.get('enemy_angle', 0.0) / np.pi, -1.0, 1.0)
    nearest_enemy_health = np.clip(situational.get('enemy_health', 1.0), 0.0, 1.0)
    nearest_enemy_threat = np.clip(situational.get('enemy_threat', 0.0), 0.0, 1.0)
    nearest_ally_dist = np.clip(situational.get('ally_dist', 500.0) / 500.0, 0.0, 1.0)
    in_cover = 1.0 if situational.get('in_cover', False) else 0.0
    under_fire = 1.0 if situational.get('under_fire', False) else 0.0
    ability_ready = 1.0 if situational.get('ability_ready', False) else 0.0

    target = unit_data.get('target', {})
    target_dist = np.clip(target.get('dist', 500.0) / 500.0, 0.0, 1.0)
    target_health = np.clip(target.get('health', 1.0), 0.0, 1.0)
    target_type = _encode_unit_type(target.get('type', 'none'))
    can_retreat = 1.0 if situational.get('can_retreat', True) else 0.0

    # Team context (4)
    if team_objective is None:
        team_objective = {}
    objective_type = _encode_objective_type(team_objective.get('type', 'none'))
    objective_dir = np.clip(team_objective.get('direction', 0.0) / np.pi, -1.0, 1.0)
    team_role = _encode_team_role(unit_data.get('role', 'dps'))
    priority = np.clip(team_objective.get('micro_priority', 0.5), 0.0, 1.0)

    # Temporal (4)
    temporal = unit_data.get('temporal', {})
    time_since_hit = np.clip(temporal.get('since_hit', 60.0) / 60.0, 0.0, 1.0)
    time_since_shot = np.clip(temporal.get('since_shot', 60.0) / 60.0, 0.0, 1.0)
    time_in_combat = np.clip(temporal.get('in_combat', 0.0) / 60.0, 0.0, 1.0)
    movement_history = _encode_movement(temporal.get('movement', 'stationary'))

    return MicroState(
        unit_type=unit_type, is_hero=is_hero, veterancy=veterancy, has_ability=has_ability,
        health=health, shield=shield, ammunition=ammunition, cooldown=cooldown,
        speed=speed, attack_range=attack_range, dps=dps, armor=armor,
        nearest_enemy_dist=nearest_enemy_dist, nearest_enemy_angle=nearest_enemy_angle,
        nearest_enemy_health=nearest_enemy_health, nearest_enemy_threat=nearest_enemy_threat,
        nearest_ally_dist=nearest_ally_dist, in_cover=in_cover, under_fire=under_fire,
        ability_ready=ability_ready, target_dist=target_dist, target_health=target_health,
        target_type=target_type, can_retreat=can_retreat,
        objective_type=objective_type, objective_dir=objective_dir,
        team_role=team_role, priority=priority,
        time_since_hit=time_since_hit, time_since_shot=time_since_shot,
        time_in_combat=time_in_combat, movement_history=movement_history,
    )


def _encode_unit_type(unit_type: str) -> float:
    """Encode unit type as normalized float."""
    types = {
        'none': 0.0,
        'infantry': 0.2,
        'vehicle': 0.4,
        'aircraft': 0.6,
        'tank': 0.5,
        'artillery': 0.7,
        'structure': 0.8,
        'hero': 1.0,
    }
    return types.get(unit_type.lower(), 0.0)


def _encode_objective_type(obj_type: str) -> float:
    """Encode team objective type."""
    types = {
        'none': 0.0,
        'attack': 0.25,
        'defend': 0.5,
        'retreat': 0.75,
        'hold': 1.0,
    }
    return types.get(obj_type.lower(), 0.0)


def _encode_team_role(role: str) -> float:
    """Encode unit's role in the team."""
    roles = {
        'tank': 0.0,      # Front line, absorb damage
        'dps': 0.33,      # Main damage dealer
        'support': 0.67,  # Healing, buffs
        'scout': 1.0,     # Reconnaissance
    }
    return roles.get(role.lower(), 0.33)


def _encode_movement(movement: str) -> float:
    """Encode recent movement pattern."""
    patterns = {
        'stationary': 0.0,
        'forward': 0.5,
        'backward': 1.0,
        'lateral': 0.75,
    }
    return patterns.get(movement.lower(), 0.0)


def state_tensor_from_dict(unit_data: Dict, team_objective: Optional[Dict] = None) -> torch.Tensor:
    """Quick conversion from unit data dict to tensor."""
    state = build_micro_state(unit_data, team_objective)
    return state.to_tensor()


if __name__ == '__main__':
    print("Testing MicroState...")

    # Test zero state
    state = MicroState.zeros()
    tensor = state.to_tensor()
    print(f"Zero state tensor shape: {tensor.shape}")
    assert tensor.shape == (32,), f"Expected (32,), got {tensor.shape}"

    # Test building from unit data
    unit_data = {
        'type': 'tank',
        'is_hero': False,
        'veterancy': 2,
        'health': 0.75,
        'ammunition': 0.5,
        'speed': 40.0,
        'range': 300.0,
        'dps': 25.0,
        'situational': {
            'enemy_dist': 200.0,
            'enemy_angle': 0.5,
            'enemy_health': 0.6,
            'enemy_threat': 0.7,
            'under_fire': True,
        },
        'target': {
            'dist': 150.0,
            'health': 0.4,
            'type': 'infantry',
        },
        'temporal': {
            'since_hit': 2.0,
            'since_shot': 0.5,
            'in_combat': 30.0,
        },
        'role': 'tank',
    }

    team_obj = {
        'type': 'attack',
        'direction': 0.3,
        'micro_priority': 0.8,
    }

    state = build_micro_state(unit_data, team_obj)
    tensor = state.to_tensor()

    print(f"\nBuilt state tensor shape: {tensor.shape}")
    print(f"Health: {state.health:.2f}")
    print(f"Under fire: {state.under_fire}")
    print(f"Enemy dist: {state.nearest_enemy_dist:.2f}")
    print(f"Target health: {state.target_health:.2f}")
    print(f"Time in combat: {state.time_in_combat:.2f}")

    print("\nMicroState test passed!")
