"""
Simulated Hierarchical Environment for Joint Training

Provides a simplified simulation of the game that supports all three layers:
- Strategic: Overall game state and macro decisions
- Tactical: Team-level objectives and positioning
- Micro: Unit-level combat decisions

This allows training and testing the hierarchical RL system without the real game.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import random


@dataclass
class SimUnit:
    """Simulated unit for micro control."""
    unit_id: int
    team_id: int
    unit_type: float  # 0-1 encoding
    x: float
    y: float
    health: float = 1.0
    dps: float = 0.5
    is_hero: bool = False
    veterancy: float = 0.0
    target_id: Optional[int] = None


@dataclass
class SimTeam:
    """Simulated team for tactical control."""
    team_id: int
    units: List[int] = field(default_factory=list)
    objective_x: float = 0.5
    objective_y: float = 0.5
    objective_type: int = 0  # 0=attack, 1=defend, 2=retreat


@dataclass
class SimPlayer:
    """Simulated player state."""
    teams: List[SimTeam] = field(default_factory=list)
    units: Dict[int, SimUnit] = field(default_factory=dict)
    money: float = 3.0  # log10 scale
    structures: float = 0.7  # log10 scale
    tech_level: float = 0.2


class SimulatedHierarchicalEnv:
    """
    Simulated environment for hierarchical RL training.

    Supports the full three-layer architecture without requiring the real game.
    """

    def __init__(self,
                 num_teams: int = 3,
                 units_per_team: int = 5,
                 episode_length: int = 200,
                 map_size: float = 1.0):
        self.num_teams = num_teams
        self.units_per_team = units_per_team
        self.episode_length = episode_length
        self.map_size = map_size

        self.step_count = 0
        self.player: Optional[SimPlayer] = None
        self.enemy: Optional[SimPlayer] = None
        self.next_unit_id = 0
        self.done = False

    def reset(self) -> Dict:
        """Reset environment and return initial strategic state."""
        self.step_count = 0
        self.next_unit_id = 0
        self.done = False

        # Create player
        self.player = SimPlayer()
        self._spawn_teams(self.player, base_x=0.2, base_y=0.5, is_player=True)

        # Create enemy
        self.enemy = SimPlayer()
        self._spawn_teams(self.enemy, base_x=0.8, base_y=0.5, is_player=False)

        return self._get_strategic_state()

    def _spawn_teams(self, player: SimPlayer, base_x: float, base_y: float,
                     is_player: bool):
        """Spawn teams for a player."""
        for t in range(self.num_teams):
            team = SimTeam(team_id=t if is_player else t + 100)
            team.objective_x = base_x + 0.3 if is_player else base_x - 0.3
            team.objective_y = 0.3 + t * 0.2

            for u in range(self.units_per_team):
                unit_id = self.next_unit_id
                self.next_unit_id += 1

                unit = SimUnit(
                    unit_id=unit_id,
                    team_id=team.team_id,
                    unit_type=np.random.uniform(0, 0.9),
                    x=base_x + np.random.uniform(-0.1, 0.1),
                    y=base_y + t * 0.1 + np.random.uniform(-0.05, 0.05),
                    health=1.0,
                    dps=np.random.uniform(0.3, 0.7),
                    is_hero=(u == 0 and t == 0),
                    veterancy=0.0,
                )
                player.units[unit_id] = unit
                team.units.append(unit_id)

            player.teams.append(team)

    def step(self) -> Tuple[Dict, float, bool, Dict]:
        """
        Advance the simulation by one timestep.

        Returns:
            state: Strategic state dict
            reward: Strategic reward
            done: Whether episode is over
            info: Additional info
        """
        self.step_count += 1

        # Simulate combat between nearby units
        self._simulate_combat()

        # Check for termination
        own_alive = sum(1 for u in self.player.units.values() if u.health > 0)
        enemy_alive = sum(1 for u in self.enemy.units.values() if u.health > 0)

        if own_alive == 0:
            self.done = True
            return self._get_strategic_state(), -100.0, True, {'won': False}

        if enemy_alive == 0:
            self.done = True
            return self._get_strategic_state(), 100.0, True, {'won': True}

        if self.step_count >= self.episode_length:
            self.done = True
            won = own_alive > enemy_alive
            reward = 50.0 if won else -50.0
            return self._get_strategic_state(), reward, True, {'won': won}

        # Calculate incremental reward
        reward = 0.1 * (own_alive - enemy_alive) / max(own_alive + enemy_alive, 1)

        return self._get_strategic_state(), reward, False, {}

    def _simulate_combat(self):
        """Simulate combat between nearby units."""
        all_player_units = [u for u in self.player.units.values() if u.health > 0]
        all_enemy_units = [u for u in self.enemy.units.values() if u.health > 0]

        # Player units attack enemy units
        for unit in all_player_units:
            if unit.target_id is not None and unit.target_id in self.enemy.units:
                target = self.enemy.units[unit.target_id]
                if target.health > 0:
                    dist = np.sqrt((unit.x - target.x)**2 + (unit.y - target.y)**2)
                    if dist < 0.15:  # In range
                        target.health -= unit.dps * 0.05
                        unit.veterancy = min(1.0, unit.veterancy + 0.01)

        # Enemy units attack player units (simple AI)
        for unit in all_enemy_units:
            # Find nearest player unit
            nearest = None
            nearest_dist = float('inf')
            for punit in all_player_units:
                dist = np.sqrt((unit.x - punit.x)**2 + (unit.y - punit.y)**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = punit

            if nearest and nearest_dist < 0.15:
                nearest.health -= unit.dps * 0.05
                unit.veterancy = min(1.0, unit.veterancy + 0.01)

    def get_team_states(self) -> Dict[int, Dict]:
        """Get states for all player teams."""
        states = {}
        for team in self.player.teams:
            states[team.team_id] = self._get_team_state_dict(team)
        return states

    def get_team_state(self, team_id: int) -> Dict:
        """Get state for a specific team."""
        for team in self.player.teams:
            if team.team_id == team_id:
                return self._get_team_state_dict(team)
        return {}

    def get_unit_states(self, team_id: int = None) -> Dict[int, Dict]:
        """Get states for units, optionally filtered by team."""
        states = {}
        for unit_id, unit in self.player.units.items():
            if unit.health <= 0:
                continue
            if team_id is not None and unit.team_id != team_id:
                continue
            states[unit_id] = self._get_unit_state_dict(unit)
        return states

    def get_unit_state(self, unit_id: int) -> Dict:
        """Get state for a specific unit."""
        if unit_id in self.player.units:
            return self._get_unit_state_dict(self.player.units[unit_id])
        return {}

    def apply_tactical_action(self, team_id: int, action) -> float:
        """
        Apply tactical action to a team.

        Args:
            team_id: Team to command
            action: TacticalCommand or dict with action info

        Returns:
            Reward for the action
        """
        team = None
        for t in self.player.teams:
            if t.team_id == team_id:
                team = t
                break

        if team is None:
            return 0.0

        # Extract action components
        if hasattr(action, 'action'):
            action_type = action.action
            target_x = getattr(action, 'target_x', 0.5)
            target_y = getattr(action, 'target_y', 0.5)
        elif isinstance(action, dict):
            action_type = action.get('action', 0)
            target_x = action.get('target_x', 0.5)
            target_y = action.get('target_y', 0.5)
        else:
            # Assume it's just an action index
            action_type = int(action) if hasattr(action, 'item') else action
            target_x = 0.5
            target_y = 0.5

        # Update team objective
        team.objective_x = float(target_x)
        team.objective_y = float(target_y)
        team.objective_type = int(action_type) % 8

        # Move units toward objective
        reward = 0.0
        for unit_id in team.units:
            if unit_id in self.player.units:
                unit = self.player.units[unit_id]
                if unit.health <= 0:
                    continue

                # Move toward objective
                dx = team.objective_x - unit.x
                dy = team.objective_y - unit.y
                dist = np.sqrt(dx*dx + dy*dy)

                if dist > 0.01:
                    move_speed = 0.02
                    unit.x += (dx / dist) * move_speed
                    unit.y += (dy / dist) * move_speed

                # Find target enemy
                if action_type in [0, 1, 5]:  # Attack actions
                    nearest_enemy = self._find_nearest_enemy(unit)
                    if nearest_enemy:
                        unit.target_id = nearest_enemy.unit_id

                        # Calculate engagement reward
                        enemy_dist = np.sqrt(
                            (unit.x - nearest_enemy.x)**2 +
                            (unit.y - nearest_enemy.y)**2
                        )
                        if enemy_dist < 0.2:
                            reward += 0.1

        return reward

    def apply_micro_action(self, unit_id: int, action) -> float:
        """
        Apply micro action to a unit.

        Args:
            unit_id: Unit to command
            action: MicroCommand or action index

        Returns:
            Reward for the action
        """
        if unit_id not in self.player.units:
            return 0.0

        unit = self.player.units[unit_id]
        if unit.health <= 0:
            return 0.0

        # Extract action components
        if hasattr(action, 'action'):
            action_type = action.action
            move_angle = getattr(action, 'move_angle', 0.0)
            move_distance = getattr(action, 'move_distance', 0.0)
        elif isinstance(action, dict):
            action_type = action.get('action', 0)
            move_angle = action.get('move_angle', 0.0)
            move_distance = action.get('move_distance', 0.0)
        else:
            action_type = int(action) if hasattr(action, 'item') else action
            move_angle = 0.0
            move_distance = 0.0

        reward = 0.0
        action_type = int(action_type) % 11

        # Handle different micro actions
        if action_type in [0, 1, 2, 3]:  # Attack actions
            target = self._find_nearest_enemy(unit)
            if target:
                unit.target_id = target.unit_id
                dist = np.sqrt((unit.x - target.x)**2 + (unit.y - target.y)**2)
                if dist < 0.15:
                    reward += 0.05  # In combat

        elif action_type in [4, 5, 6]:  # Move actions
            move_dist = float(move_distance) * 0.05
            angle = float(move_angle)

            if action_type == 5:  # Backward/kite
                # Move away from nearest enemy
                enemy = self._find_nearest_enemy(unit)
                if enemy:
                    dx = unit.x - enemy.x
                    dy = unit.y - enemy.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0.01:
                        unit.x += (dx / dist) * move_dist
                        unit.y += (dy / dist) * move_dist
                        reward += 0.1  # Kiting is good

            else:  # Forward or flank
                unit.x += np.cos(angle) * move_dist
                unit.y += np.sin(angle) * move_dist

        elif action_type == 9:  # Retreat
            # Move toward base
            dx = 0.2 - unit.x
            dy = 0.5 - unit.y
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 0.01:
                unit.x += (dx / dist) * 0.03
                unit.y += (dy / dist) * 0.03

        # Clamp position to map
        unit.x = max(0, min(1, unit.x))
        unit.y = max(0, min(1, unit.y))

        # Survival bonus
        reward += 0.01

        return reward

    def _find_nearest_enemy(self, unit: SimUnit) -> Optional[SimUnit]:
        """Find nearest enemy unit to the given unit."""
        nearest = None
        nearest_dist = float('inf')

        for enemy in self.enemy.units.values():
            if enemy.health <= 0:
                continue
            dist = np.sqrt((unit.x - enemy.x)**2 + (unit.y - enemy.y)**2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = enemy

        return nearest

    def _get_strategic_state(self) -> Dict:
        """Build strategic state dict (44 dimensions)."""
        own_infantry = sum(1 for u in self.player.units.values()
                          if u.health > 0 and u.unit_type < 0.3)
        own_vehicles = sum(1 for u in self.player.units.values()
                          if u.health > 0 and 0.3 <= u.unit_type < 0.6)
        own_aircraft = sum(1 for u in self.player.units.values()
                          if u.health > 0 and u.unit_type >= 0.6)

        enemy_infantry = sum(1 for u in self.enemy.units.values()
                            if u.health > 0 and u.unit_type < 0.3)
        enemy_vehicles = sum(1 for u in self.enemy.units.values()
                            if u.health > 0 and 0.3 <= u.unit_type < 0.6)
        enemy_aircraft = sum(1 for u in self.enemy.units.values()
                            if u.health > 0 and u.unit_type >= 0.6)

        own_total = own_infantry + own_vehicles + own_aircraft
        enemy_total = enemy_infantry + enemy_vehicles + enemy_aircraft

        return {
            'player': 1,
            'money': self.player.money,
            'power': 10.0,
            'income': 2.0,
            'supply': 0.5,
            'own_infantry': [np.log10(own_infantry + 1), 0.8, 0.0],
            'own_vehicles': [np.log10(own_vehicles + 1), 0.8, 0.0],
            'own_aircraft': [np.log10(own_aircraft + 1), 0.8, 0.0],
            'own_structures': [self.player.structures, 0.9, 0.0],
            'enemy_infantry': [np.log10(enemy_infantry + 1), 0.7, 0.0],
            'enemy_vehicles': [np.log10(enemy_vehicles + 1), 0.7, 0.0],
            'enemy_aircraft': [np.log10(enemy_aircraft + 1), 0.7, 0.0],
            'enemy_structures': [self.enemy.structures, 0.8, 0.0],
            'game_time': self.step_count * 0.5 / 60.0,  # In minutes
            'tech_level': self.player.tech_level,
            'base_threat': 0.2,
            'army_strength': (own_total + 1) / max(enemy_total + 1, 1),
            'under_attack': 0.0,
            'distance_to_enemy': 0.5,
            'is_usa': 1.0,
            'is_china': 0.0,
            'is_gla': 0.0,
        }

    def _get_team_state_dict(self, team: SimTeam) -> Dict:
        """Build tactical state dict (64 dimensions)."""
        alive_units = [self.player.units[uid] for uid in team.units
                      if uid in self.player.units and self.player.units[uid].health > 0]

        if not alive_units:
            return {'empty': True}

        avg_x = sum(u.x for u in alive_units) / len(alive_units)
        avg_y = sum(u.y for u in alive_units) / len(alive_units)
        avg_health = sum(u.health for u in alive_units) / len(alive_units)

        return {
            'team_id': team.team_id,
            'strategy_embedding': [0.5] * 8,
            'team_count': len(alive_units),
            'team_health': avg_health,
            'center_x': avg_x,
            'center_y': avg_y,
            'objective_x': team.objective_x,
            'objective_y': team.objective_y,
            'objective_type': team.objective_type,
            'under_fire': 0.0,
            'cohesion': 0.8,
            'nearby_enemies': [0.1, 0.1, 0.1, 0.1],
            'nearby_allies': [0.2, 0.2, 0.2, 0.2],
        }

    def _get_unit_state_dict(self, unit: SimUnit) -> Dict:
        """Build micro state dict (32 dimensions)."""
        nearest_enemy = self._find_nearest_enemy(unit)
        enemy_dist = 1.0
        enemy_angle = 0.0
        enemy_health = 0.0

        if nearest_enemy:
            dx = nearest_enemy.x - unit.x
            dy = nearest_enemy.y - unit.y
            enemy_dist = min(1.0, np.sqrt(dx*dx + dy*dy) / 0.3)
            enemy_angle = np.arctan2(dy, dx) / np.pi
            enemy_health = nearest_enemy.health

        return {
            'unit_id': unit.unit_id,
            'unit_type': unit.unit_type,
            'is_hero': 1.0 if unit.is_hero else 0.0,
            'veterancy': unit.veterancy,
            'health': unit.health,
            'dps': unit.dps,
            'x': unit.x,
            'y': unit.y,
            'nearest_enemy_dist': enemy_dist,
            'nearest_enemy_angle': enemy_angle,
            'nearest_enemy_health': enemy_health,
            'under_fire': 0.0,
            'objective_type': 0.0,
            'objective_dir': 0.0,
            'team_role': 0.5,
            'priority': 0.5,
        }

    def close(self):
        """Clean up resources."""
        pass

    def render(self):
        """Print current state."""
        own_alive = sum(1 for u in self.player.units.values() if u.health > 0)
        enemy_alive = sum(1 for u in self.enemy.units.values() if u.health > 0)
        print(f"Step {self.step_count}: Own={own_alive} Enemy={enemy_alive}")


if __name__ == '__main__':
    # Quick test
    print("Testing SimulatedHierarchicalEnv...")

    env = SimulatedHierarchicalEnv(num_teams=2, units_per_team=3, episode_length=50)
    state = env.reset()

    print(f"Initial strategic state keys: {list(state.keys())}")
    print(f"Num teams: {len(env.get_team_states())}")
    print(f"Num units: {len(env.get_unit_states())}")

    for step in range(50):
        # Apply tactical actions
        for team_id, team_state in env.get_team_states().items():
            reward = env.apply_tactical_action(team_id, {'action': 0, 'target_x': 0.8, 'target_y': 0.5})

        # Apply micro actions
        for unit_id, unit_state in env.get_unit_states().items():
            reward = env.apply_micro_action(unit_id, {'action': 1})

        # Step environment
        next_state, reward, done, info = env.step()

        if step % 10 == 0:
            env.render()

        if done:
            print(f"Episode ended at step {step+1}: {'Won' if info.get('won') else 'Lost'}")
            break

    print("Test passed!")
