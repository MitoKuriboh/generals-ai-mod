"""
Micro Reward Functions

Rewards for unit-level decisions focused on:
- Survival (weighted by unit value)
- Damage efficiency
- Successful kiting
- Ability usage
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from .model import MicroAction


@dataclass
class MicroRewardConfig:
    """Configuration for micro rewards."""

    # Survival (weighted by unit DPS value)
    survival_weight: float = 0.3

    # Damage efficiency
    damage_efficiency_weight: float = 0.4

    # Kiting success
    kiting_weight: float = 0.3

    # Ability usage
    ability_weight: float = 0.2

    # Penalties
    death_penalty: float = -1.0      # Per unit DPS
    friendly_fire_penalty: float = -0.5
    wasted_ammo_penalty: float = -0.05

    # Bonuses
    kill_bonus: float = 0.5          # Per target DPS
    dodge_bonus: float = 0.3         # Took no damage when could have


def micro_reward(
    state: Dict,
    action: int,
    next_state: Dict,
    team_objective: Dict,
    config: Optional[MicroRewardConfig] = None
) -> float:
    """
    Calculate reward for a micro decision.

    Args:
        state: Current micro state
        action: Action taken (MicroAction enum value)
        next_state: Resulting micro state
        team_objective: Current team objective
        config: Reward configuration

    Returns:
        Scalar reward value
    """
    if config is None:
        config = MicroRewardConfig()

    reward = 0.0

    # 1. Survival (weighted by unit value)
    reward += config.survival_weight * _survival_reward(state, next_state, config)

    # 2. Damage efficiency
    reward += config.damage_efficiency_weight * _damage_efficiency_reward(
        state, action, next_state
    )

    # 3. Kiting success
    reward += config.kiting_weight * _kiting_reward(state, action, next_state)

    # 4. Ability usage
    reward += config.ability_weight * _ability_reward(state, action, next_state)

    return reward


def _survival_reward(state: Dict, next_state: Dict, config: MicroRewardConfig) -> float:
    """
    Reward for staying alive, weighted by unit value.
    """
    current_health = state.get('health', 1.0)
    next_health = next_state.get('health', 1.0)
    unit_dps = state.get('dps', 10.0) / 50.0  # Normalized DPS

    # Unit died
    if next_health <= 0 and current_health > 0:
        return config.death_penalty * unit_dps

    # Health lost
    health_lost = current_health - next_health
    if health_lost > 0:
        return -0.2 * health_lost * unit_dps

    # Stayed alive without damage
    if health_lost <= 0 and state.get('situational', {}).get('under_fire', False):
        return config.dodge_bonus

    return 0.1 if next_health > 0 else 0.0


def _damage_efficiency_reward(state: Dict, action: int, next_state: Dict) -> float:
    """
    Reward for dealing damage efficiently.
    """
    damage_dealt = next_state.get('damage_dealt', 0.0)
    ammo_used = state.get('ammunition', 1.0) - next_state.get('ammunition', 1.0)

    # Reward damage dealt
    reward = 0.3 * damage_dealt / 50.0  # Normalize to typical damage values

    # Killed an enemy
    target_health_before = state.get('target', {}).get('health', 1.0)
    target_health_after = next_state.get('target', {}).get('health', 1.0)

    if target_health_before > 0 and target_health_after <= 0:
        target_dps = state.get('target', {}).get('dps', 10.0) / 50.0
        reward += 0.5 * target_dps  # Kill bonus

    # Penalize wasted ammo (fired but missed/no damage)
    if ammo_used > 0 and damage_dealt == 0:
        reward -= 0.05 * ammo_used

    return reward


def _kiting_reward(state: Dict, action: int, next_state: Dict) -> float:
    """
    Reward for successful kiting (dealing damage while avoiding it).
    """
    situational = state.get('situational', {})
    next_situational = next_state.get('situational', {})

    under_fire = situational.get('under_fire', False)
    next_under_fire = next_situational.get('under_fire', False)

    current_health = state.get('health', 1.0)
    next_health = next_state.get('health', 1.0)
    took_damage = next_health < current_health

    # Successful kite: was under fire, moved backward/flank, took no damage
    if under_fire and action in [MicroAction.MOVE_BACKWARD, MicroAction.MOVE_FLANK]:
        if not took_damage:
            return 0.5  # Successful dodge
        else:
            return 0.1  # At least tried to kite

    # Moving backward when not under fire is suboptimal
    if not under_fire and action == MicroAction.MOVE_BACKWARD:
        return -0.1

    # Moving forward when under fire and low health is risky
    if under_fire and action == MicroAction.MOVE_FORWARD:
        if current_health < 0.3:
            return -0.2  # Risky play
        else:
            return 0.0  # Aggressive but acceptable

    return 0.0


def _ability_reward(state: Dict, action: int, next_state: Dict) -> float:
    """
    Reward for appropriate ability usage.
    """
    ability_ready = state.get('situational', {}).get('ability_ready', False)

    if action == MicroAction.USE_ABILITY:
        if ability_ready:
            # Used ability when available - generally good
            return 0.3
        else:
            # Tried to use ability when on cooldown - slight penalty
            return -0.1

    # Had ability ready but didn't use in combat
    if ability_ready and state.get('situational', {}).get('under_fire', False):
        if action not in [MicroAction.USE_ABILITY, MicroAction.RETREAT]:
            return -0.05  # Slight penalty for not using available tools

    return 0.0


def intrinsic_micro_reward(state: Dict, action: int, next_state: Dict) -> float:
    """
    Intrinsic rewards for exploration in micro control.
    """
    reward = 0.0

    # Curiosity: reward trying different actions
    # (This would need action history tracking in practice)

    # Progress toward target
    target_dist_before = state.get('target', {}).get('dist', 500.0)
    target_dist_after = next_state.get('target', {}).get('dist', 500.0)

    if action in [MicroAction.ATTACK_CURRENT, MicroAction.ATTACK_NEAREST,
                  MicroAction.MOVE_FORWARD]:
        # Reward getting closer when attacking
        if target_dist_after < target_dist_before:
            reward += 0.1 * (target_dist_before - target_dist_after) / 500.0

    return reward


class MicroRewardTracker:
    """Track micro rewards across a unit's lifetime."""

    def __init__(self, config: Optional[MicroRewardConfig] = None):
        self.config = config or MicroRewardConfig()
        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.survival_total = 0.0
        self.damage_total = 0.0
        self.kiting_total = 0.0
        self.ability_total = 0.0
        self.kills = 0
        self.deaths = 0
        self.steps = 0

    def add_step(self, state: Dict, action: int, next_state: Dict,
                 team_objective: Dict) -> float:
        """Record a step and return the reward."""

        survival = self.config.survival_weight * _survival_reward(
            state, next_state, self.config
        )
        damage = self.config.damage_efficiency_weight * _damage_efficiency_reward(
            state, action, next_state
        )
        kiting = self.config.kiting_weight * _kiting_reward(state, action, next_state)
        ability = self.config.ability_weight * _ability_reward(state, action, next_state)

        total = survival + damage + kiting + ability

        self.survival_total += survival
        self.damage_total += damage
        self.kiting_total += kiting
        self.ability_total += ability
        self.total_reward += total
        self.steps += 1

        # Track kills/deaths
        target_health_before = state.get('target', {}).get('health', 1.0)
        target_health_after = next_state.get('target', {}).get('health', 1.0)
        if target_health_before > 0 and target_health_after <= 0:
            self.kills += 1

        if state.get('health', 1.0) > 0 and next_state.get('health', 1.0) <= 0:
            self.deaths += 1

        return total

    def get_stats(self) -> Dict[str, float]:
        """Get reward statistics."""
        return {
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.steps),
            'survival': self.survival_total,
            'damage': self.damage_total,
            'kiting': self.kiting_total,
            'ability': self.ability_total,
            'kills': self.kills,
            'deaths': self.deaths,
            'kd_ratio': self.kills / max(1, self.deaths),
            'steps': self.steps,
        }


if __name__ == '__main__':
    print("Testing micro rewards...")

    config = MicroRewardConfig()

    # Test survival reward
    state = {'health': 0.8, 'dps': 25.0, 'situational': {'under_fire': True}}
    next_state = {'health': 0.8}  # Survived without damage
    rew = _survival_reward(state, next_state, config)
    print(f"Survive under fire (no damage): {rew:.3f}")
    assert rew > 0

    # Test damage efficiency
    state = {'ammunition': 1.0, 'target': {'health': 0.5, 'dps': 20.0}}
    next_state = {'ammunition': 0.9, 'damage_dealt': 30.0, 'target': {'health': 0.0}}
    rew = _damage_efficiency_reward(state, MicroAction.ATTACK_CURRENT, next_state)
    print(f"Kill target: {rew:.3f}")
    assert rew > 0

    # Test kiting
    state = {'health': 0.6, 'situational': {'under_fire': True}}
    next_state = {'health': 0.6}  # No damage taken
    rew = _kiting_reward(state, MicroAction.MOVE_BACKWARD, next_state)
    print(f"Successful kite: {rew:.3f}")
    assert rew > 0

    # Test ability usage
    state = {'situational': {'ability_ready': True, 'under_fire': True}}
    next_state = {}
    rew = _ability_reward(state, MicroAction.USE_ABILITY, next_state)
    print(f"Use ability when ready: {rew:.3f}")
    assert rew > 0

    # Test full reward
    state = {
        'health': 0.7,
        'dps': 30.0,
        'ammunition': 0.8,
        'situational': {
            'under_fire': True,
            'ability_ready': False,
        },
        'target': {
            'health': 0.4,
            'dist': 200.0,
            'dps': 15.0,
        },
    }
    next_state = {
        'health': 0.65,
        'ammunition': 0.7,
        'damage_dealt': 25.0,
        'situational': {'under_fire': True},
        'target': {'health': 0.1, 'dist': 180.0},
    }

    total = micro_reward(state, MicroAction.ATTACK_CURRENT, next_state, {}, config)
    print(f"\nTotal micro reward: {total:.3f}")

    # Test reward tracker
    tracker = MicroRewardTracker(config)
    for _ in range(10):
        tracker.add_step(state, MicroAction.ATTACK_CURRENT, next_state, {})

    stats = tracker.get_stats()
    print(f"\nReward tracker stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\nMicro rewards test passed!")
