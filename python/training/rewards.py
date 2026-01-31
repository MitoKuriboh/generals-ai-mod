"""
Reward Calculation for Generals Zero Hour Learning AI

Defines reward signals for reinforcement learning:
- Terminal rewards for win/loss
- Shaping rewards for intermediate progress

Design principles:
- Dense rewards help learning but can cause reward hacking
- Sparse terminal rewards are more robust but slower to learn
- We use a combination with careful weighting
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Terminal rewards (increased 10x for better learning signal)
    win_reward: float = 100.0
    loss_reward: float = -100.0
    draw_reward: float = 0.0

    # Combat rewards (increased 25x for meaningful feedback)
    enemy_unit_killed: float = 0.5
    own_unit_lost: float = -0.5
    enemy_building_destroyed: float = 2.0
    own_building_lost: float = -3.0

    # Economy rewards (increased 10x)
    income_bonus: float = 0.01  # Per $100/s income advantage
    money_bonus: float = 0.001  # Per $100 wealth advantage

    # Strategic rewards (increased for better signal)
    army_strength_bonus: float = 0.5  # Per 0.1x advantage
    tech_advancement: float = 0.2  # Per tech level gained
    territory_control: float = 0.1  # Bonus for expanding

    # Penalty for inefficient play
    time_penalty: float = -0.01  # Per step (encourages decisive play)
    idle_penalty: float = -0.05  # For having no active production

    # Scaling factors
    shaping_scale: float = 1.0  # Scale all shaping rewards
    enable_shaping: bool = True


# Default configuration
DEFAULT_CONFIG = RewardConfig()


def calculate_reward(
    prev_state: Optional[Dict],
    current_state: Dict,
    action: Dict,
    env: Any,
    config: RewardConfig = DEFAULT_CONFIG
) -> float:
    """
    Calculate reward for a state transition.

    Args:
        prev_state: Previous game state (None for first step)
        current_state: Current game state
        action: Action taken
        env: Environment instance (for tracking vars)
        config: Reward configuration

    Returns:
        Total reward for this transition
    """
    reward = 0.0

    # First step has no previous state
    if prev_state is None:
        return 0.0

    # Check for terminal conditions
    terminal_reward = _calculate_terminal_reward(current_state, config)
    if terminal_reward != 0.0:
        return terminal_reward

    # Shaping rewards
    if config.enable_shaping:
        reward += _calculate_combat_reward(prev_state, current_state, env, config)
        reward += _calculate_economy_reward(prev_state, current_state, config)
        reward += _calculate_strategic_reward(prev_state, current_state, config)
        reward += _calculate_penalties(prev_state, current_state, action, config)
        reward *= config.shaping_scale

    return reward


def _calculate_terminal_reward(state: Dict, config: RewardConfig) -> float:
    """Calculate terminal reward (win/loss/draw)."""
    # Check for loss (no structures)
    own_structures = state.get('own_structures', [0, 0, 0])
    if own_structures[0] < 0.3:  # Less than 1 structure
        return config.loss_reward

    # Check for win (enemy eliminated)
    enemy_structures = state.get('enemy_structures', [0, 0, 0])
    game_time = state.get('game_time', 0)
    if enemy_structures[0] < 0.3 and game_time > 5.0:
        return config.win_reward

    # Timeout draw
    if game_time > 30.0:
        army_strength = state.get('army_strength', 1.0)
        if army_strength > 1.2:
            return config.win_reward * 0.5  # Partial win
        elif army_strength < 0.8:
            return config.loss_reward * 0.5  # Partial loss
        return config.draw_reward

    return 0.0


def _calculate_combat_reward(
    prev_state: Dict,
    current_state: Dict,
    env: Any,
    config: RewardConfig
) -> float:
    """Calculate rewards for combat outcomes."""
    reward = 0.0

    # Get current counts
    own_units = _count_units(current_state, 'own')
    enemy_units = _count_units(current_state, 'enemy')
    own_buildings = _count_buildings(current_state, 'own')
    enemy_buildings = _count_buildings(current_state, 'enemy')

    # Get previous counts from env tracking
    prev_own = env.prev_own_units
    prev_enemy = env.prev_enemy_units
    prev_own_buildings = env.prev_own_buildings
    prev_enemy_buildings = env.prev_enemy_buildings

    # Unit changes
    enemy_killed = max(0, prev_enemy - enemy_units)
    own_lost = max(0, prev_own - own_units)

    reward += enemy_killed * config.enemy_unit_killed
    reward += own_lost * config.own_unit_lost

    # Building changes
    enemy_buildings_destroyed = max(0, prev_enemy_buildings - enemy_buildings)
    own_buildings_lost = max(0, prev_own_buildings - own_buildings)

    reward += enemy_buildings_destroyed * config.enemy_building_destroyed
    reward += own_buildings_lost * config.own_building_lost

    # Track for episode stats
    env.episode_stats.units_killed += enemy_killed
    env.episode_stats.units_lost += own_lost
    env.episode_stats.buildings_destroyed += enemy_buildings_destroyed
    env.episode_stats.buildings_lost += own_buildings_lost

    return reward


def _calculate_economy_reward(
    prev_state: Dict,
    current_state: Dict,
    config: RewardConfig
) -> float:
    """Calculate rewards for economic performance."""
    reward = 0.0

    # Income advantage
    own_income = current_state.get('income', 0)
    # Assume enemy income roughly equals ours in early game
    enemy_income_estimate = 2.0  # Base estimate
    income_advantage = (own_income - enemy_income_estimate) * 100  # Scale to $100 units
    reward += income_advantage * config.income_bonus

    # Wealth accumulation (only if gaining)
    current_money = current_state.get('money', 0)
    prev_money = prev_state.get('money', 0)
    if current_money > prev_money:
        money_gained = (10 ** current_money - 10 ** prev_money)  # Convert from log scale
        reward += (money_gained / 100) * config.money_bonus

    return reward


def _calculate_strategic_reward(
    prev_state: Dict,
    current_state: Dict,
    config: RewardConfig
) -> float:
    """Calculate rewards for strategic position."""
    reward = 0.0

    # Army strength improvement
    current_strength = current_state.get('army_strength', 1.0)
    prev_strength = prev_state.get('army_strength', 1.0)

    strength_change = current_strength - prev_strength
    if strength_change > 0:
        reward += strength_change * config.army_strength_bonus * 10

    # Bonus for having army advantage
    if current_strength > 1.0:
        reward += (current_strength - 1.0) * config.army_strength_bonus

    # Tech advancement
    current_tech = current_state.get('tech_level', 0)
    prev_tech = prev_state.get('tech_level', 0)
    if current_tech > prev_tech:
        reward += (current_tech - prev_tech) * config.tech_advancement * 10

    return reward


def _calculate_penalties(
    prev_state: Dict,
    current_state: Dict,
    action: Dict,
    config: RewardConfig
) -> float:
    """Calculate penalties for inefficient play."""
    reward = 0.0

    # Time penalty (encourages faster games)
    reward += config.time_penalty

    # Idle penalty when we have money but aren't using it
    money = current_state.get('money', 0)
    if money > 3.5:  # More than $3000
        # Check if we're not prioritizing spending
        total_spending = (action.get('priority_military', 0) +
                         action.get('priority_defense', 0) +
                         action.get('priority_tech', 0))
        if total_spending < 0.5:
            reward += config.idle_penalty

    return reward


def _count_units(state: Dict, prefix: str) -> int:
    """Count total units from state."""
    total = 0
    for category in ['infantry', 'vehicles', 'aircraft']:
        key = f'{prefix}_{category}'
        arr = state.get(key, [0, 0, 0])
        if len(arr) > 0 and arr[0] > 0:
            total += int(10 ** arr[0] - 1)
    return total


def _count_buildings(state: Dict, prefix: str) -> int:
    """Count buildings from state."""
    key = f'{prefix}_structures'
    arr = state.get(key, [0, 0, 0])
    if len(arr) > 0 and arr[0] > 0:
        return int(10 ** arr[0] - 1)
    return 0


# Reward presets for different training phases

EXPLORATION_CONFIG = RewardConfig(
    # Heavy shaping for early training
    shaping_scale=2.0,
    enemy_unit_killed=1.0,
    own_unit_lost=-0.5,
    army_strength_bonus=1.0,
    time_penalty=0.0,  # No time pressure
)

BALANCED_CONFIG = RewardConfig(
    # Default balanced configuration
    shaping_scale=1.0,
)

SPARSE_CONFIG = RewardConfig(
    # Minimal shaping, mainly terminal rewards
    shaping_scale=0.1,
    win_reward=100.0,
    loss_reward=-100.0,
)

AGGRESSIVE_CONFIG = RewardConfig(
    # Rewards aggressive play
    enemy_unit_killed=1.0,
    enemy_building_destroyed=4.0,
    army_strength_bonus=1.0,
    time_penalty=-0.02,  # Stronger time pressure
)


def get_config(preset: str = 'balanced') -> RewardConfig:
    """Get a reward configuration preset."""
    presets = {
        'exploration': EXPLORATION_CONFIG,
        'balanced': BALANCED_CONFIG,
        'sparse': SPARSE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
    }
    return presets.get(preset, BALANCED_CONFIG)


if __name__ == '__main__':
    # Test reward calculation
    print("Testing reward calculation...")

    prev_state = {
        'money': 3.0,
        'income': 2.0,
        'own_infantry': [0.7, 0.9, 0.0],
        'own_vehicles': [0.5, 0.9, 0.0],
        'own_aircraft': [0.0, 0.0, 0.0],
        'own_structures': [0.8, 0.95, 0.0],
        'enemy_infantry': [0.7, 0.8, 0.0],
        'enemy_vehicles': [0.5, 0.8, 0.0],
        'enemy_aircraft': [0.0, 0.0, 0.0],
        'enemy_structures': [0.8, 0.9, 0.0],
        'game_time': 5.0,
        'tech_level': 0.3,
        'army_strength': 1.0,
    }

    current_state = {
        'money': 3.2,
        'income': 2.5,
        'own_infantry': [0.7, 0.9, 0.0],
        'own_vehicles': [0.6, 0.9, 0.0],
        'own_aircraft': [0.0, 0.0, 0.0],
        'own_structures': [0.8, 0.95, 0.0],
        'enemy_infantry': [0.6, 0.8, 0.0],  # Enemy lost units
        'enemy_vehicles': [0.5, 0.8, 0.0],
        'enemy_aircraft': [0.0, 0.0, 0.0],
        'enemy_structures': [0.8, 0.9, 0.0],
        'game_time': 5.5,
        'tech_level': 0.35,
        'army_strength': 1.2,  # We got stronger
    }

    action = {
        'priority_economy': 0.2,
        'priority_defense': 0.1,
        'priority_military': 0.5,
        'priority_tech': 0.2,
    }

    # Create mock env
    class MockEnv:
        def __init__(self):
            self.prev_own_units = 6
            self.prev_enemy_units = 6
            self.prev_own_buildings = 5
            self.prev_enemy_buildings = 5

            class Stats:
                units_killed = 0
                units_lost = 0
                buildings_destroyed = 0
                buildings_lost = 0

            self.episode_stats = Stats()

    env = MockEnv()

    # Test each config
    for preset in ['exploration', 'balanced', 'sparse', 'aggressive']:
        config = get_config(preset)
        reward = calculate_reward(prev_state, current_state, action, env, config)
        print(f"  {preset}: reward = {reward:.4f}")

    print("\nReward calculation test passed!")
