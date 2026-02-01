"""
Tactical Reward Functions

Rewards for team-level decisions that align with strategic goals
while encouraging effective combat and unit preservation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from .model import TacticalAction


@dataclass
class TacticalRewardConfig:
    """Configuration for tactical rewards."""

    # Goal alignment with strategic layer
    goal_alignment_weight: float = 0.5

    # Combat efficiency (damage dealt vs taken)
    combat_efficiency_weight: float = 0.3

    # Unit preservation
    unit_preservation_weight: float = 0.4

    # Objective progress
    objective_progress_weight: float = 0.5

    # Penalties
    indecision_penalty: float = -0.1      # Same action repeatedly without progress
    friendly_fire_penalty: float = -0.5   # Attacking allies

    # Terminal rewards
    team_destroyed_penalty: float = -10.0
    objective_complete_bonus: float = 5.0


def tactical_reward(
    state: Dict,
    action: int,
    next_state: Dict,
    strategic_goals: Dict,
    config: Optional[TacticalRewardConfig] = None
) -> float:
    """
    Calculate reward for a tactical decision.

    Args:
        state: Current tactical state
        action: Action taken (TacticalAction enum value)
        next_state: Resulting tactical state
        strategic_goals: Current goals from strategic layer
        config: Reward configuration

    Returns:
        Scalar reward value
    """
    if config is None:
        config = TacticalRewardConfig()

    reward = 0.0

    # 1. Goal alignment with strategic layer
    reward += config.goal_alignment_weight * _goal_alignment_reward(
        action, strategic_goals
    )

    # 2. Combat efficiency (trade ratio)
    reward += config.combat_efficiency_weight * _combat_efficiency_reward(
        state, next_state
    )

    # 3. Unit preservation
    reward += config.unit_preservation_weight * _unit_preservation_reward(
        state, next_state
    )

    # 4. Objective progress
    reward += config.objective_progress_weight * _objective_progress_reward(
        state, next_state
    )

    # 5. Terminal conditions
    reward += _terminal_rewards(state, next_state, config)

    return reward


def _goal_alignment_reward(action: int, strategic_goals: Dict) -> float:
    """
    Reward for taking actions aligned with strategic layer goals.

    High aggression -> prefer offensive actions
    Low aggression -> prefer defensive actions
    """
    aggression = strategic_goals.get('aggression', 0.5)

    offensive_actions = {
        TacticalAction.ATTACK_MOVE,
        TacticalAction.ATTACK_TARGET,
        TacticalAction.HUNT,
    }

    defensive_actions = {
        TacticalAction.DEFEND_POSITION,
        TacticalAction.HOLD,
        TacticalAction.RETREAT,
    }

    if aggression > 0.7:
        # High aggression: reward offensive
        if action in offensive_actions:
            return 0.5
        elif action in defensive_actions:
            return -0.2
    elif aggression < 0.3:
        # Low aggression: reward defensive
        if action in defensive_actions:
            return 0.5
        elif action in offensive_actions:
            return -0.2
    else:
        # Balanced: slight preference for holding ground
        if action == TacticalAction.HOLD or action == TacticalAction.DEFEND_POSITION:
            return 0.2

    return 0.0


def _combat_efficiency_reward(state: Dict, next_state: Dict) -> float:
    """
    Reward for favorable damage trades.

    Trade ratio = damage dealt / damage taken
    Positive reward for ratio > 1, negative for ratio < 1
    """
    damage_dealt = next_state.get('damage_dealt', 0.0)
    damage_taken = next_state.get('damage_taken', 0.0)

    if damage_taken > 0:
        trade_ratio = damage_dealt / damage_taken
        # Normalize: ratio of 2.0 -> reward of 0.2
        return 0.2 * (trade_ratio - 1.0)
    elif damage_dealt > 0:
        # Dealt damage without taking any - bonus
        return 0.3

    return 0.0


def _unit_preservation_reward(state: Dict, next_state: Dict) -> float:
    """
    Penalty for losing units.

    Scaled by unit value - losing expensive units is worse.
    """
    # Count units in each category
    def total_units(s: Dict) -> float:
        composition = s.get('composition', {})
        return (
            composition.get('infantry_count', 0) +
            composition.get('vehicle_count', 0) * 2 +  # Vehicles worth more
            composition.get('aircraft_count', 0) * 3   # Aircraft worth most
        )

    current_units = total_units(state)
    next_units = total_units(next_state)

    if current_units > 0:
        loss_ratio = (current_units - next_units) / current_units
        return -loss_ratio  # Negative reward for losses

    return 0.0


def _objective_progress_reward(state: Dict, next_state: Dict) -> float:
    """
    Reward for making progress toward objectives.
    """
    current_progress = state.get('objective', {}).get('progress', 0.0)
    next_progress = next_state.get('objective', {}).get('progress', 0.0)

    progress_delta = next_progress - current_progress

    # Bonus for objective completion
    if next_progress >= 1.0 and current_progress < 1.0:
        return 1.0 + progress_delta

    return progress_delta


def _terminal_rewards(state: Dict, next_state: Dict, config: TacticalRewardConfig) -> float:
    """
    Terminal rewards for team destruction or objective completion.
    """
    reward = 0.0

    # Team destroyed
    def is_team_destroyed(s: Dict) -> bool:
        composition = s.get('composition', {})
        total = (
            composition.get('infantry_count', 0) +
            composition.get('vehicle_count', 0) +
            composition.get('aircraft_count', 0)
        )
        return total == 0

    if is_team_destroyed(next_state) and not is_team_destroyed(state):
        reward += config.team_destroyed_penalty

    # Objective complete
    current_progress = state.get('objective', {}).get('progress', 0.0)
    next_progress = next_state.get('objective', {}).get('progress', 0.0)

    if next_progress >= 1.0 and current_progress < 1.0:
        reward += config.objective_complete_bonus

    return reward


def intrinsic_reward(state: Dict, action: int, next_state: Dict) -> float:
    """
    Intrinsic rewards for exploration and learning.

    Used during early training to encourage exploration.
    """
    reward = 0.0

    # Curiosity: reward novel situations
    threat_change = abs(
        next_state.get('situational', {}).get('threat_level', 0) -
        state.get('situational', {}).get('threat_level', 0)
    )
    reward += 0.1 * threat_change

    # Progress: reward movement toward objectives
    dist_change = (
        state.get('status', {}).get('dist_to_objective', 1.0) -
        next_state.get('status', {}).get('dist_to_objective', 1.0)
    )
    reward += 0.2 * max(0, dist_change)  # Only reward getting closer

    return reward


class TacticalRewardTracker:
    """Track tactical rewards across an episode."""

    def __init__(self, config: Optional[TacticalRewardConfig] = None):
        self.config = config or TacticalRewardConfig()
        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.goal_alignment_total = 0.0
        self.combat_efficiency_total = 0.0
        self.unit_preservation_total = 0.0
        self.objective_progress_total = 0.0
        self.steps = 0

    def add_step(self, state: Dict, action: int, next_state: Dict,
                 strategic_goals: Dict) -> float:
        """Record a step and return the reward."""

        goal_rew = self.config.goal_alignment_weight * _goal_alignment_reward(
            action, strategic_goals
        )
        combat_rew = self.config.combat_efficiency_weight * _combat_efficiency_reward(
            state, next_state
        )
        preserve_rew = self.config.unit_preservation_weight * _unit_preservation_reward(
            state, next_state
        )
        objective_rew = self.config.objective_progress_weight * _objective_progress_reward(
            state, next_state
        )
        terminal_rew = _terminal_rewards(state, next_state, self.config)

        total = goal_rew + combat_rew + preserve_rew + objective_rew + terminal_rew

        self.goal_alignment_total += goal_rew
        self.combat_efficiency_total += combat_rew
        self.unit_preservation_total += preserve_rew
        self.objective_progress_total += objective_rew
        self.total_reward += total
        self.steps += 1

        return total

    def get_stats(self) -> Dict[str, float]:
        """Get reward statistics."""
        return {
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.steps),
            'goal_alignment': self.goal_alignment_total,
            'combat_efficiency': self.combat_efficiency_total,
            'unit_preservation': self.unit_preservation_total,
            'objective_progress': self.objective_progress_total,
            'steps': self.steps,
        }


if __name__ == '__main__':
    print("Testing tactical rewards...")

    config = TacticalRewardConfig()

    # Test goal alignment
    high_aggression = {'aggression': 0.9}
    low_aggression = {'aggression': 0.1}

    # Offensive action with high aggression should be positive
    rew = _goal_alignment_reward(TacticalAction.ATTACK_MOVE, high_aggression)
    print(f"ATTACK_MOVE with high aggression: {rew:.2f}")
    assert rew > 0

    # Defensive action with low aggression should be positive
    rew = _goal_alignment_reward(TacticalAction.DEFEND_POSITION, low_aggression)
    print(f"DEFEND_POSITION with low aggression: {rew:.2f}")
    assert rew > 0

    # Test combat efficiency
    state = {}
    next_state = {'damage_dealt': 100, 'damage_taken': 50}
    rew = _combat_efficiency_reward(state, next_state)
    print(f"Combat efficiency (2:1 ratio): {rew:.2f}")
    assert rew > 0

    # Test full reward
    state = {
        'composition': {'infantry_count': 10, 'vehicle_count': 5},
        'objective': {'progress': 0.5},
        'status': {'dist_to_objective': 0.8},
        'situational': {'threat_level': 0.3},
    }
    next_state = {
        'composition': {'infantry_count': 10, 'vehicle_count': 5},
        'objective': {'progress': 0.6},
        'damage_dealt': 50,
        'damage_taken': 30,
        'status': {'dist_to_objective': 0.6},
        'situational': {'threat_level': 0.4},
    }

    total_reward = tactical_reward(
        state, TacticalAction.ATTACK_MOVE, next_state, high_aggression, config
    )
    print(f"Total tactical reward: {total_reward:.3f}")

    # Test reward tracker
    tracker = TacticalRewardTracker(config)
    for _ in range(10):
        tracker.add_step(state, TacticalAction.ATTACK_MOVE, next_state, high_aggression)

    stats = tracker.get_stats()
    print(f"\nReward tracker stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}")

    print("\nTactical rewards test passed!")
