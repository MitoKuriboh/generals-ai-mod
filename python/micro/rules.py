"""
Rule-Based Micro Controller

A hand-crafted policy that demonstrates good micro behavior.
Used for imitation learning to bootstrap the MicroNetwork.

This implements common RTS micro patterns:
- Kiting: Attack then retreat when enemy approaches
- Focus fire: Prioritize low-health enemies
- Ability usage: Use abilities at optimal times
- Retreat: Fall back when health is critical
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .model import MicroAction
from .state import MicroState


class RuleBasedMicro:
    """
    Expert policy for micro control using hand-crafted rules.

    This policy demonstrates good micro behavior that can be
    cloned by the neural network through imitation learning.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

        # Configurable thresholds
        self.critical_health = 0.2     # Retreat if below this
        self.low_health = 0.4          # Start kiting if below this
        self.kite_range = 0.3          # Enemy distance at which to kite
        self.safe_range = 0.6          # Distance considered safe
        self.focus_threshold = 0.3     # Attack weakest if enemy health below this

    def get_action(self, state: MicroState) -> Tuple[int, float, float]:
        """
        Get action from rule-based policy.

        Args:
            state: MicroState object

        Returns:
            action: MicroAction enum value
            move_angle: Movement angle (-pi to pi)
            move_distance: Movement distance (0 to 1)
        """
        # Priority 1: Use ability if ready and in combat
        if state.ability_ready > 0.5 and state.under_fire > 0.5:
            return MicroAction.USE_ABILITY, 0.0, 0.0

        # Priority 2: Retreat if health critical
        if state.health < self.critical_health:
            if state.can_retreat > 0.5:
                # Move directly away from enemy
                retreat_angle = -state.nearest_enemy_angle * np.pi
                return MicroAction.RETREAT, retreat_angle, 0.8
            else:
                # Can't retreat, fight to the death
                return MicroAction.ATTACK_CURRENT, 0.0, 0.0

        # Priority 3: Kite if low health and enemy close
        # Kite immediately when ranged - don't wait to be shot first (BUG #4 fix)
        if state.health < self.low_health and state.nearest_enemy_dist < self.kite_range:
            if state.attack_range > 0.5:  # Ranged units kite immediately
                return self._kiting_action(state)

        # Priority 4: Focus fire on weak enemies
        if state.nearest_enemy_health < self.focus_threshold:
            return MicroAction.ATTACK_WEAKEST, 0.0, 0.0

        # Priority 5: Under fire - decide between attack and kite
        if state.under_fire > 0.5:
            # If we have range advantage, kite (H1 fix: aligned threshold to 0.5)
            if state.attack_range > 0.5 and state.nearest_enemy_dist < self.kite_range:
                return self._kiting_action(state)
            # Otherwise, stand and fight
            return MicroAction.ATTACK_CURRENT, 0.0, 0.0

        # Priority 6: Move forward to engage if enemy far
        if state.nearest_enemy_dist > self.safe_range:
            move_angle = state.nearest_enemy_angle * np.pi
            return MicroAction.MOVE_FORWARD, move_angle, 0.5

        # Default: Attack current target
        return MicroAction.ATTACK_CURRENT, 0.0, 0.0

    def _kiting_action(self, state: MicroState) -> Tuple[int, float, float]:
        """
        Execute a kiting maneuver.
        """
        # Decide between backward and flank based on nearby allies
        if state.nearest_ally_dist < 0.3:
            # Ally nearby - flank to avoid collision
            action = MicroAction.MOVE_FLANK
            # Move perpendicular to enemy
            flank_direction = 1.0 if self.rng.random() > 0.5 else -1.0
            move_angle = (state.nearest_enemy_angle + flank_direction * 0.5) * np.pi
        else:
            # No ally - move directly backward
            action = MicroAction.MOVE_BACKWARD
            move_angle = -state.nearest_enemy_angle * np.pi

        # Move distance based on enemy threat
        move_distance = 0.3 + 0.3 * state.nearest_enemy_threat

        return action, move_angle, move_distance

    def get_action_dict(self, state: MicroState) -> Dict:
        """
        Get action as a dictionary (matches MicroNetwork output format).
        """
        action, angle, distance = self.get_action(state)
        return {
            'action': action,
            'move_angle': angle,
            'move_distance': distance,
        }


class KitingExpert(RuleBasedMicro):
    """
    Expert that specializes in kiting behavior.
    More aggressive with kiting, less with engagement.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.critical_health = 0.3
        self.low_health = 0.6        # Start kiting earlier
        self.kite_range = 0.4        # Kite at longer range
        self.safe_range = 0.7

    def get_action(self, state: MicroState) -> Tuple[int, float, float]:
        # Always kite if enemy is close, regardless of health
        if state.nearest_enemy_dist < self.kite_range and state.attack_range > 0.5:
            return self._kiting_action(state)

        return super().get_action(state)


class AggressiveExpert(RuleBasedMicro):
    """
    Expert that specializes in aggressive play.
    Focus on damage output over survival.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.critical_health = 0.1   # Very low retreat threshold
        self.low_health = 0.25
        self.focus_threshold = 0.5   # More aggressive focus fire

    def get_action(self, state: MicroState) -> Tuple[int, float, float]:
        # Use ability more aggressively
        if state.ability_ready > 0.5:
            return MicroAction.USE_ABILITY, 0.0, 0.0

        # Focus fire on anything moderately damaged
        if state.nearest_enemy_health < self.focus_threshold:
            return MicroAction.ATTACK_WEAKEST, 0.0, 0.0

        # Hunt down enemies
        if state.nearest_enemy_dist > 0.3:
            angle = state.nearest_enemy_angle * np.pi
            return MicroAction.HUNT, angle, 0.7

        return super().get_action(state)


def collect_expert_demonstrations(
    expert: RuleBasedMicro,
    num_samples: int = 1000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect demonstration data from expert policy.

    Args:
        expert: Rule-based expert policy
        num_samples: Number of samples to collect
        seed: Random seed

    Returns:
        states: [num_samples, 32] state vectors
        actions: [num_samples] discrete actions
        angles: [num_samples] movement angles
        distances: [num_samples] movement distances
    """
    rng = np.random.default_rng(seed)

    states = []
    actions = []
    angles = []
    distances = []

    for _ in range(num_samples):
        # Generate random state
        state = _generate_random_state(rng)
        state_tensor = state.to_tensor().numpy()

        # Get expert action
        action, angle, distance = expert.get_action(state)

        states.append(state_tensor)
        actions.append(action)
        angles.append((angle / np.pi + 1) / 2)  # Normalize to [0, 1]
        distances.append(distance)

    return (
        np.array(states),
        np.array(actions),
        np.array(angles),
        np.array(distances),
    )


def _generate_random_state(rng: np.random.Generator) -> MicroState:
    """Generate a random but plausible micro state for demonstration."""
    # Combat scenarios
    in_combat = rng.random() > 0.3

    return MicroState(
        # Identity
        unit_type=rng.choice([0.2, 0.4, 0.5, 0.6]),
        is_hero=1.0 if rng.random() > 0.9 else 0.0,
        veterancy=rng.uniform(0, 1),
        has_ability=1.0 if rng.random() > 0.5 else 0.0,

        # Status
        health=rng.uniform(0.1, 1.0),
        shield=rng.uniform(0, 0.5) if rng.random() > 0.7 else 0.0,
        ammunition=rng.uniform(0.3, 1.0),
        cooldown=rng.uniform(0, 0.5),
        speed=rng.uniform(0.3, 0.7),
        attack_range=rng.uniform(0.3, 0.8),
        dps=rng.uniform(0.2, 0.8),
        armor=rng.uniform(0, 0.5),

        # Situational
        nearest_enemy_dist=rng.uniform(0.1, 0.8) if in_combat else rng.uniform(0.5, 1.0),
        nearest_enemy_angle=rng.uniform(-1, 1),
        nearest_enemy_health=rng.uniform(0.1, 1.0),
        nearest_enemy_threat=rng.uniform(0.2, 0.9) if in_combat else rng.uniform(0, 0.3),
        nearest_ally_dist=rng.uniform(0.1, 0.8),
        in_cover=1.0 if rng.random() > 0.8 else 0.0,
        under_fire=1.0 if in_combat and rng.random() > 0.3 else 0.0,
        ability_ready=1.0 if rng.random() > 0.6 else 0.0,
        target_dist=rng.uniform(0.1, 0.6) if in_combat else 1.0,
        target_health=rng.uniform(0.1, 1.0),
        target_type=rng.choice([0.2, 0.4, 0.5, 0.6]),
        can_retreat=1.0 if rng.random() > 0.2 else 0.0,

        # Team context
        objective_type=rng.choice([0.25, 0.5, 0.75]),
        objective_dir=rng.uniform(-1, 1),
        team_role=rng.choice([0.0, 0.33, 0.67]),
        priority=rng.uniform(0.3, 0.9),

        # Temporal
        time_since_hit=rng.uniform(0, 1) if rng.random() > 0.5 else 0.0,
        time_since_shot=rng.uniform(0, 0.5),
        time_in_combat=rng.uniform(0, 0.5) if in_combat else 0.0,
        movement_history=rng.choice([0.0, 0.5, 1.0]),
    )


if __name__ == '__main__':
    print("Testing rule-based micro...")

    expert = RuleBasedMicro(seed=42)

    # Test various scenarios
    scenarios = [
        ("Low health, under fire", MicroState(
            unit_type=0.4, is_hero=0.0, veterancy=0.5, has_ability=0.0,
            health=0.15, shield=0.0, ammunition=0.5, cooldown=0.0,
            speed=0.5, attack_range=0.5, dps=0.5, armor=0.3,
            nearest_enemy_dist=0.2, nearest_enemy_angle=0.5,
            nearest_enemy_health=0.7, nearest_enemy_threat=0.8,
            nearest_ally_dist=0.5, in_cover=0.0, under_fire=1.0,
            ability_ready=0.0, target_dist=0.2, target_health=0.7,
            target_type=0.4, can_retreat=1.0,
            objective_type=0.25, objective_dir=0.3, team_role=0.33, priority=0.7,
            time_since_hit=0.1, time_since_shot=0.2, time_in_combat=0.3, movement_history=0.5,
        )),
        ("Ability ready, in combat", MicroState(
            unit_type=0.4, is_hero=0.0, veterancy=0.5, has_ability=1.0,
            health=0.7, shield=0.0, ammunition=0.8, cooldown=0.0,
            speed=0.5, attack_range=0.5, dps=0.5, armor=0.3,
            nearest_enemy_dist=0.3, nearest_enemy_angle=-0.2,
            nearest_enemy_health=0.5, nearest_enemy_threat=0.5,
            nearest_ally_dist=0.4, in_cover=0.0, under_fire=1.0,
            ability_ready=1.0, target_dist=0.3, target_health=0.5,
            target_type=0.5, can_retreat=1.0,
            objective_type=0.25, objective_dir=0.1, team_role=0.33, priority=0.8,
            time_since_hit=0.2, time_since_shot=0.1, time_in_combat=0.4, movement_history=0.0,
        )),
        ("Enemy low health", MicroState(
            unit_type=0.4, is_hero=0.0, veterancy=0.3, has_ability=0.0,
            health=0.8, shield=0.0, ammunition=0.9, cooldown=0.0,
            speed=0.5, attack_range=0.6, dps=0.6, armor=0.2,
            nearest_enemy_dist=0.25, nearest_enemy_angle=0.1,
            nearest_enemy_health=0.2, nearest_enemy_threat=0.3,
            nearest_ally_dist=0.3, in_cover=0.0, under_fire=0.0,
            ability_ready=0.0, target_dist=0.25, target_health=0.2,
            target_type=0.4, can_retreat=1.0,
            objective_type=0.25, objective_dir=0.1, team_role=0.33, priority=0.6,
            time_since_hit=0.5, time_since_shot=0.1, time_in_combat=0.2, movement_history=0.5,
        )),
    ]

    for name, state in scenarios:
        action, angle, dist = expert.get_action(state)
        print(f"\n{name}:")
        print(f"  Action: {MicroAction.name(action)}")
        print(f"  Move: angle={angle:.2f}, dist={dist:.2f}")

    # Test demonstration collection
    print("\n\nCollecting expert demonstrations...")
    states, actions, angles, distances = collect_expert_demonstrations(
        expert, num_samples=100, seed=42
    )

    print(f"Collected {len(states)} demonstrations")
    print(f"Action distribution:")
    for i in range(11):
        count = np.sum(actions == i)
        print(f"  {MicroAction.name(i)}: {count}")

    print("\nRule-based micro test passed!")
