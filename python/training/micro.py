"""
Micro Layer - Unit-level decision making for C&C Generals AI

Actions (matching C++ MicroActionType):
    0: ATTACK_CURRENT    - Continue current target
    1: ATTACK_NEAREST    - Switch to nearest enemy
    2: ATTACK_WEAKEST    - Focus weakest enemy
    3: ATTACK_PRIORITY   - High-value target
    4: MOVE_FORWARD      - Advance toward enemy
    5: MOVE_BACKWARD     - Kite backward
    6: MOVE_FLANK        - Circle strafe
    7: HOLD_FIRE         - Stealth/hold position
    8: USE_ABILITY       - Special power
    9: RETREAT           - Full retreat
   10: FOLLOW_TEAM       - Default team behavior
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import math


# Micro action constants (match C++ MicroActionType)
MICRO_ATTACK_CURRENT = 0
MICRO_ATTACK_NEAREST = 1
MICRO_ATTACK_WEAKEST = 2
MICRO_ATTACK_PRIORITY = 3
MICRO_MOVE_FORWARD = 4
MICRO_MOVE_BACKWARD = 5
MICRO_MOVE_FLANK = 6
MICRO_HOLD_FIRE = 7
MICRO_USE_ABILITY = 8
MICRO_RETREAT = 9
MICRO_FOLLOW_TEAM = 10

ACTION_NAMES = [
    'ATTACK_CURRENT', 'ATTACK_NEAREST', 'ATTACK_WEAKEST', 'ATTACK_PRIORITY',
    'MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_FLANK', 'HOLD_FIRE',
    'USE_ABILITY', 'RETREAT', 'FOLLOW_TEAM'
]


@dataclass
class MicroState:
    """Unit micro state from C++ (32 floats)."""
    unit_id: int

    # Unit identity (4)
    unit_type: float           # 0.1-0.95 encoded type
    is_hero: float             # 1 if hero
    veterancy: float           # 0-1
    has_ability: float         # 1 if has special

    # Unit status (8)
    health: float              # 0-1 ratio
    shield: float              # 0-1 ratio
    ammunition: float          # 0-1 clip ratio
    cooldown: float            # 0=ready, 1=reloading
    speed: float               # normalized
    attack_range: float        # normalized weapon range
    dps: float                 # normalized damage
    armor: float               # normalized

    # Situational (12)
    nearest_enemy_dist: float  # 0-1 normalized
    nearest_enemy_angle: float # -1 to 1 (from -pi to pi)
    nearest_enemy_health: float
    nearest_enemy_threat: float
    nearest_ally_dist: float
    in_cover: float
    under_fire: float
    ability_ready: float
    target_dist: float
    target_health: float
    target_type: float
    can_retreat: float

    # Team context (4)
    objective_type: float
    objective_dir: float
    team_role: float
    priority: float

    # Temporal (4)
    time_since_hit: float
    time_since_shot: float
    time_in_combat: float
    movement_history: float


@dataclass
class MicroCommand:
    """Command for a unit."""
    unit_id: int
    action: int
    move_angle: float  # radians, -pi to pi
    move_distance: float  # 0-1 normalized


def parse_micro_state(unit_id: int, state_array: List[float]) -> MicroState:
    """Parse 32-float array into MicroState."""
    s = np.array(state_array)
    return MicroState(
        unit_id=unit_id,
        # Unit identity (4)
        unit_type=s[0],
        is_hero=s[1],
        veterancy=s[2],
        has_ability=s[3],
        # Unit status (8)
        health=s[4],
        shield=s[5],
        ammunition=s[6],
        cooldown=s[7],
        speed=s[8],
        attack_range=s[9],
        dps=s[10],
        armor=s[11],
        # Situational (12)
        nearest_enemy_dist=s[12],
        nearest_enemy_angle=s[13],
        nearest_enemy_health=s[14],
        nearest_enemy_threat=s[15],
        nearest_ally_dist=s[16],
        in_cover=s[17],
        under_fire=s[18],
        ability_ready=s[19],
        target_dist=s[20],
        target_health=s[21],
        target_type=s[22],
        can_retreat=s[23],
        # Team context (4)
        objective_type=s[24],
        objective_dir=s[25],
        team_role=s[26],
        priority=s[27],
        # Temporal (4)
        time_since_hit=s[28],
        time_since_shot=s[29],
        time_in_combat=s[30],
        movement_history=s[31],
    )


class MicroDecisionMaker:
    """
    Rule-based micro decision maker.

    Makes unit-level decisions that adapt to the current combat situation.
    This is what enables smart micro behavior like kiting, focus fire, and retreating.

    Decision priority:
    1. Retreat if health critical
    2. Use ability if ready and in combat
    3. Kite if ranged + low health + enemy close
    4. Focus fire on weak enemies
    5. Attack priority targets (heroes)
    6. Default attack behavior
    """

    # Thresholds
    CRITICAL_HEALTH = 0.2   # Retreat if below
    LOW_HEALTH = 0.4        # Start kiting if below
    KITE_RANGE = 0.35       # Enemy distance threshold for kiting (normalized)
    FOCUS_THRESHOLD = 0.3   # Attack weakest if enemy health below this
    RANGED_THRESHOLD = 0.5  # Consider unit ranged if range > this
    COVER_THRESHOLD = 0.5   # Consider unit in cover if above this (H1 fix)
    CAN_RETREAT_THRESHOLD = 0.5  # Retreat path clear if above this (H2 fix)

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.rng = np.random.default_rng()

    def decide(self, state: MicroState) -> MicroCommand:
        """
        Decide micro action for a unit based on current situation.

        Key decision factors:
        - Unit health and status
        - Enemy proximity and health
        - Unit type (ranged vs melee)
        - Ability availability
        - Retreat path availability
        - Cover status (H1 fix)
        """
        # Pre-compute commonly used conditions
        is_ranged = state.attack_range > self.RANGED_THRESHOLD
        enemy_close = state.nearest_enemy_dist < self.KITE_RANGE
        in_cover = state.in_cover > self.COVER_THRESHOLD  # H1: Track cover status
        can_retreat = state.can_retreat > self.CAN_RETREAT_THRESHOLD  # H2: Track retreat path

        # === PRIORITY 1: Critical health - RETREAT ===
        if state.health < self.CRITICAL_HEALTH:
            if can_retreat:
                return self._retreat(state, "Critical health")
            else:
                # Can't retreat, fight to death
                return self._attack_current(state, "No retreat, fighting")

        # === PRIORITY 2: Use ability if ready and valuable ===
        if state.ability_ready > 0.5 and state.has_ability > 0.5:
            # Use ability if in combat (under fire or have target)
            if state.under_fire > 0.5 or state.target_dist < 0.8:
                return self._use_ability(state, "Ability ready in combat")

        # === PRIORITY 3: Cover-aware behavior (H1 fix) ===
        # If in good cover, adopt defensive stance - don't kite away from cover
        if in_cover:
            if state.under_fire > 0.5:
                # Under fire but in cover - hold position and fight
                return self._attack_current(state, "Holding cover position")
            elif state.health < self.LOW_HEALTH:
                # Low health but in cover - stay put, don't expose by kiting
                return self._attack_current(state, "Defending from cover (low HP)")

        # === PRIORITY 4: Kite if low health + ranged + enemy close (H2 fix) ===
        if state.health < self.LOW_HEALTH and is_ranged and enemy_close:
            # H2: Only kite if we have a clear retreat path
            if can_retreat:
                return self._kite(state, "Low health kiting")
            else:
                # No retreat path - stand and fight
                return self._attack_current(state, "No kite path, fighting")

        # === PRIORITY 5: Focus fire on weak enemies ===
        if state.nearest_enemy_health < self.FOCUS_THRESHOLD and state.nearest_enemy_health > 0:
            return self._attack_weakest(state, "Focus fire on weak")

        # === PRIORITY 6: Hero hunting ===
        # target_type > 0.2 and < 0.3 indicates hero (based on getUnitTypeEncoding)
        if state.target_type > 0.2 and state.target_type < 0.3:
            return self._attack_priority(state, "Hunting hero target")

        # === PRIORITY 7: Kite even at full health if ranged and enemy very close (H2 fix) ===
        if is_ranged and state.nearest_enemy_dist < 0.2 and state.under_fire > 0.5:
            # H2: Check can_retreat before range maintenance kiting
            if can_retreat and not in_cover:  # H1: Don't leave cover for range maintenance
                return self._kite(state, "Range maintenance kiting")
            elif in_cover:
                return self._attack_current(state, "Holding cover (enemy close)")

        # === PRIORITY 8: Advance if enemy far and we're attacking ===
        if state.target_dist > 0.7 and state.time_in_combat > 0.3:
            return self._move_forward(state, "Closing distance")

        # === DEFAULT: Attack current target or follow team ===
        if state.time_in_combat > 0.1:
            return self._attack_current(state, "Continuing attack")

        return self._follow_team(state, "Default behavior")

    def _retreat(self, state: MicroState, reason: str) -> MicroCommand:
        """Full retreat toward base."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: RETREAT ({reason})")
        # Retreat direction: opposite of enemy
        retreat_angle = -state.nearest_enemy_angle * math.pi
        return MicroCommand(
            unit_id=state.unit_id,
            action=MICRO_RETREAT,
            move_angle=retreat_angle,
            move_distance=0.8
        )

    def _use_ability(self, state: MicroState, reason: str) -> MicroCommand:
        """Use special ability."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: USE_ABILITY ({reason})")
        return MicroCommand(
            unit_id=state.unit_id,
            action=MICRO_USE_ABILITY,
            move_angle=0.0,
            move_distance=0.0
        )

    def _kite(self, state: MicroState, reason: str) -> MicroCommand:
        """Kite backward from enemy."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: MOVE_BACKWARD ({reason})")

        # Check for nearby ally to avoid collision
        if state.nearest_ally_dist < 0.3:
            # Flank instead of straight back
            flank_dir = 1.0 if self.rng.random() > 0.5 else -1.0
            move_angle = (-state.nearest_enemy_angle + flank_dir * 0.5) * math.pi
            action = MICRO_MOVE_FLANK
        else:
            # Straight backward
            move_angle = -state.nearest_enemy_angle * math.pi
            action = MICRO_MOVE_BACKWARD

        # Move distance based on threat
        move_dist = 0.3 + 0.3 * state.nearest_enemy_threat

        return MicroCommand(
            unit_id=state.unit_id,
            action=action,
            move_angle=move_angle,
            move_distance=min(1.0, move_dist)
        )

    def _attack_weakest(self, state: MicroState, reason: str) -> MicroCommand:
        """Focus fire on weakest enemy."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: ATTACK_WEAKEST ({reason})")
        return MicroCommand(
            unit_id=state.unit_id,
            action=MICRO_ATTACK_WEAKEST,
            move_angle=0.0,
            move_distance=0.0
        )

    def _attack_priority(self, state: MicroState, reason: str) -> MicroCommand:
        """Attack high-value target."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: ATTACK_PRIORITY ({reason})")
        return MicroCommand(
            unit_id=state.unit_id,
            action=MICRO_ATTACK_PRIORITY,
            move_angle=0.0,
            move_distance=0.0
        )

    def _attack_current(self, state: MicroState, reason: str) -> MicroCommand:
        """Continue attacking current target."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: ATTACK_CURRENT ({reason})")
        return MicroCommand(
            unit_id=state.unit_id,
            action=MICRO_ATTACK_CURRENT,
            move_angle=0.0,
            move_distance=0.0
        )

    def _move_forward(self, state: MicroState, reason: str) -> MicroCommand:
        """Move toward enemy/target."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: MOVE_FORWARD ({reason})")
        # Move toward nearest enemy
        move_angle = state.nearest_enemy_angle * math.pi
        return MicroCommand(
            unit_id=state.unit_id,
            action=MICRO_MOVE_FORWARD,
            move_angle=move_angle,
            move_distance=0.5
        )

    def _follow_team(self, state: MicroState, reason: str) -> MicroCommand:
        """Default team behavior."""
        if self.verbose:
            print(f"    Unit {state.unit_id}: FOLLOW_TEAM ({reason})")
        return MicroCommand(
            unit_id=state.unit_id,
            action=MICRO_FOLLOW_TEAM,
            move_angle=0.0,
            move_distance=0.0
        )
