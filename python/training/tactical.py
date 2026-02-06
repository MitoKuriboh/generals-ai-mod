"""
Tactical Layer - Team-level decision making for C&C Generals AI

Actions:
    0: ATTACK_MOVE     - Attack-move to position
    1: ATTACK_TARGET   - Focus on specific target
    2: DEFEND_POSITION - Guard location
    3: RETREAT         - Fall back to base
    4: HOLD            - Hold position
    5: HUNT            - Seek and destroy
    6: REINFORCE       - Merge with another team
    7: SPECIAL         - Use special ability
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


# Tactical action constants (match C++ TacticalActionType)
TACTICAL_ATTACK_MOVE = 0
TACTICAL_ATTACK_TARGET = 1
TACTICAL_DEFEND_POSITION = 2
TACTICAL_RETREAT = 3
TACTICAL_HOLD = 4
TACTICAL_HUNT = 5
TACTICAL_REINFORCE = 6
TACTICAL_SPECIAL = 7


@dataclass
class TacticalState:
    """Team tactical state from C++ (64 floats)."""
    team_id: int

    # Strategic embedding from higher layer (8)
    strategy: np.ndarray  # [eco, def, mil, tech, inf, veh, air, aggression]

    # Team composition (12) - [count, health, ready] x 4
    infantry: np.ndarray
    vehicles: np.ndarray
    aircraft: np.ndarray
    mixed: np.ndarray

    # Team status (8)
    team_health: float
    ammunition: float
    cohesion: float
    experience: float
    dist_to_objective: float
    dist_to_base: float
    under_fire: float
    has_transport: float

    # Situational (16)
    nearby_enemies: np.ndarray  # 4 quadrants
    nearby_allies: np.ndarray   # 4 quadrants
    terrain_advantage: float
    threat_level: float
    target_value: float
    supply_dist: float
    retreat_path: float
    reinforce_possible: float
    special_ready: float

    # Current objective (8)
    objective_type: float
    objective_x: float
    objective_y: float
    priority: float
    progress: float
    time_on_objective: float

    # Temporal (4)
    time_since_engagement: float
    time_since_command: float
    frames_since_spawn: float


@dataclass
class TacticalCommand:
    """Command for a team."""
    team_id: int
    action: int
    target_x: float
    target_y: float
    attitude: float  # 0=passive, 1=aggressive


def parse_tactical_state(team_id: int, state_array: List[float]) -> TacticalState:
    """Parse 64-float array into TacticalState."""
    s = np.array(state_array)
    return TacticalState(
        team_id=team_id,
        strategy=s[0:8],
        infantry=s[8:11],
        vehicles=s[11:14],
        aircraft=s[14:17],
        mixed=s[17:20],
        team_health=s[20],
        ammunition=s[21],
        cohesion=s[22],
        experience=s[23],
        dist_to_objective=s[24],
        dist_to_base=s[25],
        under_fire=s[26],
        has_transport=s[27],
        nearby_enemies=s[28:32],
        nearby_allies=s[32:36],
        terrain_advantage=s[36],
        threat_level=s[37],
        target_value=s[38],
        supply_dist=s[39],
        retreat_path=s[40],
        reinforce_possible=s[41],
        special_ready=s[42],
        objective_type=s[44],
        objective_x=s[45],
        objective_y=s[46],
        priority=s[47],
        progress=s[48],
        time_on_objective=s[49],
        time_since_engagement=s[52],
        time_since_command=s[53],
        frames_since_spawn=s[54],
    )


class TacticalDecisionMaker:
    """
    Rule-based tactical decision maker.

    Makes team-level decisions that ADAPT to the current situation.
    This is what makes the AI actually responsive during gameplay.
    """

    # Thresholds for state-based decisions (H4 fix)
    LOW_AMMO_THRESHOLD = 0.3       # Retreat to resupply when below
    LOW_COHESION_THRESHOLD = 0.4   # Regroup before attacking when below
    HIGH_TERRAIN_THRESHOLD = 0.7   # Hold position when above
    LOW_SUPPLY_DIST = 0.3          # Resupply accessible when below
    MIN_PROGRESS_THRESHOLD = 0.2   # Good progress on objective
    TARGET_VALUE_HIGH = 0.7        # Prioritize high-value targets

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def decide(self, state: TacticalState) -> TacticalCommand:
        """
        Decide tactical action for a team based on current situation.

        Key decision factors:
        - Team health and status
        - Threat level and enemy positions
        - Strategic layer aggression
        - Current objective
        - Ammunition and supply status (H4 fix)
        - Team cohesion (H4 fix)
        - Terrain advantage (H4 fix)
        - Target value (H4 fix)
        """
        aggression = state.strategy[7] if len(state.strategy) > 7 else 0.5

        # Pre-compute commonly used conditions
        enemies_around = np.sum(state.nearby_enemies > 0.3)
        allies_around = np.sum(state.nearby_allies > 0.3)

        # === CRITICAL SITUATIONS ===

        # 1. Team critically damaged and under fire -> RETREAT
        if state.team_health < 0.3 and state.under_fire > 0.5:
            if state.retreat_path > 0.3:  # Path clear enough
                return self._retreat(state, "Critical health + under fire")

        # 2. High threat, low health -> RETREAT
        if state.threat_level > 0.7 and state.team_health < 0.5:
            if state.retreat_path > 0.4:
                return self._retreat(state, "High threat + low health")

        # 3. Overwhelmed (enemies in multiple quadrants, few allies) -> RETREAT
        if enemies_around >= 3 and allies_around < 2 and state.team_health < 0.6:
            return self._retreat(state, "Surrounded")

        # === H4 FIX: AMMUNITION & SUPPLY AWARENESS ===

        # 4. Low ammunition -> retreat to resupply
        if state.ammunition < self.LOW_AMMO_THRESHOLD:
            if state.supply_dist < self.LOW_SUPPLY_DIST and state.retreat_path > 0.3:
                return self._retreat(state, "Low ammo, resupplying")
            # If supply is far, continue fighting until critical
            elif state.ammunition < 0.1:
                return self._retreat(state, "Critically low ammo")

        # === H4 FIX: COHESION CHECK ===

        # 5. Low cohesion -> regroup before attacking
        if state.cohesion < self.LOW_COHESION_THRESHOLD:
            # Team is too spread out - hold position to regroup
            if aggression > 0.5 and state.team_health > 0.5:
                return self._hold(state, "Regrouping (low cohesion)")

        # === SPECIAL ABILITY ===

        # 6. Special ready and good opportunity
        if state.special_ready > 0.8 and state.threat_level > 0.4:
            return self._special(state, "Special ability ready")

        # === H4 FIX: TERRAIN ADVANTAGE ===

        # 7. High terrain advantage -> hold position to exploit it
        if state.terrain_advantage > self.HIGH_TERRAIN_THRESHOLD:
            if max(state.nearby_enemies) > 0.2 and state.team_health > 0.5:
                return self._defend(state, "Holding terrain advantage")
            elif not state.under_fire and aggression < 0.7:
                return self._hold(state, "Holding high ground")

        # === DEFENSIVE SITUATIONS ===

        # 8. Low aggression strategy -> prefer defensive actions
        if aggression < 0.3:
            if state.under_fire > 0.5:
                return self._defend(state, "Defensive stance + under attack")
            if state.dist_to_base < 0.3:
                return self._hold(state, "Defensive stance near base")

        # 9. Enemies nearby but objective is defense
        if state.objective_type < 0.3 and max(state.nearby_enemies) > 0.5:
            return self._defend(state, "Defense objective with enemies")

        # === H4 FIX: TARGET VALUE PRIORITIZATION ===

        # 10. High-value target nearby -> prioritize attack
        if state.target_value > self.TARGET_VALUE_HIGH and state.team_health > 0.5:
            if max(state.nearby_enemies) > 0.2:
                return self._attack_target(state, "High-value target priority")

        # === OFFENSIVE SITUATIONS ===

        # 11. High aggression + healthy team -> ATTACK or HUNT
        if aggression > 0.7 and state.team_health > 0.7:
            if max(state.nearby_enemies) > 0.3:
                # Enemies visible, attack them
                return self._attack_move(state, "Aggressive + enemies visible")
            else:
                # No enemies visible, hunt for them
                return self._hunt(state, "Aggressive + seeking targets")

        # 12. Strong position (high terrain, good health, allies nearby)
        if state.terrain_advantage > 0.5 and state.team_health > 0.6 and allies_around >= 1:
            if max(state.nearby_enemies) > 0.2:
                return self._attack_target(state, "Strong position + target")

        # 13. Objective attack and making good progress (H4: use progress field)
        if state.objective_type > 0.5 and state.progress > self.MIN_PROGRESS_THRESHOLD:
            return self._attack_move(state, "Continuing attack objective")

        # === SUPPORT SITUATIONS ===

        # 14. Can reinforce nearby team
        if state.reinforce_possible > 0.7 and state.team_health < 0.5:
            return self._reinforce(state, "Low health, can merge")

        # === DEFAULT BEHAVIOR ===

        # Moderate aggression + adequate cohesion: attack-move toward objective
        if aggression > 0.5 and state.team_health > 0.5 and state.cohesion > 0.3:
            return self._attack_move(state, "Default offensive")

        # Low-moderate: hold and wait
        return self._hold(state, "Default hold")

    def _retreat(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: RETREAT ({reason})")
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_RETREAT,
            target_x=0.5,  # Base position (C++ will calculate)
            target_y=0.5,
            attitude=0.0   # Passive
        )

    def _defend(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: DEFEND ({reason})")
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_DEFEND_POSITION,
            target_x=state.objective_x,
            target_y=state.objective_y,
            attitude=0.5   # Balanced
        )

    def _hold(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: HOLD ({reason})")
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_HOLD,
            target_x=state.objective_x,
            target_y=state.objective_y,
            attitude=0.3
        )

    def _attack_move(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: ATTACK_MOVE ({reason})")
        # Find direction with most enemies
        enemy_dir = np.argmax(state.nearby_enemies)
        # Convert quadrant to approximate position offset
        offsets = [(0.1, 0.1), (0.1, -0.1), (-0.1, -0.1), (-0.1, 0.1)]  # NE, SE, SW, NW
        dx, dy = offsets[enemy_dir]
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_ATTACK_MOVE,
            target_x=min(1, max(0, state.objective_x + dx)),
            target_y=min(1, max(0, state.objective_y + dy)),
            attitude=0.8   # Aggressive
        )

    def _attack_target(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: ATTACK_TARGET ({reason})")
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_ATTACK_TARGET,
            target_x=state.objective_x,
            target_y=state.objective_y,
            attitude=1.0   # Full aggro
        )

    def _hunt(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: HUNT ({reason})")
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_HUNT,
            target_x=state.objective_x,
            target_y=state.objective_y,
            attitude=0.9
        )

    def _reinforce(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: REINFORCE ({reason})")
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_REINFORCE,
            target_x=0.5,
            target_y=0.5,
            attitude=0.2
        )

    def _special(self, state: TacticalState, reason: str) -> TacticalCommand:
        if self.verbose:
            print(f"  Team {state.team_id}: SPECIAL ({reason})")
        # Find enemy concentration for ability targeting
        enemy_dir = np.argmax(state.nearby_enemies)
        offsets = [(0.1, 0.1), (0.1, -0.1), (-0.1, -0.1), (-0.1, 0.1)]
        dx, dy = offsets[enemy_dir]
        return TacticalCommand(
            team_id=state.team_id,
            action=TACTICAL_SPECIAL,
            target_x=min(1, max(0, state.objective_x + dx)),
            target_y=min(1, max(0, state.objective_y + dy)),
            attitude=0.7
        )
