"""
Tactical Layer for Hierarchical RL in C&C Generals Zero Hour

This module handles team-level decision making:
- Where to send teams
- Attack/defend/retreat decisions
- Team coordination

Input: 64-dim state (team state + strategic goals)
Output: 8 discrete actions + 3 continuous params
"""

from .model import TacticalNetwork, TACTICAL_STATE_DIM, TACTICAL_ACTION_DIM
from .state import TacticalState, build_tactical_state
from .rewards import tactical_reward, TacticalRewardConfig

__all__ = [
    'TacticalNetwork',
    'TACTICAL_STATE_DIM',
    'TACTICAL_ACTION_DIM',
    'TacticalState',
    'build_tactical_state',
    'tactical_reward',
    'TacticalRewardConfig',
]
