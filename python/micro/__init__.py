"""
Micro Layer for Hierarchical RL in C&C Generals Zero Hour

This module handles unit-level micro control:
- Kiting and positioning
- Focus fire and target selection
- Ability usage and retreat decisions

Input: 32-dim state (unit state + team objective)
Output: 11 discrete actions + 2 continuous params (move direction)
"""

from .model import MicroNetwork, MICRO_STATE_DIM, MICRO_ACTION_DIM
from .state import MicroState, build_micro_state
from .rewards import micro_reward, MicroRewardConfig
from .rules import RuleBasedMicro

__all__ = [
    'MicroNetwork',
    'MICRO_STATE_DIM',
    'MICRO_ACTION_DIM',
    'MicroState',
    'build_micro_state',
    'micro_reward',
    'MicroRewardConfig',
    'RuleBasedMicro',
]
