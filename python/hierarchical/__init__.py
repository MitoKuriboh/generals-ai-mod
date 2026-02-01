"""
Hierarchical Coordinator for C&C Generals Zero Hour

Orchestrates the three-layer hierarchy:
- Strategic: Macro decisions (build priorities, aggression)
- Tactical: Team-level decisions (where to send teams)
- Micro: Unit-level decisions (kiting, focus fire)

This module provides:
- Multi-layer inference with proper latency budgeting
- Batched communication protocol for efficiency
- Joint training coordination
"""

from .coordinator import HierarchicalCoordinator
from .batch_bridge import BatchedMLBridge
from .sim_env import SimulatedHierarchicalEnv

__all__ = [
    'HierarchicalCoordinator',
    'BatchedMLBridge',
    'SimulatedHierarchicalEnv',
]
