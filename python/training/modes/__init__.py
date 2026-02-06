"""
Training mode implementations.

Available modes:
- ManualTrainer: Wait for user to start games manually
- AutoTrainer: Auto-launch game with -autoSkirmish
- SimulatedTrainer: Use SimulatedEnv for fast testing
"""

from .base import BaseTrainer
from .manual import ManualTrainer
from .auto import AutoTrainer
from .simulated import SimulatedTrainer

__all__ = ['BaseTrainer', 'ManualTrainer', 'AutoTrainer', 'SimulatedTrainer']
