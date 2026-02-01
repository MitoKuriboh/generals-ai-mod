"""
Central Configuration for Generals Zero Hour Learning AI

Single source of truth for all training parameters.
Import from here to ensure consistency across modules.
"""

import os
from pathlib import Path

# =============================================================================
# Protocol Version (MUST match C++ ML_PROTOCOL_VERSION in MLBridge.cpp)
# =============================================================================
PROTOCOL_VERSION = 2  # Increment when state/recommendation format changes

# =============================================================================
# Dimensions (MUST match C++ MLGameState/MLRecommendation structs)
# =============================================================================
STATE_DIM = 44   # Features from game state
ACTION_DIM = 8   # Continuous action outputs

# =============================================================================
# PPO Hyperparameters
# =============================================================================
LEARNING_RATE = 3e-4
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # GAE lambda for advantage estimation
CLIP_EPSILON = 0.2        # PPO clipping range
ENTROPY_COEF = 0.01       # Initial entropy coefficient
ENTROPY_MIN = 0.001       # Minimum entropy coefficient
ENTROPY_DECAY = 0.995     # Entropy decay per episode (faster decay)
VALUE_COEF = 0.5          # Value loss coefficient
MAX_GRAD_NORM = 0.5       # Gradient clipping

# =============================================================================
# Training Parameters
# =============================================================================
PPO_EPOCHS = 4            # PPO update epochs per batch
BATCH_SIZE = 64           # Mini-batch size for updates
MAX_EPISODE_STEPS = 3000  # Max steps before truncation (~100 min game time)
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N episodes

# =============================================================================
# Network Architecture
# =============================================================================
HIDDEN_DIM = 256          # Hidden layer size (increased for RTS complexity)

# =============================================================================
# Paths (configurable via environment variables)
# =============================================================================
PIPE_NAME = r'\\.\pipe\generals_ml_bridge'

# Base directory: use GENERALS_AI_DIR env var or default
_DEFAULT_BASE = r'C:\Users\Public\game-ai-agent'
BASE_DIR = Path(os.environ.get('GENERALS_AI_DIR', _DEFAULT_BASE))

# Derived paths (auto-created if needed)
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'unified'
LOG_DIR = BASE_DIR / 'runs' / 'unified_training'
ML_LOG_PATH = BASE_DIR / 'ml_decisions.log'

# Ensure directories exist when module is imported
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Convert to strings for backward compatibility with code expecting str paths
CHECKPOINT_DIR_STR = str(CHECKPOINT_DIR)
LOG_DIR_STR = str(LOG_DIR)
ML_LOG_PATH_STR = str(ML_LOG_PATH)

# =============================================================================
# Reward Configuration
# =============================================================================
WIN_REWARD = 100.0
LOSS_REWARD = -100.0
DRAW_REWARD = 0.0
REWARD_CLIP = 100.0       # Normalizer clips at this value (matches terminal rewards)
