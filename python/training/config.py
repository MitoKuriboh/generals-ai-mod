"""
Central Configuration for Generals Zero Hour Learning AI

Single source of truth for all training parameters.
Import from here to ensure consistency across modules.
"""

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
# Paths (Windows execution environment)
# =============================================================================
PIPE_NAME = r'\\.\pipe\generals_ml_bridge'
CHECKPOINT_DIR = r'C:\Users\Public\game-ai-agent\checkpoints\unified'
LOG_DIR = r'C:\Users\Public\game-ai-agent\runs\unified_training'
ML_LOG_PATH = r'C:\Users\Public\ml_decisions.log'

# =============================================================================
# Reward Configuration
# =============================================================================
WIN_REWARD = 100.0
LOSS_REWARD = -100.0
DRAW_REWARD = 0.0
REWARD_CLIP = 100.0       # Normalizer clips at this value (matches terminal rewards)
