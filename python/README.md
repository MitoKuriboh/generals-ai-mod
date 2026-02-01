# ML Bridge Python Server

Python server component for the Generals Zero Hour Learning AI.

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Components

### ML Bridge Server (Testing)

Rule-based server for testing the ML bridge communication:

```bash
# Basic usage
python ml_bridge_server.py

# With logging
python ml_bridge_server.py --log states.jsonl --verbose

# Observation only (no responses)
python ml_bridge_server.py --no-respond
```

### State Analysis

Analyze logged game states:

```bash
python analyze_states.py states.jsonl --summary --plot
```

### Training Infrastructure

Full PPO training system in `training/`:

```bash
# Test with simulated environment
python -m training.train --simulated --episodes 100

# Train with real game
python -m training.train --episodes 1000 --checkpoint-dir checkpoints

# Resume from checkpoint
python -m training.train --resume checkpoints/agent_ep500.pt

# Use aggressive reward preset
python -m training.train --simulated --reward-preset aggressive
```

## Training Module Structure

```
training/
├── __init__.py     # Module exports
├── model.py        # PolicyNetwork (actor-critic neural network)
├── ppo.py          # PPO algorithm with GAE
├── env.py          # Environment wrappers (GeneralsEnv, SimulatedEnv)
├── rewards.py      # Reward calculation with configurable presets
├── train.py        # Main training loop with checkpointing
└── metrics.py      # Training metrics and visualization
```

### Key Classes

- **PolicyNetwork**: Actor-critic network with 44 input features, 8 continuous outputs
- **PPOAgent**: PPO-Clip algorithm with configurable hyperparameters
- **GeneralsEnv**: Environment wrapper for real game communication
- **SimulatedEnv**: Lightweight environment for testing without the game
- **RewardConfig**: Configurable reward function with presets

## Protocol

Communication uses Windows named pipes with length-prefixed JSON messages.

### Pipe Name
`\\.\pipe\generals_ml_bridge`

### Message Format
- 4 bytes: message length (little-endian uint32)
- N bytes: JSON payload

### Game State (Game → Python)
```json
{
  "player": 1,
  "money": 3.5,
  "power": 50,
  "income": 10,
  "supply": 0.7,
  "own_infantry": [1.2, 0.9, 0],
  "own_vehicles": [0.8, 0.95, 0],
  "own_aircraft": [0.3, 1.0, 0],
  "own_structures": [1.0, 0.8, 0],
  "enemy_infantry": [0.9, 0.8, 0],
  "enemy_vehicles": [0.6, 0.9, 0],
  "enemy_aircraft": [0.0, 0.0, 0],
  "enemy_structures": [0.7, 0.9, 0],
  "game_time": 5.2,
  "tech_level": 0.4,
  "base_threat": 0.1,
  "army_strength": 1.3,
  "under_attack": 0,
  "distance_to_enemy": 0.6
}
```

### Recommendation (Python → Game)
```json
{
  "priority_economy": 0.25,
  "priority_defense": 0.25,
  "priority_military": 0.25,
  "priority_tech": 0.25,
  "prefer_infantry": 0.33,
  "prefer_vehicles": 0.34,
  "prefer_aircraft": 0.33,
  "aggression": 0.5,
  "target_player": -1
}
```

## Reward Presets

| Preset | Description |
|--------|-------------|
| `exploration` | Heavy shaping for early training |
| `balanced` | Default balanced configuration |
| `sparse` | Minimal shaping, mainly terminal rewards |
| `aggressive` | Rewards aggressive play style |

## Training Options

### Option 1: Manual Training (No Game Rebuild Required)

Use when the game doesn't have auto-skirmish support. You manually start each game.

```bash
# Start the trainer (waits for game connection)
python train_manual.py --episodes 10

# Then manually:
# 1. Launch game
# 2. Start Skirmish with Learning AI opponent
# 3. Play (training happens automatically)
# 4. When game ends, start another skirmish
```

### Option 2: Automated Training (Requires Game Rebuild)

Game automatically starts/restarts skirmishes for hands-off training.

```bash
# Train with auto-skirmish (game handles restarts)
python train_with_game.py --episodes 100

# Headless mode (faster, no graphics)
python train_with_game.py --episodes 500 --headless

# Against harder AI
python train_with_game.py --episodes 200 --ai 2
```

### Option 3: Simulated Environment (Testing)

Test training infrastructure without the game.

```bash
python -m training.train --simulated --episodes 100
```

## Training Workflow

1. **Simulated Testing**: Verify training loop works
   ```bash
   python -m training.train --simulated --episodes 50
   ```

2. **Manual Training**: Train with manual game starts (no rebuild needed)
   ```bash
   # Terminal: Start trainer
   python train_manual.py --episodes 20

   # Then launch game and start skirmishes manually
   ```

3. **Automated Training**: Full automation (requires game rebuild)
   ```bash
   python train_with_game.py --episodes 500 --headless
   ```

4. **Monitor Progress**: Check metrics and checkpoints
   ```bash
   python -c "from training.metrics import analyze_training_run; analyze_training_run('logs/training_*.jsonl')"
   ```

5. **Use Trained Model**: Load checkpoint for inference
   ```python
   from training import PPOAgent, PPOConfig
   agent = PPOAgent(PPOConfig())
   agent.load('checkpoints/best_agent.pt')
   ```

## Checkpoint Management

Checkpoints are saved every 10 episodes (configurable via `CHECKPOINT_INTERVAL` in `training/config.py`).

### Cleanup Old Checkpoints

Training generates many checkpoint files. To keep only the most recent:

```bash
# Keep last 5 checkpoints (Linux/WSL)
cd checkpoints/unified
ls -t agent_ep*.pt | tail -n +6 | xargs rm -f

# Keep only checkpoints divisible by 50 (milestone saves)
find . -name "agent_ep*.pt" | grep -vE "agent_ep(50|100|150|200|250|300|350|400|450|500)\.pt" | xargs rm -f
```

### Checkpoint Directory Structure

```
checkpoints/unified/
├── agent_ep10.pt       # Episode 10 checkpoint
├── agent_ep20.pt       # Episode 20 checkpoint
├── ...
├── agent_ep350.pt      # Latest checkpoint
└── best_agent.pt       # Best performing model (optional)
```

### Storage Estimates

- Each checkpoint: ~1-2 MB
- 500 episodes at interval=10: ~50 checkpoints = ~100 MB
- Recommendation: Keep last 5 + milestone saves
