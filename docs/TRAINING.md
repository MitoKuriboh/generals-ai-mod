# Training Guide

## Quick Start

```bash
# Activate environment
source ~/.bashrc
gai  # cd to project

# Start training
make train              # Manual mode (user starts games)
make train-sim          # Simulated environment (fast testing)
make train-auto         # Auto-launch game
```

## Training Modes

### Manual Mode (Recommended for Real Game)
```bash
python train.py --mode manual --episodes 100
```
- Creates named pipe, waits for game connection
- User manually starts skirmishes with Learning AI opponent
- Handles multiple episodes with reconnection between games

### Auto Mode
```bash
python train.py --mode auto --episodes 100 --headless
```
- Auto-launches game with `-autoSkirmish` flags
- Supports headless mode for faster training
- Requires game built with auto-skirmish support

### Simulated Mode (Testing)
```bash
python train.py --mode simulated --episodes 100
```
- Uses `SimulatedEnv` for fast testing
- No game required
- Good for validating pipeline changes

## Common Options

```bash
# Resume from checkpoint
python train.py --mode manual --episodes 500 --resume checkpoints/best_agent.pt

# Different map/difficulty
python train.py --mode auto --map "Tournament Desert" --ai 2  # vs Hard AI

# Verbose output
python train.py --mode manual --episodes 20 --verbose
```

## Recommended Progression

1. **Easy AI, 100 episodes** - Learn basics
2. **Easy AI, 500 episodes** - Solidify strategy
3. **Medium AI, 500 episodes** - Adapt to pressure
4. **Hard AI, 1000 episodes** - Master the game

## Checkpoint Management

Checkpoints are saved to `python/checkpoints/`:
- `best_agent.pt` - Best win rate so far
- `final_agent.pt` - Latest training state
- `agent_ep{N}.pt` - Periodic checkpoints
- `*_state.json` - Training metadata

## Hierarchical Training

For tactical and micro layers:

```bash
# Train all three layers jointly
python -m hierarchical.train_joint --use_simulated --episodes 500

# Start hierarchical server
python -m servers.hierarchical_server \
  --strategic checkpoints/best_agent.pt \
  --tactical checkpoints/tactical/tactical_best.pt \
  --micro checkpoints/micro/micro_best.pt
```

## Troubleshooting

### Pipe Connection Issues
- Ensure game is built with Learning AI support
- Check firewall isn't blocking named pipes
- Try increasing timeout: `--timeout 60`

### Training Errors
- Emergency checkpoints saved automatically on error
- Check `logs/` for detailed error messages

### Low Win Rate
- Start with simulated mode to verify pipeline
- Use `exploration` reward preset initially
- Ensure terminal rewards are being applied (check logs)
