# Project State

## Current Phase: Phase 7 Complete ✓ (Training Automation)

Training pipeline verified working. Simulated training achieves 81.5% win rate. Game automation implemented for headless training.

## Completed Work

### Phase 1: Skeleton Integration ✓
Basic Learning AI class integrated into game engine.

### Phase 2: ML Bridge ✓
Windows named pipe communication between game and Python.

### Phase 3: State Verification ✓
Logging and analysis tools for verifying state extraction.

### Phase 4: Decision Override ✓

| Component | Implementation |
|-----------|----------------|
| Team Selection | Weighted random selection based on `prefer_infantry/vehicles/aircraft` |
| Team Classification | Analyzes team template to categorize as infantry/vehicle/aircraft/mixed |
| Building Classification | Categories: economy, power, defense, military, tech, super |
| Attack Timing | `checkReadyTeams()` holds teams when aggression is low |
| Python Rules | Phase-based strategy (early/mid/late) + situational adjustments |

### ML Influence Points

```
┌─────────────────────────────────────────────────────────────────┐
│  ML Recommendation Fields                                       │
├─────────────────────────────────────────────────────────────────┤
│  Build Priorities (sum to 1.0):                                 │
│  ├── priority_economy  → power plants, supply                   │
│  ├── priority_defense  → turrets, walls                         │
│  ├── priority_military → barracks, war factory, airfield        │
│  └── priority_tech     → tech center, strategy center           │
├─────────────────────────────────────────────────────────────────┤
│  Army Composition (sum to 1.0):                                 │
│  ├── prefer_infantry   → weight for infantry teams              │
│  ├── prefer_vehicles   → weight for vehicle teams               │
│  └── prefer_aircraft   → weight for aircraft teams              │
├─────────────────────────────────────────────────────────────────┤
│  Aggression (0.0 to 1.0):                                       │
│  ├── < 0.3  → Hold teams, build up army before attacking        │
│  ├── 0.3-0.7 → Normal attack timing                             │
│  └── > 0.7  → Attack immediately when teams ready               │
└─────────────────────────────────────────────────────────────────┘
```

### Decision Flow

```
selectTeamToBuild():
  1. Find all buildable teams at highest priority
  2. Classify each team (infantry/vehicle/aircraft/mixed)
  3. Get ML weight for each category
  4. Weighted random selection
  5. Build selected team

checkReadyTeams():
  1. If aggression > 0.7: attack immediately
  2. Calculate hold time: (1 - aggression) * 30 seconds
  3. If not enough time since last attack: hold teams
  4. Otherwise: allow teams to activate
```

### Python Server Strategy

The rule-based server implements:

**Phase-Based:**
- Early (0-3 min): Economy focus, prefer infantry, low aggression
- Mid (3-8 min): Balanced, tech up, shift to vehicles
- Late (8+ min): Military focus, mixed army, high aggression

**Situational:**
- Under attack → defensive stance
- Low money → economy priority
- High money → military/tech priority
- Strong army → high aggression
- Weak army → defensive, build up
- Enemy air → counter with own air

### Phase 5: Training Pipeline ✓

| Component | Implementation |
|-----------|----------------|
| Neural Network | `model.py` - Actor-critic PolicyNetwork with 44 input features, 8 continuous outputs |
| PPO Algorithm | `ppo.py` - PPO-Clip with GAE, configurable hyperparameters |
| Environment | `env.py` - GeneralsEnv (real game) and SimulatedEnv (testing) |
| Rewards | `rewards.py` - Combat, economy, strategic rewards with presets |
| Training Loop | `train.py` - Episode collection, PPO updates, checkpointing |
| Metrics | `metrics.py` - Running stats, learning curves, export |

### Training Infrastructure

```
python/training/
├── __init__.py     # Module exports
├── model.py        # PolicyNetwork (actor-critic)
├── ppo.py          # PPO algorithm, RolloutBuffer
├── env.py          # Game environment wrapper
├── rewards.py      # Reward calculation
├── train.py        # Main training loop
└── metrics.py      # Training metrics and plotting
```

### Usage

```bash
# Test with simulated environment
python -m training.train --simulated --episodes 100

# Train with real game (requires game running)
python -m training.train --episodes 1000 --checkpoint-dir checkpoints

# Resume from checkpoint
python -m training.train --resume checkpoints/agent_ep500.pt

# Use different reward preset
python -m training.train --simulated --reward-preset aggressive
```

### Reward Presets

- `exploration` - Heavy shaping for early training
- `balanced` - Default configuration
- `sparse` - Minimal shaping, mainly terminal rewards
- `aggressive` - Rewards aggressive play style

### Phase 6: Training and Evaluation ✓

| Component | Implementation |
|-----------|----------------|
| Inference Server | `ml_inference_server.py` - Serves trained model recommendations |
| Evaluation Tool | `evaluate_model.py` - Analyze runs, compare, plot, export |
| Experiment Tracking | `experiments.py` - Track hyperparameters and results |
| Presets | baseline, exploration, fine_tune, aggressive configurations |

### Simulated Training Results

Verified training pipeline with 200 episodes:

| Metric | Start | End |
|--------|-------|-----|
| Win Rate | 20% | 81.5% |
| Recent Win Rate (last 50) | - | 100% |
| Avg Reward | ~1.2 | ~35.0 |
| Total Steps | 0 | 20,000 |

Model learned to prioritize military (62%) and aircraft (80%), demonstrating meaningful strategy acquisition.

**Note:** All 350 episodes of training are against `SimulatedEnv`, not the real game. Playing C&C Generals manually does not generate training data. Real game training requires running `python -m training.train` or `game_launcher.py` while the game is running.

### Phase 7: Training Automation (In Progress)

| Component | Implementation |
|-----------|----------------|
| Auto-Skirmish Mode | Command line flags to start skirmish without GUI |
| Game End Detection | `checkGameEnd()` detects victory/defeat, notifies ML |
| Python Launcher | `game_launcher.py` manages game process, collects episodes |
| Headless Mode | `-noDraw`, `-noAudio`, `-noFPSLimit` for fast training |

### Command Line Options Added

```
-autoSkirmish         Start skirmish automatically without GUI
-observer             Put local player in observer slot (watch AI vs AI)
-aiDifficulty <0-3>   AI difficulty (0=Easy, 1=Med, 2=Hard, 3=Learning)
-skirmishMap <name>   Map for auto-skirmish
-noDraw               Disable rendering (faster)
-noAudio              Disable audio
-noFPSLimit           Remove frame rate cap
-quickstart           Skip intro movies
-seed <number>        Random seed for reproducibility
```

**Observer Mode (Jan 31, 2026):**
- `-autoSkirmish -observer` now works correctly
- Fixed: Added `setLocalIP()` call - game needed to identify local player
- Observer goes in slot 0 with `PLAYERTEMPLATE_OBSERVER`
- Learning AI and opponent AI fill remaining slots
- Fixed: LoadScreen.cpp NULL pointer crash when FactionObserver not in PlayerTemplateStore
  - `MultiPlayerLoadScreen::init()` and `GameSpyLoadScreen::init()` now handle NULL `pt`
  - Fallback to "GUI:PlayerObserver" text when observer template missing

**Learning AI Skirmish Fix (Jan 31, 2026):**
- Fixed: Game crash when selecting Learning AI in Skirmish
- Root cause: AILearningPlayer missing from memory pool initialization
- Fix: Added `{ "AILearningPlayer", 8, 8 }` to MemoryInit.cpp
- Additional hardening: NULL checks in LoadScreen.cpp for color lookups and widget pointers

**Self-Play Server Fix (Jan 31, 2026):**
- Fixed: `KeyError: 3` crash in `self_play_server.py`
- Root cause: Player ID tracking hardcoded as 0/1, but game sends 3/4 in skirmish with observer
- Changes to `C:\Users\Public\game-ai-agent\self_play_server.py`:
  - Added `PLAYER_SLOTS = [3, 4]` configuration constant
  - Changed `wins` and `recent_rewards` from fixed dicts to dynamic dicts
  - GAE computation now uses `np.unique(player_ids)` instead of hardcoded `[0, 1]`
  - Checkpoint format updated with migration support for old `player0_wins`/`player1_wins`
  - TensorBoard logging now dynamic per player ID

### Game Launcher Usage

```bash
# Single episode (visual)
python game_launcher.py --episodes 1

# Headless training (fast)
python game_launcher.py --headless --episodes 100 --save-dir ./episodes

# With specific map and seed
python game_launcher.py --map "Alpine Assault" --seed 12345 --episodes 10
```

### Evaluation Workflow

```bash
# 1. Test with simulated environment
python -m training.train --simulated --episodes 100

# 2. Train against real game
python -m training.train --episodes 500 --checkpoint-dir checkpoints

# 3. Evaluate trained model
python ml_inference_server.py --model checkpoints/best_agent.pt --log eval.jsonl

# 4. Analyze results
python evaluate_model.py eval.jsonl --summary --plot
```

### Experiment Presets

| Preset | Purpose | Key Settings |
|--------|---------|--------------|
| `baseline` | Default configuration | lr=3e-4, entropy=0.01 |
| `exploration` | Initial training | lr=5e-4, entropy=0.05 |
| `fine_tune` | Polish trained model | lr=1e-4, entropy=0.005 |
| `aggressive` | Aggressive playstyle | gamma=0.995, aggressive rewards |

### Phase 7 Implementation Complete

**Manual Training (train_manual.py):**
- Created for training without game rebuild
- Creates named pipe, waits for game connection
- User manually starts skirmishes
- Handles multiple episodes with reconnection between games

**Auto-Skirmish (C++ Changes):**
- `GameClient.cpp`: Added `startAutoSkirmish()` function
- `GameClient.cpp`: Modified `m_afterIntro` block to call auto-skirmish when flag set
- `ScoreScreen.cpp`: Added game end handling to restart auto-skirmish
- Game now auto-starts and restarts games when launched with `-autoSkirmish`

**Auto-Launch Trainer (C++ Changes):**
- `MLBridge.cpp`: Added `launchTrainer()` function
- When Learning AI tries to connect and pipe doesn't exist, automatically launches `python/train_manual.py`
- Works via `CreateProcess()` with `pythonw.exe` (no console window)
- **Just start a skirmish with Learning AI - trainer starts automatically!**

### Learning AI Integration (Completed)

**Files Modified:**
- `Include/GameNetwork/GameInfo.h` - Added `SLOT_LEARNING_AI` to SlotState enum
- `Source/GameNetwork/GameInfo.cpp` - Updated isOccupied(), isAI(), setState(), serialization
- `Include/Common/GameCommon.h` - Added `DIFFICULTY_LEARNING` to GameDifficulty enum
- `Source/GameLogic/System/GameLogic.cpp` - Added SLOT_LEARNING_AI case for difficulty mapping
- `Source/Common/RTS/Player.cpp` - Creates AILearningPlayer when difficulty is DIFFICULTY_LEARNING
- `Source/GameClient/GUI/.../SkirmishGameOptionsMenu.cpp` - Added "Learning AI" to dropdown (index 5)
- `Include/Common/GlobalData.h` - Added m_autoSkirmish, m_autoSkirmishAI, m_autoSkirmishMap fields
- `Source/Common/GlobalData.cpp` - Initialize auto-skirmish fields
- `Source/Common/CommandLine.cpp` - Added -autoSkirmish, -aiDifficulty, -skirmishMap parsers
- `Source/GameClient/GameClient.cpp` - Implemented startAutoSkirmish() with observer mode support
- `Source/GameClient/GUI/.../ScoreScreen.cpp` - Auto-restart when in auto-skirmish mode

**Features:**
1. Learning AI now selectable in Skirmish menu (appears after "Hard AI")
2. Auto-skirmish mode: `-autoSkirmish -aiDifficulty 3 -skirmishMap "Maps/Test/Test.map"`
3. Game auto-restarts after episode ends when in auto-skirmish mode

## Build Status

**Game Build: SUCCESSFUL ✓**
- Build date: Jan 31, 2026
- Output: `C:\dev\generals\build\win32\GeneralsMD\Release\generalszh.exe`
- Learning AI integration complete

## Next Steps

1. ~~Deploy built exe to Steam folder~~ ✓
2. ~~Test Learning AI selection in skirmish menu~~ ✓ (Fixed Jan 31, 2026)
3. Test manual training with `python train_manual.py`
4. Test automated training with auto-skirmish
5. Graduate to Hard AI once >80% vs Easy

## Training Workflow

### Prerequisites
```bash
# On Windows, in the python directory:
pip install torch pywin32
```

### Quick Start (Recommended)
```bash
# From C:\Users\Public\generals-ai-mod\python (or wherever game runs)

# Option 1: Use batch file
train.bat --episodes 100

# Option 2: Direct Python
python train_with_game.py --episodes 100
```

### Training Options
```bash
# Train 100 episodes vs Easy AI (visual, see what's happening)
python train_with_game.py --episodes 100

# Train headless (faster, no graphics)
python train_with_game.py --episodes 500 --headless

# Train vs Hard AI
python train_with_game.py --episodes 200 --ai 2

# Resume from checkpoint
python train_with_game.py --episodes 500 --resume checkpoints/best_agent.pt

# Specific map
python train_with_game.py --map "Tournament Desert" --episodes 100
```

### How It Works
```
train_with_game.py (unified launcher):
  1. Creates named pipe for game communication
  2. Launches game with -autoSkirmish flags
  3. Game connects to pipe, sends state every decision point
  4. Python receives state, runs through PPO policy
  5. Python sends recommendation back to game
  6. Game applies recommendation, continues
  7. Episode ends (win/loss) → PPO update
  8. Game restarts automatically → next episode
```

### Recommended Training Progression
1. **Easy AI, 100 episodes** - Learn basics
2. **Easy AI, 500 episodes** - Solidify strategy
3. **Medium AI, 500 episodes** - Adapt to pressure
4. **Hard AI, 1000 episodes** - Master the game

## Verification Steps

To verify Phase 4:
1. Build the game
2. Run Python server with verbose mode: `python ml_bridge_server.py -v`
3. Start skirmish with Learning AI
4. Observe:
   - Team selection logs show ML-weighted choices
   - Attack timing varies with aggression recommendations
   - Early game: low aggression, prefer infantry
   - Late game: high aggression, mixed army

## File Locations

```
GeneralsMD/Code/GameEngine/
├── Include/GameLogic/
│   ├── AILearningPlayer.h    (TeamCategory, BuildingCategory enums)
│   └── MLBridge.h
├── Source/GameLogic/AI/
│   ├── AILearningPlayer.cpp  (classifyTeam, getTeamCategoryWeight, etc.)
│   └── MLBridge.cpp
└── GameEngine.dsp

python/
├── ml_bridge_server.py       (rule-based strategy for testing)
├── ml_inference_server.py    (trained model inference)
├── evaluate_model.py         (analysis and comparison)
├── analyze_states.py         (state log analysis)
├── game_launcher.py          (automated game launching for training)
├── requirements.txt
├── README.md
└── training/
    ├── __init__.py
    ├── model.py              (PolicyNetwork)
    ├── ppo.py                (PPO algorithm)
    ├── env.py                (environment wrappers)
    ├── rewards.py            (reward calculation)
    ├── train.py              (training loop)
    ├── metrics.py            (metrics tracking)
    └── experiments.py        (experiment management)

docs/
└── VERIFICATION.md
```

## Plan Reference

Full implementation plan: `/home/mito/.claude/plans/purring-humming-horizon.md`

Milestones:
- [x] M1: Skeleton - Learning AI selectable, game runs
- [x] M2: Bridge - Python receives state, sends commands
- [x] M3: State - Verification tools ready
- [x] M4: Decisions - ML visibly influences unit composition
- [x] M5: Training - PPO training infrastructure complete
- [x] M6: Evaluation - Training pipeline verified (81.5% simulated win rate)
- [x] M7: Automation - Headless training infrastructure complete (built Jan 31, 2026)
- [ ] M8: Easy AI - Learning AI beats Easy AI >80%
- [ ] M9: Competitive - Learning AI beats Hard AI >50%
