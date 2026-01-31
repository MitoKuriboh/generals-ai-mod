# Project Index - generals-ai-mod

Complete catalog of all files, structures, and integration points for the Learning AI project.

**Last Updated:** Jan 31, 2026
**Project Phase:** 7 (Training Automation) - 81.5% simulated win rate

---

## Section 1: C++ Game Engine Files

### Core Learning AI (NEW FILES)

| File | Path | Purpose | Lines |
|------|------|---------|-------|
| AILearningPlayer.h | `GeneralsMD/Code/GameEngine/Include/GameLogic/` | ML-driven AI class header | ~80 |
| AILearningPlayer.cpp | `GeneralsMD/Code/GameEngine/Source/GameLogic/AI/` | Implementation | 500 |
| MLBridge.h | `GeneralsMD/Code/GameEngine/Include/GameLogic/` | Named pipe communication header | ~60 |
| MLBridge.cpp | `GeneralsMD/Code/GameEngine/Source/GameLogic/AI/` | Pipe implementation | 533 |

### Enums Added

| Enum | File | Values Added |
|------|------|--------------|
| `SlotState` | GameInfo.h:46 | `SLOT_LEARNING_AI` |
| `GameDifficulty` | GameCommon.h:146 | `DIFFICULTY_LEARNING` |
| `TeamCategory` | AILearningPlayer.h:44-51 | INFANTRY, VEHICLE, AIRCRAFT, MIXED, UNKNOWN |
| `BuildingCategory` | AILearningPlayer.h:54-63 | ECONOMY, POWER, DEFENSE, MILITARY, TECH, SUPER, UNKNOWN |

### Modified Files

| File | Path | Changes |
|------|------|---------|
| GameInfo.h | Include/GameNetwork/ | Added SLOT_LEARNING_AI |
| GameInfo.cpp | Source/GameNetwork/ | isOccupied(), isAI(), setState(), getApparentColor() fix |
| GameCommon.h | Include/Common/ | Added DIFFICULTY_LEARNING |
| Player.cpp | Source/Common/RTS/ | Creates AILearningPlayer for DIFFICULTY_LEARNING |
| SkirmishGameOptionsMenu.cpp | Source/GameClient/GUI/GUICallbacks/Menus/ | "Learning AI" dropdown |
| GameClient.cpp | Source/GameClient/ | startAutoSkirmish(), observer mode |
| GlobalData.h | Include/Common/ | m_autoSkirmish, m_autoSkirmishAI, m_autoSkirmishMap fields |
| GlobalData.cpp | Source/Common/ | Field initialization |
| CommandLine.cpp | Source/Common/ | -autoSkirmish, -aiDifficulty, -skirmishMap, -observer flags |
| MemoryInit.cpp | Source/Common/System/ | `{ "AILearningPlayer", 8, 8 }` memory pool |
| LoadScreen.cpp | Source/GameClient/GUI/ | NULL checks for colors, widgets, observer handling |
| ScoreScreen.cpp | Source/GameClient/GUI/GUICallbacks/Menus/ | Auto-restart in auto-skirmish mode |
| GameLogic.cpp | Source/GameLogic/System/ | SLOT_LEARNING_AI → DIFFICULTY_LEARNING mapping |

---

## Section 2: Data Structures

### MLGameState (18 floats exported to Python)

```
playerIndex       - AI player's index
money             - Normalized (actual / 1000)
powerBalance      - Power surplus/deficit
incomeRate        - Income per minute estimate
supplyUsed        - Population count

ownInfantry[3]    - Own infantry: small/medium/large
ownVehicles[3]    - Own vehicles: light/medium/heavy
ownAircraft[3]    - Own aircraft: small/medium/large
ownStructures[3]  - Own buildings: basic/production/tech

enemyInfantry[3]  - Enemy infantry counts
enemyVehicles[3]  - Enemy vehicle counts
enemyAircraft[3]  - Enemy aircraft counts
enemyStructures[3]- Enemy building counts

gameTimeMinutes   - Game time in minutes
techLevel         - Current tech level (0-3)
baseThreat        - Threat level to base
armyStrength      - Relative army strength
underAttack       - 1.0 if under attack, 0.0 otherwise
distanceToEnemy   - Normalized distance to enemy base
```

### MLRecommendation (8 outputs from Python)

```
priorityEconomy   - Build priority: economy (0.0-1.0)
priorityDefense   - Build priority: defense (0.0-1.0)
priorityMilitary  - Build priority: military (0.0-1.0)
priorityTech      - Build priority: tech (0.0-1.0)
                    (sum to 1.0)

preferInfantry    - Unit preference: infantry (0.0-1.0)
preferVehicles    - Unit preference: vehicles (0.0-1.0)
preferAircraft    - Unit preference: aircraft (0.0-1.0)
                    (sum to 1.0)

aggression        - Attack timing (0.0=defensive, 1.0=aggressive)
```

---

## Section 3: Python Files

### Training Module (`python/training/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | Module exports | - |
| `model.py` | Neural network | PolicyNetwork (44→8, actor-critic) |
| `ppo.py` | PPO algorithm | PPOAgent, PPOConfig, RolloutBuffer |
| `env.py` | Environment wrapper | GeneralsEnv, SimulatedEnv |
| `rewards.py` | Reward shaping | RewardConfig, presets |
| `train.py` | Training loop | Trainer, TrainingConfig |
| `metrics.py` | Metrics tracking | TrainingMetrics, RunningStats |
| `experiments.py` | Experiment management | ExperimentTracker, Experiment |

### Standalone Scripts (`python/`)

| File | Purpose |
|------|---------|
| `ml_bridge_server.py` | Rule-based test server (named pipe) |
| `ml_inference_server.py` | Trained model inference server |
| `train_manual.py` | Training for manual game starts |
| `train_with_game.py` | Unified game+training launcher |
| `game_launcher.py` | Game process management |
| `evaluate_model.py` | Post-training analysis |
| `analyze_states.py` | State extraction debugging |

### Batch Files (`python/`)

| File | Purpose |
|------|---------|
| `train.bat` | Launches train_with_game.py |
| `run_server.bat` | Launches ml_inference_server.py |

---

## Section 4: Documentation Files

| File | Path | Contents |
|------|------|----------|
| README.md | `/home/mito/generals-ai-mod/` | Project overview, structure |
| DESIGN.md | `/home/mito/generals-ai-mod/` | Architecture, decisions |
| STATE.md | `/home/mito/generals-ai-mod/` | Current progress, phases |
| INDEX.md | `/home/mito/generals-ai-mod/` | This file - master index |
| RESEARCH.md | `/home/mito/generals-ai-mod/docs/` | AI system analysis |
| VERIFICATION.md | `/home/mito/generals-ai-mod/docs/` | Testing checklist |
| README.md | `/home/mito/generals-ai-mod/python/` | Python module docs |

---

## Section 5: Command Line Options

```
-autoSkirmish         Start skirmish automatically
-observer             Local player as observer
-aiDifficulty <0-3>   0=Easy, 1=Med, 2=Hard, 3=Learning
-skirmishMap <name>   Map for auto-skirmish
-noDraw               Disable rendering
-noAudio              Disable audio
-noFPSLimit           Remove FPS cap
-quickstart           Skip intro movies
-seed <number>        Random seed
```

---

## Section 6: Communication Protocol

**Named Pipe:** `\\.\pipe\generals_ml_bridge`

**Message Format:** 4-byte length prefix + JSON payload

### State Message (Game → Python)

```json
{
  "type": "state",
  "player_index": 1,
  "money": 3.2,
  "power_balance": 50.0,
  "income_rate": 1.5,
  "supply_used": 25,
  "own_infantry": [5, 2, 0],
  "own_vehicles": [3, 1, 0],
  "own_aircraft": [0, 0, 0],
  "own_structures": [4, 2, 1],
  "enemy_infantry": [3, 1, 0],
  "enemy_vehicles": [2, 1, 0],
  "enemy_aircraft": [1, 0, 0],
  "enemy_structures": [3, 2, 0],
  "game_time_minutes": 5.2,
  "tech_level": 2,
  "base_threat": 0.3,
  "army_strength": 1.5,
  "under_attack": 0.0,
  "distance_to_enemy": 0.6
}
```

### Recommendation Message (Python → Game)

```json
{
  "priority_economy": 0.25,
  "priority_defense": 0.1,
  "priority_military": 0.5,
  "priority_tech": 0.15,
  "prefer_infantry": 0.2,
  "prefer_vehicles": 0.3,
  "prefer_aircraft": 0.5,
  "aggression": 0.7
}
```

### Game End Message (Game → Python)

```json
{
  "type": "game_end",
  "victory": true,
  "game_time": 5.2,
  "army_strength": 1.5
}
```

---

## Section 7: Build Commands

```bash
# Full rebuild
cd /mnt/c/dev/generals && cmd.exe /c build.bat

# Incremental (delete .obj first)
rm -f "/mnt/c/dev/generals/build/win32/GeneralsMD/Code/GameEngine/CMakeFiles/z_gameengine.dir/Release/Source/GameLogic/AI/AILearningPlayer.cpp.obj"
cd /mnt/c/dev/generals && cmd.exe /c build.bat

# Deploy
cp "/mnt/c/dev/generals/build/win32/GeneralsMD/Release/generalszh.exe" "/mnt/c/Program Files (x86)/Steam/steamapps/common/Command & Conquer Generals - Zero Hour/"
```

---

## Section 8: Training Commands

```bash
# Simulated (no game needed)
python -m training.train --simulated --episodes 100

# Manual (user starts games)
python train_manual.py --episodes 100

# Automated (launches game)
python train_with_game.py --episodes 100

# Headless (fast)
python train_with_game.py --episodes 500 --headless

# With specific AI opponent
python train_with_game.py --episodes 200 --ai 2  # vs Hard

# Resume checkpoint
python train_with_game.py --resume checkpoints/best_agent.pt

# Different reward preset
python -m training.train --simulated --reward-preset aggressive
```

---

## Section 9: Key Integration Points

| Integration | File | Location | Description |
|-------------|------|----------|-------------|
| AI Creation | Player.cpp | :916-920 | Checks DIFFICULTY_LEARNING, creates AILearningPlayer |
| Memory Pool | MemoryInit.cpp | :139 | `{ "AILearningPlayer", 8, 8 }` |
| Dropdown UI | SkirmishGameOptionsMenu.cpp | :1174-1175 | Adds "Learning AI" option |
| Slot Handling | GameInfo.cpp | multiple | SLOT_LEARNING_AI in isOccupied(), isAI(), setState() |
| Difficulty Map | GameLogic.cpp | :1432 | SLOT_LEARNING_AI → DIFFICULTY_LEARNING |
| Auto-Skirmish | GameClient.cpp | :131-156 | startAutoSkirmish() with observer support |
| ML Comm | AILearningPlayer.cpp | update() | Every 30 frames |

---

## Section 10: Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1: Skeleton | Learning AI selectable, game runs | ✓ |
| M2: Bridge | Python receives state, sends commands | ✓ |
| M3: State | Verification tools ready | ✓ |
| M4: Decisions | ML visibly influences unit composition | ✓ |
| M5: Training | PPO training infrastructure complete | ✓ |
| M6: Evaluation | Training pipeline verified (81.5% simulated win rate) | ✓ |
| M7: Automation | Headless training infrastructure (Jan 31, 2026) | ✓ |
| M8: Easy AI | Learning AI beats Easy AI >80% | Pending |
| M9: Competitive | Learning AI beats Hard AI >50% | Pending |

---

## Section 11: Directory Structure

```
/home/mito/generals-ai-mod/           # WSL - docs, git, Python
├── INDEX.md                          # This file
├── README.md                         # Project overview
├── DESIGN.md                         # Architecture decisions
├── STATE.md                          # Current progress
├── docs/
│   ├── RESEARCH.md                   # AI system analysis
│   └── VERIFICATION.md               # Testing checklist
├── python/
│   ├── README.md                     # Python module docs
│   ├── requirements.txt              # Dependencies
│   ├── train.bat                     # Training launcher
│   ├── run_server.bat                # Inference server launcher
│   ├── ml_bridge_server.py           # Rule-based server
│   ├── ml_inference_server.py        # Trained model server
│   ├── train_manual.py               # Manual training
│   ├── train_with_game.py            # Automated training
│   ├── game_launcher.py              # Game process manager
│   ├── evaluate_model.py             # Analysis tool
│   ├── analyze_states.py             # State debugging
│   ├── checkpoints/                  # Saved models
│   ├── logs/                         # Training logs
│   └── training/
│       ├── __init__.py
│       ├── model.py                  # PolicyNetwork
│       ├── ppo.py                    # PPO algorithm
│       ├── env.py                    # Environment wrappers
│       ├── rewards.py                # Reward calculation
│       ├── train.py                  # Training loop
│       ├── metrics.py                # Metrics tracking
│       └── experiments.py            # Experiment management
└── GeneralsMD/Code/GameEngine/       # C++ game engine
    ├── Include/
    │   ├── Common/
    │   │   ├── GameCommon.h          # DIFFICULTY_LEARNING
    │   │   └── GlobalData.h          # Auto-skirmish fields
    │   ├── GameLogic/
    │   │   ├── AILearningPlayer.h    # Learning AI header
    │   │   └── MLBridge.h            # Pipe communication
    │   └── GameNetwork/
    │       └── GameInfo.h            # SLOT_LEARNING_AI
    └── Source/
        ├── Common/
        │   ├── CommandLine.cpp       # CLI flags
        │   ├── GlobalData.cpp        # Field init
        │   ├── System/
        │   │   └── MemoryInit.cpp    # Memory pool
        │   └── RTS/
        │       └── Player.cpp        # AI creation
        ├── GameClient/
        │   ├── GameClient.cpp        # Auto-skirmish
        │   └── GUI/
        │       ├── LoadScreen.cpp    # NULL checks
        │       └── GUICallbacks/Menus/
        │           ├── SkirmishGameOptionsMenu.cpp  # Dropdown
        │           └── ScoreScreen.cpp              # Auto-restart
        ├── GameLogic/
        │   ├── AI/
        │   │   ├── AILearningPlayer.cpp  # Main implementation
        │   │   └── MLBridge.cpp          # Pipe implementation
        │   └── System/
        │       └── GameLogic.cpp         # Difficulty mapping
        └── GameNetwork/
            └── GameInfo.cpp              # Slot handling

C:\Users\Public\generals-ai-mod\     # Windows - execution
└── python/                          # Symlink or copy for Windows execution
```

---

## Section 12: Quick Reference

### Start Training (Recommended)

```bash
# From Windows, in C:\Users\Public\generals-ai-mod\python
train.bat --episodes 100

# Or from WSL
cd /home/mito/generals-ai-mod/python
python train_with_game.py --episodes 100
```

### Test Learning AI Manually

1. Start Python server: `python ml_bridge_server.py -v`
2. Launch game, start Skirmish
3. Select "Learning AI" from dropdown
4. Observe AI behavior influenced by ML recommendations

### Deploy After Build

```bash
cp "/mnt/c/dev/generals/build/win32/GeneralsMD/Release/generalszh.exe" \
   "/mnt/c/Program Files (x86)/Steam/steamapps/common/Command & Conquer Generals - Zero Hour/"
```

### Resume Context After Break

1. Read STATE.md for current phase and progress
2. Read this INDEX.md for file locations
3. Check docs/RESEARCH.md for technical details
