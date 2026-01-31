# Research Notes

## Existing AI System Analysis

### AISkirmishPlayer.cpp

Located at: `GeneralsMD/Code/GameEngine/Source/GameLogic/AI/AISkirmishPlayer.cpp`

**Author:** Michael S. Booth, January 2002

**Class hierarchy:** `AISkirmishPlayer` extends `AIPlayer`

**Key methods:**

1. `processBaseBuilding()` (line 99-314)
   - Iterates through build list
   - Checks if buildings need rebuilding
   - Handles power plant priority when underpowered
   - Uses dozers for construction
   - Timing controlled by `m_structureTimer`

2. `processTeamBuilding()` (line 859-865)
   - Calls `selectTeamToBuild()` then `queueUnits()`
   - Teams are predefined unit compositions

3. `acquireEnemy()` (line 485-547)
   - Finds closest enemy by distance
   - Penalizes enemies already targeted by other AIs (prevents ganging up)
   - Prefers enemies attacking us
   - Deprioritizes crippled enemies

4. `buildAIBaseDefenseStructure()` (line 604-711)
   - Places defenses in arc around base
   - Alternates front/flank coverage
   - Uses geometric calculations for placement

5. `computeSuperweaponTarget()` (line 1135-1178)
   - Special case for cluster mines (defends own base)
   - Otherwise delegates to parent class

**Difficulty influence:**
- `AISideInfo` has `m_easy`, `m_normal`, `m_hard` (number of gatherers)
- `TAiData` has timing modifiers:
  - `m_structureSeconds` - delay between structure attempts
  - `m_teamSeconds` - delay between team builds
  - `m_structuresWealthyMod` / `m_structuresPoorMod` - speed multipliers

### AI.h

Located at: `GeneralsMD/Code/GameEngine/Include/GameLogic/AI.h`

**Key structures:**

1. `AISideInfo` - Per-faction AI configuration
   - Gatherer counts by difficulty
   - Skill sets (general's powers choices)
   - Base defense structure templates

2. `TAiData` - Global AI parameters
   - Timing values
   - Resource thresholds (wealthy/poor)
   - Guard/attack modifiers
   - Pathfinding parameters

3. `AICommandInterface` - Command abstraction
   - Move, attack, guard, special powers
   - All commands go through `aiDoCommand()`

### How Decisions Are Made (Current System)

1. **What to build:** Follows predefined `BuildListInfo` in order
2. **When to build:** Timer-based with resource modifiers
3. **What units:** Predefined team templates with conditions
4. **Who to attack:** Distance + heuristics
5. **Where to defend:** Geometric placement around base

All purely rule-based. No learning, no adaptation.

## Integration Points for ML

### Best hook points:

1. **`selectTeamToBuild()`** - Override to choose teams via ML
2. **`processBaseBuilding()`** - Override build order decisions
3. **`acquireEnemy()`** - Override target selection
4. **`update()`** - Main loop, can add ML decision cycle

### State available to AI:

```cpp
// Own player
m_player->getMoney()->countMoney()
m_player->getEnergy()->hasSufficientPower()
m_player->getBuildList()

// All objects
TheGameLogic->getFirstObject() // iterate all
obj->getControllingPlayer()
obj->isKindOf(KINDOF_...)
obj->getPosition()

// Enemies
ThePlayerList->getPlayerCount()
ThePlayerList->getNthPlayer(i)
player->getRelationship() == ENEMIES

// Pathfinding
TheAI->pathfinder()->...
```

## Difficulty System (Complete Flow)

### 1. UI Selection
`SkirmishGameOptionsMenu.cpp:1136-1140` - Dropdown populated with:
```cpp
GadgetComboBoxSetItemData(comboBoxPlayer[i], 2, (void *)SLOT_EASY_AI);
GadgetComboBoxSetItemData(comboBoxPlayer[i], 3, (void *)SLOT_MED_AI);
GadgetComboBoxSetItemData(comboBoxPlayer[i], 4, (void *)SLOT_BRUTAL_AI);
```

### 2. Slot State Enum
`GameInfo.h:39-47`:
```cpp
enum SlotState {
    SLOT_OPEN,
    SLOT_CLOSED,
    SLOT_EASY_AI,
    SLOT_MED_AI,
    SLOT_BRUTAL_AI,
    SLOT_PLAYER
};
```

### 3. Difficulty Enum
`GameCommon.h:141-148`:
```cpp
enum GameDifficulty {
    DIFFICULTY_EASY,
    DIFFICULTY_NORMAL,
    DIFFICULTY_HARD,
    DIFFICULTY_COUNT
};
```

### 4. Game Start - Slot to Difficulty Mapping
`GameLogic.cpp:1425-1434`:
```cpp
if (isSkirmishOrSkirmishReplay) {
    d.setBool(TheKey_playerIsSkirmish, true);
    switch (slot->getState()) {
        case SLOT_EASY_AI : d.setInt(TheKey_skirmishDifficulty, DIFFICULTY_EASY); break;
        case SLOT_MED_AI : d.setInt(TheKey_skirmishDifficulty, DIFFICULTY_NORMAL); break;
        case SLOT_BRUTAL_AI : d.setInt(TheKey_skirmishDifficulty, DIFFICULTY_HARD); break;
        default: break;
    }
}
```

### 5. Player Creation - AI Instantiation
`Player.cpp:778-786`:
```cpp
if (t == PLAYER_COMPUTER) {
    if (skirmish || TheAI->getAiData()->m_forceSkirmishAI) {
        m_ai = newInstance(AISkirmishPlayer)( this );
    } else {
        m_ai = newInstance(AIPlayer)( this );
    }
}
```

### 6. AI Reads Difficulty
`AIPlayer.cpp:104`:
```cpp
m_difficulty = TheScriptEngine->getGlobalDifficulty();
```

### 7. Difficulty Affects Behavior
`AIPlayer.cpp:203-212` - Gatherer count based on difficulty:
```cpp
if (difficulty == DIFFICULTY_EASY) {
    desiredGatherers = resInfo->m_easy;
}
if (difficulty == DIFFICULTY_NORMAL) {
    desiredGatherers = resInfo->m_normal;
}
if (difficulty == DIFFICULTY_HARD) {
    desiredGatherers = resInfo->m_hard;
}
```

## Adding a New Difficulty Level

To add "DIFFICULTY_LEARNING" / "SLOT_LEARNING_AI":

1. **GameCommon.h** - Add `DIFFICULTY_LEARNING` to enum
2. **GameInfo.h** - Add `SLOT_LEARNING_AI` to SlotState enum
3. **GameInfo.cpp** - Update `isAI()`, `isOccupied()` methods
4. **GameLogic.cpp** - Add case for SLOT_LEARNING_AI → DIFFICULTY_LEARNING
5. **Player.cpp** - Create `AILearningPlayer` for learning difficulty
6. **SkirmishGameOptionsMenu.cpp** - Add dropdown option
7. **Create AILearningPlayer class** - New subclass with ML decisions

## External Resources

- [EA GitHub Repository](https://github.com/electronicarts/CnC_Generals_Zero_Hour)
- [TheSuperHackers Community Fork](https://github.com/TheSuperHackers/GeneralsGameCode)
- Released under GPLv3, February 2025

## Training Progress Analysis (2026-01-31)

### Saved Checkpoints

Location: `python/checkpoints/`

| File | Episode | Wins | Losses | Win Rate |
|------|---------|------|--------|----------|
| `best_agent.pt` | 100 | 71 | 29 | **71%** |
| `agent_ep100.pt` | 100 | 71 | 29 | 71% |
| `agent_ep200.pt` | 200 | - | - | ~81.5% |
| `final_agent.pt` | 100 | 71 | 29 | 71% |

### Training Logs

Location: `python/logs/`

| File | Episodes | Notes |
|------|----------|-------|
| `training_20260130_164614.jsonl` | 200 | Main training run |
| `training_20260130_170441.jsonl` | 100 | Recent run, 100% win rate at end |
| **Total** | 350 episodes | All simulated |

### Key Finding: All Training is Simulated

**Indicators:**
- `simulated: true` in config
- 0 units killed/lost (simulated env doesn't model combat)
- Playing C&C Generals as China manually doesn't generate training data

### Recent Performance (Episodes 81-100)

- 100% win rate (all 20 episodes won)
- Average reward: 4-7 per episode
- Army strength: 1.5-2.5x enemy

### Training Modes Explained

```
Simulated Training (what exists):
  SimulatedEnv → generates fake states
  PolicyNetwork → recommends actions
  PPO → updates network
  Checkpoint saved

Real Game Training (not yet done):
  1. Start C&C Generals with Learning AI
  2. Run: python -m training.train --episodes 100
  3. Game sends state via named pipe
  4. Python sends recommendations back
  5. Episode ends → reward calculated → PPO update
```

### Commands for Real Game Training

```bash
# Using training script directly
cd /home/mito/generals-ai-mod/python
python -m training.train --episodes 100 --checkpoint-dir checkpoints

# Using game launcher for headless training
python game_launcher.py --headless --episodes 100
```

### What's Needed for Real Training

1. Build game with auto-skirmish support (Phase 7 code complete)
2. Install pywin32: `pip install pywin32`
3. Run training script while game is running, OR
4. Use game_launcher.py for automated training

## Research Tasks

- [x] Read `AISkirmishPlayer.cpp` for skirmish AI behavior
- [x] Read `AIPlayer.cpp` for base class behavior
- [x] Find where difficulty is selected (UI → game init) - COMPLETE
- [x] Trace AI player instantiation - COMPLETE
- [x] Analyze training progress and logs - COMPLETE (2026-01-31)
- [ ] Study `AIStates.cpp` for state machine
- [ ] Look at INI files for AI configuration
- [ ] Understand team/build list data format
