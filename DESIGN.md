# Design Document

## Overview

Create a new AI difficulty level that uses machine learning to make strategic decisions, while leveraging the existing game engine's AI infrastructure for execution.

## Architecture Concept

```
┌─────────────────────────────────────────────────────────┐
│                    Game Engine                          │
├─────────────────────────────────────────────────────────┤
│  AISkirmishPlayer (existing)                            │
│  ├── processBaseBuilding()  ← what to build            │
│  ├── processTeamBuilding()  ← what units to train      │
│  ├── acquireEnemy()         ← who to attack            │
│  └── update()               ← main loop                │
├─────────────────────────────────────────────────────────┤
│  AILearningPlayer (new)                                 │
│  ├── Inherits from AISkirmishPlayer or AIPlayer        │
│  ├── Overrides decision methods                         │
│  ├── Calls ML model for strategic choices              │
│  └── Uses existing infrastructure for execution        │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  ML Decision Layer                                      │
│  ├── State encoder (game state → features)             │
│  ├── Policy network (features → actions)               │
│  └── Value network (features → win probability)        │
└─────────────────────────────────────────────────────────┘
```

## Key Decisions

### 1. Integration Point

**Option A: New AIPlayer subclass**
- Create `AILearningPlayer` that extends `AISkirmishPlayer`
- Override key decision methods
- Cleanest separation, most maintainable

**Option B: Modify AISkirmishPlayer directly**
- Add learning mode flag
- Less code but messier

**Decision:** Option A - new subclass

### 2. ML Framework Integration

Options:
- Embed lightweight inference (ONNX Runtime, TensorRT)
- External process with IPC
- Pure C++ implementation of trained model

TBD based on performance requirements.

### 3. State Representation

What the ML model sees:
- Own units (types, counts, positions, health)
- Own structures (types, positions, health, production queues)
- Own resources (money, power status)
- Visible enemy units/structures
- Map features (terrain, resource locations)
- Game time / phase

### 4. Action Space

What the ML model decides:
- Build priority (which structure next)
- Unit composition (which teams to build)
- Attack timing (when to push)
- Target selection (which enemy to focus)
- Expansion decisions

### 5. Training Approach

Options:
- Self-play reinforcement learning
- Imitation learning from replays
- Hybrid approach

TBD after basic integration works.

## Implementation Phases

### Phase 1: Understanding
- Map existing AI completely
- Document all decision points
- Identify hook locations

### Phase 2: Infrastructure
- Create AILearningPlayer class
- Add new difficulty option
- Verify it can be selected and runs

### Phase 3: State Extraction
- Build game state encoder
- Export state for training data collection

### Phase 4: ML Integration
- Integrate inference runtime
- Connect to decision points

### Phase 5: Training
- Collect training data
- Train initial models
- Iterate on performance

## Architecture Diagrams

### High-Level Communication Flow

```
┌─────────────────────┐                    ┌─────────────────────┐
│    Game (C++)       │                    │    Python ML        │
│                     │                    │                     │
│  ┌───────────────┐  │   Named Pipe       │  ┌───────────────┐  │
│  │AILearningPlayer│◄─┼───────────────────┼──│ PPO Agent     │  │
│  └───────┬───────┘  │ \\.\pipe\generals  │  └───────┬───────┘  │
│          │          │   _ml_bridge       │          │          │
│          ▼          │                    │          ▼          │
│  ┌───────────────┐  │                    │  ┌───────────────┐  │
│  │   MLBridge    │  │                    │  │ PolicyNetwork │  │
│  │  - serialize  │──┼── JSON + 4-byte ──►│  │ (Actor-Critic)│  │
│  │  - deserialize│◄─┼── length prefix ───┼──│               │  │
│  └───────────────┘  │                    │  └───────────────┘  │
│                     │                    │                     │
└─────────────────────┘                    └─────────────────────┘
       Windows                                   WSL / Windows
```

### Message Protocol

```
┌────────────────────────────────────────────────────────────────┐
│                     Message Frame                               │
├──────────────┬─────────────────────────────────────────────────┤
│  Length (4B) │              JSON Payload (N bytes)             │
│  uint32 LE   │                                                  │
├──────────────┼─────────────────────────────────────────────────┤
│  0x2C010000  │  {"money":3.5,"power":50,"income":10,...}       │
│  (300 bytes) │                                                  │
└──────────────┴─────────────────────────────────────────────────┘
```

### Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │  State   │───►│  Policy  │───►│  Action  │───►│  Step    │  │
│   │ (44-dim) │    │ Network  │    │ (8-dim)  │    │   Env    │  │
│   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│        ▲                                               │        │
│        │           ┌──────────┐    ┌──────────┐        │        │
│        └───────────│  Store   │◄───│  Reward  │◄───────┘        │
│                    │ Rollout  │    │  + Done  │                 │
│                    └────┬─────┘    └──────────┘                 │
│                         │                                        │
│                         ▼                                        │
│                    ┌──────────┐                                  │
│                    │   PPO    │                                  │
│                    │  Update  │──► Repeat for N episodes        │
│                    └──────────┘                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### State/Action Spaces

```
Input State (44 features):                Output Action (8 values):
┌─────────────────────────────┐          ┌─────────────────────────┐
│ Economy:                    │          │ priority_economy   [0,1]│
│   money, power, income,     │          │ priority_defense   [0,1]│
│   supply                    │          │ priority_military  [0,1]│
├─────────────────────────────┤          │ priority_tech      [0,1]│
│ Own Forces (4 categories):  │          ├─────────────────────────┤
│   infantry, vehicles,       │          │ prefer_infantry    [0,1]│
│   aircraft, structures      │          │ prefer_vehicles    [0,1]│
│   (each: count, health, ?)  │          │ prefer_aircraft    [0,1]│
├─────────────────────────────┤          ├─────────────────────────┤
│ Enemy Forces (4 categories):│          │ aggression         [0,1]│
│   (visible only via FOW)    │          │ (0=hold, 1=attack)      │
├─────────────────────────────┤          └─────────────────────────┘
│ Strategic:                  │
│   game_time, tech_level,    │
│   base_threat, army_strength│
│   under_attack, distance    │
└─────────────────────────────┘
```

---

## Open Design Questions

1. How to handle real-time inference latency?
2. What granularity of decisions? (per-frame vs periodic)
3. How to balance exploration vs exploitation in-game?
4. Model architecture for RTS decision making?
