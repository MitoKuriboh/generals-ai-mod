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

## Open Design Questions

1. How to handle real-time inference latency?
2. What granularity of decisions? (per-frame vs periodic)
3. How to balance exploration vs exploitation in-game?
4. Model architecture for RTS decision making?
