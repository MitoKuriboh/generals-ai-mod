# Changelog

Historical development log for C&C Generals AI. For current status, see `STATUS.md`.

---

## Feb 6, 2026 - Training Pipeline Stability Fixes

**Root Cause Analysis:** 10-game stability test failed due to tensor shape mismatches and resource leaks.

### Phase 1: Tensor Dimension Consistency (CRITICAL)

| Issue | File | Fix |
|-------|------|-----|
| `get_action()` returns scalars | model.py:298 | Changed `squeeze(0)` to `view(-1), view(1), view(1)` for consistent 1D shapes |
| Legacy model same issue | model.py:167 | Same fix for `LegacyPolicyNetwork.get_action()` |
| Terminal value 0D tensor | train_manual.py:422,433 | Changed `torch.tensor(0.0)` to `torch.tensor([0.0])` |
| Terminal value 0D tensor | train_with_game.py:358,372 | Same fix |

**Root Cause:** `torch.stack()` fails when mixing 0D scalars with 1D tensors.

### Phase 2: Resource Leak Prevention (HIGH)

| Issue | File | Fix |
|-------|------|-----|
| Event handle never closed | train_manual.py:210-240 | Added `win32api.CloseHandle(overlapped.hEvent)` in finally block |
| Event handle never closed | game_launcher.py:88-113 | Same fix |
| Incomplete pipe read | env.py:140-144 | Loop until all bytes received |

---

## Feb 5, 2026 - Critical Issues Fix Round 2

Deep audit identified 29 additional issues (7 critical, 13 high, 9 medium). All critical and high issues fixed.

### Phase 1: Crash Prevention (CRITICAL)

| ID | Issue | File | Fix |
|----|-------|------|-----|
| P1 | torch.exp() overflow | ppo.py, tactical/train.py, micro/train.py | `torch.exp(torch.clamp(log_probs - old_log_probs, -20, 20))` |
| P2 | Minibatch infinite loop | ppo.py, tactical/train.py, micro/train.py | `minibatch_size = max(1, n // num_minibatches)` |
| C1 | NULL enemyPos | MicroState.cpp:519 | Added null check with `*outAngle = 0.0f` fallback |
| C2 | Uninitialized basePos | TacticalState.cpp:327 | Added `if (foundBase)` check before distance calculation |

### Phase 2: Stability Fixes (HIGH)

| ID | Issue | File | Fix |
|----|-------|------|-----|
| P3 | Division by zero | model.py:90,95 | Added epsilon: `priorities / (priorities.sum() + 1e-8)` |
| C4 | TheAI->getAiData() unchecked | AILearningPlayer.cpp:880 | Added `if (TheAI && TheAI->getAiData())` guard |
| C5 | Nested O(n²) iteration | AILearningPlayer.cpp:346 | Added per-frame caching for enemy proximity check |

### Phase 3: Integration Robustness

| ID | Issue | File | Fix |
|----|-------|------|-----|
| I1 | Stale recommendation silent | MLBridge.cpp/h | Added `checkAndHandleStaleness()` with reconnection trigger at 5-sec timeout |
| I4 | Reconnection 10-sec gap | MLBridge.h:303 | Reduced `RECONNECT_INTERVAL_FRAMES` from 300 to 60 (~2 sec) |
| P6 | Pipe read incomplete | env.py:135 | Added `len(data) < length` validation, 1MB sanity check |

---

## Feb 5, 2026 - Comprehensive Gap Analysis Fixes

Full audit of project identified 68 remaining issues. MVP fixes implemented.

### Critical Fixes Implemented

| Issue | File | Fix |
|-------|------|-----|
| Path hardcoding | MLBridge.cpp | Added GENERALS_AI_DIR env var support, multi-location search |
| Pipe timeout too short | env.py | Increased 5s→30s default, added configurable timeout + 3 retries |
| TheAI null check | AILearningPlayer.cpp | Added null check + group member validation |
| Protocol version not validated | ml_bridge_server.py, ml_inference_server.py | Added validate_protocol_version() |
| Training error handling | train.py, train_with_game.py | Added try-catch with emergency checkpoint |
| Memory leak in coordinator | coordinator.py | Added periodic cleanup_stale_units() |

---

## Feb 2, 2026 - Tactical Command Crash Fix

**Problem:** Game crashed after Python coordinator started returning tactical commands.

**Root Cause:** `executeTacticalCommand()` called `TheAI->createGroup()` without null-checking.

**Fixes:**
- Added `if (!TheAI)` null check
- Added `group->getCount() == 0` check
- Enhanced logging for debug

---

## Feb 1, 2026 - Hierarchical RL Implementation

### Phase 8: Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  STRATEGIC LAYER (Existing - ~1 sec intervals)                      │
│  Input: 44 floats | Output: 8 floats | Network: 256-256 MLP        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TACTICAL LAYER (New - ~5 sec intervals per team)                   │
│  Input: 64 floats | Output: 8 discrete + 3 continuous              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MICRO LAYER (New - ~0.5 sec intervals per unit)                    │
│  Input: 32 floats | Output: 11 discrete + 2 continuous             │
└─────────────────────────────────────────────────────────────────────┘
```

### New Python Modules
- `tactical/` - Tactical layer (team commands)
- `micro/` - Micro layer (unit control)
- `hierarchical/` - Coordination and joint training
- `servers/` - Hierarchical inference server

### Capability Negotiation Protocol
Response-based capability declaration - Python server declares capabilities in every response.

---

## Feb 1, 2026 - ML Mechanism Review (Rounds 1-3)

### Round 1: High Priority Fixes

| Issue | File | Fix |
|-------|------|-----|
| Recommendation staleness | MLBridge.h/cpp | Added 2-second timeout |
| Force array unused | AILearningPlayer.cpp | Now tracks avg health ratio |
| JSON value validation | MLBridge.cpp | Added clamping for all parsed values |

### Round 2: Critical Training Fixes

| Issue | File | Fix |
|-------|------|-----|
| **Terminal Reward Not Applied** | train_*.py | Added terminal transition storage BEFORE agent.update() |
| Hidden 10x multipliers | rewards.py | Removed from strategic rewards |
| Buffer overflow in stateToJson | MLBridge.cpp | Converted to snprintf |
| Race condition in game-end | AILearningPlayer.cpp | Set flag BEFORE sending |

### Round 3: Policy Gradient Fixes

| Issue | File | Fix |
|-------|------|-----|
| **Beta distribution log-prob on clamped** | model.py | Compute log_prob on raw sample BEFORE clamping |
| Entropy decay never implemented | ppo.py | Added decay after each update |

---

## Feb 1, 2026 - Cleanup & Documentation Phase

### Quick Wins
- Deleted `python_ml.tar.gz` (1.1 MB obsolete backup)
- Consolidated reward constants
- Documented DECISION_INTERVAL magic number

### Documentation
- Expanded SECURITY.md
- Added architecture diagrams to DESIGN.md
- Documented test coverage in VERIFICATION.md

### Benchmark Results
```
Model Inference: 0.125 ms (P99: 0.493 ms)
Full Roundtrip:  0.162 ms (P99: 0.270 ms)
Max throughput:  6178 decisions/sec
```

---

## Feb 1, 2026 - Comprehensive Improvement Phase

### Critical Fixes

| Issue | File | Fix |
|-------|------|-----|
| STATE_DIM mismatch | training_server.py | Changed 25→44 to match model.py |
| Double sigmoid | model.py | Replaced with tanh+rescale |
| Unnormalized money | model.py | Added /5.0 normalization |
| Slow entropy decay | training_server.py | Changed 0.999→0.995 |

### C++ Improvements

| Change | Location | Description |
|--------|----------|-------------|
| Named constants | AILearningPlayer.h | Added MLConfig namespace |
| Income tracking | AILearningPlayer.cpp | calculateIncomeRate() |
| Supply tracking | AILearningPlayer.cpp | calculateSupplyUsed() |
| Under attack flag | AILearningPlayer.cpp | calculateUnderAttack() |

---

## Jan 31, 2026 - Phase 7: Training Automation

### Auto-Skirmish Mode
Command line flags to start skirmish without GUI:
```
-autoSkirmish         Start skirmish automatically
-observer             Put local player in observer slot
-aiDifficulty <0-3>   AI difficulty
-skirmishMap <name>   Map for auto-skirmish
-noDraw               Disable rendering
-noAudio              Disable audio
-noFPSLimit           Remove frame rate cap
```

### Key Fixes
- Learning AI now selectable in Skirmish menu
- Fixed pipe name mismatch between game and self-play server
- Game auto-starts and restarts when in auto-skirmish mode

---

## Earlier Phases

### Phase 1: Skeleton Integration
Basic Learning AI class integrated into game engine.

### Phase 2: ML Bridge
Windows named pipe communication between game and Python.

### Phase 3: State Verification
Logging and analysis tools for verifying state extraction.

### Phase 4: Decision Override
Team selection with weighted random based on ML preferences.
Building classification and attack timing with aggression control.

### Phase 5: Training Pipeline
Neural network (PolicyNetwork) with PPO algorithm.
SimulatedEnv for testing, GeneralsEnv for real game.

### Phase 6: Training and Evaluation
Inference server, evaluation tools, experiment tracking.
Simulated training results: 20% → 81.5% win rate over 200 episodes.
