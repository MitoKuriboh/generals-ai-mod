# ML Bridge Protocol Documentation

Communication protocol between C&C Generals Zero Hour (C++) and Python ML system.

## Overview

The ML Bridge uses Windows Named Pipes for bidirectional communication. The protocol supports both:
- **Strategic-only mode**: Simple state/recommendation exchange
- **Hierarchical mode**: Batched state for strategic, tactical, and micro layers

## Connection

**Pipe Name:** `\\.\pipe\generals_ml_bridge`

The Python server creates the pipe and waits for the game to connect. The game (client) connects when the Learning AI is activated.

**Protocol:** Length-prefixed messages
- 4-byte little-endian length prefix
- UTF-8 JSON payload

## Protocol Version

Current version: **2**

Both sides should validate the protocol version on first message:
- C++: `ML_PROTOCOL_VERSION` in MLBridge.cpp
- Python: `PROTOCOL_VERSION` in training/config.py

Version mismatches generate warnings but don't fail (allows testing with slight differences).

## Message Types

### Game State (Game -> Python)

**Strategic State (flat format):**
```json
{
  "version": 2,
  "player": 3,
  "money": 3.45,
  "power": 50.0,
  "income": 5.0,
  "supply": 0.6,
  "own_infantry": [1.0, 0.8, 0.0],
  "own_vehicles": [0.7, 0.9, 0.0],
  "own_aircraft": [0.3, 1.0, 0.0],
  "own_structures": [1.0, 0.95, 0.0],
  "enemy_infantry": [0.8, 0.7, 0.0],
  "enemy_vehicles": [0.5, 0.8, 0.0],
  "enemy_aircraft": [0.2, 1.0, 0.0],
  "enemy_structures": [0.9, 0.9, 0.0],
  "game_time": 5.0,
  "tech_level": 0.7,
  "base_threat": 0.3,
  "army_strength": 1.2,
  "under_attack": 0.0,
  "distance_to_enemy": 0.6,
  "is_usa": 1.0,
  "is_china": 0.0,
  "is_gla": 0.0
}
```

**Field Descriptions:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| version | int | 2 | Protocol version |
| player | int | 0-7 | Player index |
| money | float | 0-5 | log10 of current money ($1000 = 3.0) |
| power | float | -100 to 200 | Power balance (surplus/deficit) |
| income | float | 0-20 | Income rate per second |
| supply | float | 0-1 | Supply usage fraction |
| own_* | [3]float | | [count_log10, avg_health, dps_ratio] |
| enemy_* | [3]float | | [count_log10, avg_health, dps_ratio] |
| game_time | float | 0-60+ | Game time in minutes |
| tech_level | float | 0-1 | Research/upgrade progress |
| base_threat | float | 0-1 | Enemy proximity to base |
| army_strength | float | 0-10+ | Ratio of own to enemy army value |
| under_attack | float | 0/1 | 1 if base under attack |
| distance_to_enemy | float | 0-1 | Normalized distance to enemy base |
| is_usa/china/gla | float | 0/1 | Faction one-hot encoding |

**Batched State (hierarchical format):**
```json
{
  "frame": 1234,
  "player_id": 3,
  "strategic": { /* same as flat format fields */ },
  "teams": [
    {"id": 1, "state": [64 floats...]},
    {"id": 2, "state": [64 floats...]}
  ],
  "units": [
    {"id": 101, "state": [32 floats...]},
    {"id": 102, "state": [32 floats...]}
  ]
}
```

### Game End (Game -> Python)

```json
{
  "type": "game_end",
  "victory": true,
  "game_time": 12.5,
  "army_strength": 2.3
}
```

### Recommendation (Python -> Game)

**Strategic Recommendation:**
```json
{
  "version": 2,
  "capabilities": {
    "hierarchical": false,
    "tactical": false,
    "micro": false
  },
  "priority_economy": 0.25,
  "priority_defense": 0.15,
  "priority_military": 0.45,
  "priority_tech": 0.15,
  "prefer_infantry": 0.3,
  "prefer_vehicles": 0.4,
  "prefer_aircraft": 0.3,
  "aggression": 0.7,
  "target_player": -1
}
```

**Field Descriptions:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| priority_* | float | 0-1 | Build priority weights (should sum to 1) |
| prefer_* | float | 0-1 | Army composition preferences (should sum to 1) |
| aggression | float | 0-1 | Attack vs defend tendency |
| target_player | int | -1 to 7 | -1 = auto-select target |

**Capabilities Object:**

| Field | Type | Description |
|-------|------|-------------|
| hierarchical | bool | Server supports batched tactical/micro |
| tactical | bool | Server has tactical model loaded |
| micro | bool | Server has micro model loaded |

**Batched Response (hierarchical mode):**
```json
{
  "frame": 1234,
  "version": 2,
  "capabilities": {
    "hierarchical": true,
    "tactical": true,
    "micro": false
  },
  "strategic": { /* recommendation fields */ },
  "teams": [
    {"id": 1, "action": 0, "x": 0.6, "y": 0.5, "attitude": 0.8}
  ],
  "units": [
    {"id": 101, "action": 5, "angle": -0.5, "dist": 0.3}
  ]
}
```

### Tactical Commands

| Action | Value | Description |
|--------|-------|-------------|
| HOLD | 0 | Hold position |
| ATTACK | 1 | Attack toward (x, y) |
| DEFEND | 2 | Defend position (x, y) |
| RETREAT | 3 | Retreat to (x, y) |
| REGROUP | 4 | Regroup at (x, y) |
| FLANK_LEFT | 5 | Flanking maneuver left |
| FLANK_RIGHT | 6 | Flanking maneuver right |
| AMBUSH | 7 | Set up ambush at (x, y) |
| SUPPORT | 8 | Support other teams |
| SCOUT | 9 | Scout toward (x, y) |

Fields: `id`, `action`, `x` (0-1), `y` (0-1), `attitude` (0-1 aggression)

### Micro Commands

| Action | Value | Description |
|--------|-------|-------------|
| FOLLOW_TEAM | 0 | Follow team orders |
| ATTACK_TARGET | 1 | Attack specific target |
| RETREAT | 2 | Retreat in direction |
| KITE | 3 | Kite (attack-move backward) |
| FLANK | 4 | Flank to angle |
| FOCUS_FIRE | 5 | Focus fire with group |
| USE_ABILITY | 6 | Use special ability |

Fields: `id`, `action`, `angle` (-1 to 1, maps to -pi to pi), `dist` (0-1)

## State Dimensions

**Strategic State:** 44 floats (STATE_DIM in config.py)
**Tactical State:** 64 floats (TacticalState::DIM)
**Micro State:** 32 floats (MicroState::DIM)

**Action Dimensions:**
- Strategic: 8 continuous values (ACTION_DIM)
- Tactical: 4 values (action one-hot expandable, x, y, attitude)
- Micro: 3 values (action one-hot expandable, angle, distance)

## Version History

| Version | Changes |
|---------|---------|
| 1 | Initial protocol - strategic only |
| 2 | Added hierarchical support, capabilities negotiation, batched format |

## Error Handling

- Invalid JSON: Log error, skip message
- Protocol mismatch: Log warning, continue (allows testing)
- Missing fields: Use defaults
- Out-of-range values: Clamp to valid range
- NaN/Inf values: Replace with defaults

## Files

**C++:**
- `MLBridge.cpp` - Main communication class
- `MLBridge.h` - Struct definitions
- `TacticalState.cpp/h` - Team state encoding
- `MicroState.cpp/h` - Unit state encoding

**Python:**
- `training/config.py` - Protocol version, constants
- `training/env.py` - Game environment wrapper
- `ml_bridge_server.py` - Simple strategic server
- `ml_inference_server.py` - PPO inference server
- `servers/hierarchical_server.py` - Full hierarchical server
- `hierarchical/batch_bridge.py` - Batched protocol handler

## Testing

```bash
# Run Python server first
python -m servers.hierarchical_server --strategic checkpoints/best_agent.pt

# Or simple test server
python ml_bridge_server.py --verbose

# Then start game with Learning AI opponent
```
