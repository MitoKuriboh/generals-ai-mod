# State Verification Checklist

This document describes how to verify that the ML Bridge is working correctly and that state extraction accurately reflects the game.

## Quick Start

1. **Build the game** with the Learning AI modifications
2. **Start the Python server**:
   ```bash
   cd python
   pip install -r requirements.txt
   python ml_bridge_server.py --log states.jsonl --verbose
   ```
3. **Start the game** and begin a skirmish with Learning AI opponent
4. **Observe the server output** for received states
5. **Analyze the log** after the game:
   ```bash
   python analyze_states.py states.jsonl
   ```

## Verification Steps

### 1. Connection Test

**What to verify:** Game connects to Python server

**Expected behavior:**
- Server prints "Game connected!" when skirmish starts
- Server prints "Pipe broken, client disconnected" when game ends

**Troubleshooting:**
- If no connection: ensure server starts BEFORE the game
- Check that pipe name matches: `\\.\pipe\generals_ml_bridge`
- Check Windows firewall/security settings

### 2. State Reception Test

**What to verify:** States are received approximately every second

**Expected behavior:**
- States arrive every ~30 frames (about 1 second at 30 FPS)
- State count increases steadily during gameplay

**Troubleshooting:**
- If no states: check that Learning AI opponent was selected
- Enable DEBUG_ML_STATE in AILearningPlayer.cpp and check game logs

### 3. Economy Values

**What to verify:** Money and power values match in-game display

**Manual verification:**
1. Start game, note your starting money (usually $2000-$10000)
2. Check server output: `money` should be log10(actual_money)
   - $1000 → money ≈ 3.0
   - $5000 → money ≈ 3.7
   - $10000 → money ≈ 4.0

**Expected ranges:**
| Field | Min | Max | Notes |
|-------|-----|-----|-------|
| money | 2.0 | 5.0 | log10 scale |
| power | -100 | 200 | production - consumption |

### 4. Force Counts

**What to verify:** Unit counts match actual units on map

**Manual verification:**
1. Count your infantry units in game
2. Check server output: `own_infantry[0]` is log10(count+1)
   - 0 units → 0.0
   - 9 units → 1.0
   - 99 units → 2.0

**Expected initial values (game start):**
- `own_structures[0]` > 0 (should have command center + starting buildings)
- `own_infantry[0]` usually starts at 0 or small value
- `own_vehicles[0]` usually starts at 0

**Troubleshooting:**
- If all zeros: check isKindOf() detection in countForces()
- If too high: check object filtering (dead objects, neutral)

### 5. Strategic Values

**What to verify:** Strategic indicators make sense

**game_time:**
- Should start near 0 and increase
- At 5 minutes game time, value should be ~5.0

**tech_level:**
- Starts at 0 (no tech buildings)
- Increases as tech structures are built
- Range: 0.0 to 1.0

**base_threat:**
- Should be 0 at game start (no enemies near base)
- Increases when enemy units approach
- Range: 0.0 to 1.0

**army_strength:**
- Ratio of your army value to enemy army value
- 1.0 = equal, >1 = stronger, <1 = weaker
- At start: often 2.0 (no visible enemy army)

**under_attack:**
- 0.0 = not under attack
- 1.0 = base under attack
- Should flip to 1.0 when enemies attack your base

**distance_to_enemy:**
- Normalized distance between command centers
- Range: 0.0 to 1.0 (1.0 = far away)

### 6. Fog of War

**What to verify:** Enemy values only count visible units

**Expected behavior:**
- `enemy_*` values start low (only see enemy base area)
- Increase as you scout enemy territory
- Should never see units in unexplored areas

### 7. State Consistency

**What to verify:** Values change appropriately over time

**Expected patterns:**
- Money fluctuates (income and spending)
- Unit counts increase as you build
- base_threat spikes during attacks
- army_strength changes with battles

**Use analysis script:**
```bash
python analyze_states.py states.jsonl --plot
```

This will show graphs of key values over time.

## Common Issues

### Issue: All values are zero
- **Cause:** Player pointer is null or wrong
- **Fix:** Check m_player initialization in constructor

### Issue: Money always shows 0
- **Cause:** getMoney() access issue
- **Fix:** Verify Money class interface, check for null

### Issue: No enemy forces detected
- **Cause:** Fog of war filtering too aggressive
- **Fix:** Check shroud status threshold in countForces()

### Issue: Unit counts way too high
- **Cause:** Counting projectiles, effects, or other non-unit objects
- **Fix:** Add more filtering in countForces() loop

### Issue: base_threat always 0
- **Cause:** Command center not found, or wrong position
- **Fix:** Check KINDOF_COMMANDCENTER detection

### Issue: State never changes
- **Cause:** Update not being called, or wrong frame interval
- **Fix:** Check m_frameCounter increment, ML_DECISION_INTERVAL value

## Debug Logging

Enable detailed C++ logging by ensuring this is defined in AILearningPlayer.cpp:

```cpp
#define DEBUG_ML_STATE 1
```

This will output detailed state information to the game's debug log.

## Analysis Script

The `analyze_states.py` script provides:

1. **Summary statistics**: min/max/average for all values
2. **Issue detection**: automatically flags potential problems
3. **Time series plots**: visualize state changes over game
4. **CSV export**: for external analysis tools

Usage:
```bash
# Show summary
python analyze_states.py states.jsonl

# Show plots
python analyze_states.py states.jsonl --plot

# Save plot to file
python analyze_states.py states.jsonl --plot-file game1.png

# Export to CSV
python analyze_states.py states.jsonl --export states.csv
```

## Success Criteria

State extraction is verified when:

1. ✓ Connection establishes reliably
2. ✓ States received at ~1 second intervals
3. ✓ Money value matches in-game (within log10 accuracy)
4. ✓ Unit counts match actual units (±10%)
5. ✓ game_time increases linearly
6. ✓ Strategic values respond to game events
7. ✓ No obvious anomalies in analysis output
