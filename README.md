# Generals Zero Hour - Learning AI Mod

A new AI difficulty level for C&C Generals Zero Hour that uses machine learning instead of scripted behavior.

## Goal

Create a 4th AI difficulty ("Learning" or "Adaptive") that:
- Learns from gameplay rather than following fixed scripts
- Integrates directly with the game engine (not external screen capture)
- Uses the same game state access as existing Easy/Medium/Hard AI
- Can be trained and improved over time

## Approach

Unlike the external agent approach (screen capture + input simulation), this mod works **inside** the game:
- Direct access to game state (units, resources, fog of war, etc.)
- No perception problem - perfect information about what the AI "sees"
- Can issue commands through the existing AI command interface
- Builds on the existing `AISkirmishPlayer` architecture

## Project Structure

```
generals-ai-mod/
├── GeneralsMD/          # Zero Hour source (our target)
├── Generals/            # Base game source (reference)
├── docs/
│   └── RESEARCH.md      # Findings about the AI system
├── DESIGN.md            # Architecture decisions
├── STATE.md             # Current progress
├── README.md            # This file
└── README_EA.md         # Original EA readme (build instructions)
```

## Key Source Files

- `GeneralsMD/Code/GameEngine/Source/GameLogic/AI/AISkirmishPlayer.cpp` - Skirmish AI implementation
- `GeneralsMD/Code/GameEngine/Source/GameLogic/AI/AIPlayer.cpp` - Base AI player class
- `GeneralsMD/Code/GameEngine/Include/GameLogic/AI.h` - AI system definitions

## Building

See `README_EA.md` for original build instructions. Summary:
- Requires Visual Studio (VS6 for binary matching, or modern VS with code changes)
- Multiple third-party dependencies (DirectX SDK, STLport, etc.)
- Build outputs to `/Run/` directory

## License

Based on EA's GPLv3 release of the C&C Generals source code (February 2025).
