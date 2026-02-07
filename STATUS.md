# Project Status

## Current Phase: Phase 8 - Advanced RL Techniques

Major upgrade to neural network architecture and training pipeline. Added 2025 research techniques.

## What Works
- Strategic layer training (PPO) - 81.5% win rate after 350 episodes
- Game integration via named pipes
- Learning AI selectable in Skirmish menu
- Auto-skirmish mode for headless training
- Hierarchical RL architecture (Tactical + Micro layers)
- Comprehensive test suite (53 tests passing)

### New Features (Phase 8)
- **RND Curiosity Exploration** - Intrinsic motivation for novel states
- **Tactical Attention Layer** - Multi-head attention over unit features
- **Strategic Transformer** - Decision Transformer for sequence modeling
- **Mixture of Experts (MoE)** - 4 specialized strategy experts with learned routing
- **GRPO Training** - Critic-free policy optimization
- **Adaptive Curriculum Learning** - Auto-adjusting difficulty based on win rate
- **Improved Staleness Detection** - Faster heartbeat protocol (1.5s timeout)

## What Doesn't
- Real game training not yet tested at scale
- Hierarchical layers trained only in simulation
- Some placeholder values in C++ state builders

## Key Metrics
| Metric | Value |
|--------|-------|
| Simulated Win Rate | 81.5% |
| Episodes Trained | 350 |
| Test Coverage | 53 tests passing |
| Model Latency | 0.125 ms (P99: 0.493 ms) |

## New Files Added
| File | Purpose |
|------|---------|
| `python/training/curiosity.py` | RND curiosity-driven exploration |
| `python/training/grpo.py` | GRPO critic-free trainer |
| `python/training/model.py` | Added StrategicTransformer + StrategicMoE |
| `python/tactical/model.py` | Added UnitAttention layer |
| `python/hierarchical/train_joint.py` | Added AdaptiveCurriculum |

## Architecture Summary

### Strategic Layer Options
1. **PolicyNetwork** (default) - 2-layer MLP with Beta distributions
2. **StrategicTransformer** - Decision Transformer over game history
3. **StrategicMoE** - 4 experts (Aggro/Defense/Economy/Tech)

### Training Options
1. **PPO** (default) - Actor-critic with GAE
2. **GRPO** - Critic-free group-relative optimization
3. **+Curiosity** - Add RND intrinsic rewards

### Tactical Layer
- TacticalNetwork with optional multi-head attention
- Attention weights unit groups (infantry, vehicles, aircraft, mixed)

## Build Status
- **Last Build:** Feb 7, 2026
- **Location:** `C:\Program Files (x86)\Steam\steamapps\common\Command & Conquer Generals - Zero Hour\`
- **Status:** SUCCESSFUL

## Immediate Next Steps
1. Test new architecture variants in simulation
2. Compare: PolicyNetwork vs MoE vs Transformer
3. Enable curriculum learning for harder opponents
4. Test 10-game stability with real game

## Milestones
- [x] M1-M7: Infrastructure complete
- [x] M8a: Advanced RL techniques implemented
- [ ] M8b: Benchmark new architectures
- [ ] M9: Learning AI beats Easy AI >80%
- [ ] M10: Learning AI beats Hard AI >50%
