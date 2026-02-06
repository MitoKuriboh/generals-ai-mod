#!/usr/bin/env python3
"""
Auto training mode.

Auto-launches the game with -autoSkirmish flags.
Handles communication and PPO training automatically.
"""

import sys
import time
from typing import Optional

from .base import (
    BaseTrainer, EpisodeResult,
    wrap_recommendation_with_capabilities, validate_protocol_version
)
from ..model import state_dict_to_tensor

# Import game launcher if available
try:
    # Add parent directories to path
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from game_launcher import GameLauncher, Episode
    HAS_LAUNCHER = True
except ImportError:
    HAS_LAUNCHER = False


class AutoTrainer(BaseTrainer):
    """
    Training handler with automatic game launching.

    Launches the game with -autoSkirmish flags and handles
    all communication and training automatically.
    """

    def __init__(
        self,
        game_path: str = None,
        map_name: str = "Alpine Assault",
        ai_difficulty: int = 0,
        headless: bool = False,
        seed: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not HAS_LAUNCHER:
            raise ImportError("game_launcher module not found")

        self.launcher = GameLauncher(
            game_path=game_path,
            ai_difficulty=ai_difficulty,
            map_name=map_name,
            headless=headless,
            seed=seed,
        )

    def setup(self) -> bool:
        """Setup is done per-episode for auto mode."""
        print(f"\nAuto Training Mode")
        print(f"  Map: {self.launcher.map_name}")
        print(f"  AI: {['Easy', 'Medium', 'Hard', 'Learning'][self.launcher.ai_difficulty]}")
        print(f"  Headless: {self.launcher.headless}")
        return True

    def cleanup(self):
        """Stop any running game."""
        try:
            self.launcher.stop_game()
        except Exception as e:
            # Game may have already stopped or crashed
            print(f"[AutoTrainer] Cleanup warning: {e}")

    def run_episode(self) -> Optional[EpisodeResult]:
        """Run a single training episode with auto-launched game."""
        # Start game
        if not self.launcher.start_game():
            return None

        states = []
        rewards = []
        self._prev_state = None

        victory = False
        game_time = 0.0
        final_army = 0.0
        game_ended = False

        try:
            while self.launcher.is_running() and not game_ended:
                # Get state from game
                state = self.launcher.get_state()

                if state is None:
                    time.sleep(0.1)
                    continue

                # Validate protocol version on first state
                if len(states) == 0:
                    validate_protocol_version(state)

                # Check for game end
                if state.get('type') == 'game_end':
                    game_ended = True
                    victory = state.get('victory', False)
                    game_time = state.get('game_time', 0.0)
                    final_army = state.get('army_strength', 0.0)
                    break

                states.append(state)

                # Calculate reward
                reward = self.calculate_reward(self._prev_state, state)
                rewards.append(reward)
                self._prev_state = state

                # Get action from PPO policy
                recommendation = self.get_recommendation(state)

                # Store transition in PPO buffer
                self.store_transition(state, reward, done=False)

                # Send to game
                wrapped = wrap_recommendation_with_capabilities(recommendation)
                self.launcher.send_recommendation(wrapped)

                # PPO update every 256 steps
                self.maybe_update(state, threshold=256)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.launcher.stop_game()
            time.sleep(2)  # Brief pause between episodes

        if not game_ended:
            return None

        # Store terminal transition
        if states and self._current_action is not None:
            self.store_terminal_transition(states[-1], victory)

        # Final PPO update
        self.final_update()

        episode_reward = sum(rewards)
        if victory:
            episode_reward += 100.0
        else:
            episode_reward -= 100.0

        return EpisodeResult(
            victory=victory,
            game_time=game_time,
            final_army_strength=final_army,
            steps=len(states),
            reward=episode_reward,
        )
