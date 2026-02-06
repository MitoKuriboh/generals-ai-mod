#!/usr/bin/env python3
"""
Simulated training mode.

Uses SimulatedEnv for fast testing without the real game.
"""

from typing import Optional

from .base import BaseTrainer, EpisodeResult
from ..env import SimulatedEnv


class SimulatedTrainer(BaseTrainer):
    """
    Training handler using simulated environment.

    Uses SimulatedEnv for fast testing without the real game.
    Good for validating pipeline changes and hyperparameter tuning.
    """

    def __init__(
        self,
        episode_length: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.episode_length = episode_length
        self.env = None

    def setup(self) -> bool:
        """Create simulated environment."""
        print(f"\nSimulated Training Mode")
        print(f"  Episode Length: {self.episode_length} steps")
        self.env = SimulatedEnv(episode_length=self.episode_length)
        return True

    def cleanup(self):
        """Clean up environment."""
        if self.env:
            self.env.close()
            self.env = None

    def run_episode(self) -> Optional[EpisodeResult]:
        """Run a single episode in simulated environment."""
        state, info = self.env.reset()
        episode_reward = 0.0
        done = False
        step_count = 0

        while not done:
            # Select action
            action, log_prob, value = self.agent.select_action(state)

            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step_count += 1

            # Store transition
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            episode_reward += reward
            state = next_state

            # PPO update when enough steps collected
            if len(self.agent.buffer) >= 256:
                import torch
                with torch.no_grad():
                    _, _, last_value = self.agent.select_action(state)
                self.agent.update(last_value)

        # Get episode stats from info
        episode_stats = info.get('episode_stats')
        if episode_stats:
            return EpisodeResult(
                victory=episode_stats.won if episode_stats.won is not None else False,
                game_time=episode_stats.game_time,
                final_army_strength=episode_stats.final_army_strength,
                steps=step_count,
                reward=episode_reward,
            )

        # Fallback if no stats
        return EpisodeResult(
            victory=episode_reward > 0,
            game_time=step_count / 10.0,
            final_army_strength=1.0,
            steps=step_count,
            reward=episode_reward,
        )
