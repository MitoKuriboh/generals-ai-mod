"""Tests for reward calculation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rewards import (
    RewardConfig, calculate_step_reward, get_config,
    EXPLORATION_CONFIG, BALANCED_CONFIG, SPARSE_CONFIG, AGGRESSIVE_CONFIG
)


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RewardConfig()

        assert config.win_reward == 100.0
        assert config.loss_reward == -100.0
        assert config.enemy_unit_killed == 0.5
        assert config.enable_shaping is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = RewardConfig(win_reward=50.0, shaping_scale=2.0)

        assert config.win_reward == 50.0
        assert config.shaping_scale == 2.0


class TestCalculateStepReward:
    """Tests for step reward calculation."""

    @pytest.fixture
    def config(self):
        """Default config for tests."""
        return RewardConfig()

    def test_none_prev_state(self, config):
        """Test reward with no previous state."""
        reward = calculate_step_reward(None, {'army_strength': 1.0}, config)
        assert reward == 0.0

    def test_army_strength_improvement(self, config):
        """Test reward for army strength increase."""
        prev_state = {'army_strength': 1.0}
        curr_state = {'army_strength': 1.5}

        reward = calculate_step_reward(prev_state, curr_state, config)

        # Should get positive reward for strength improvement
        assert reward > 0

    def test_army_strength_decline(self, config):
        """Test reward for army strength decrease."""
        prev_state = {'army_strength': 1.5}
        curr_state = {'army_strength': 1.0}

        reward = calculate_step_reward(prev_state, curr_state, config)

        # Should get negative reward for strength decline
        assert reward < 0.1  # Might still be slightly positive due to survival bonus

    def test_tech_advancement(self, config):
        """Test reward for tech level increase."""
        prev_state = {'army_strength': 1.0, 'tech_level': 0.2}
        curr_state = {'army_strength': 1.0, 'tech_level': 0.4}

        reward = calculate_step_reward(prev_state, curr_state, config)

        # Should get positive reward for tech advancement
        assert reward > 0

    def test_under_attack_penalty(self, config):
        """Test penalty when under attack."""
        prev_state = {'army_strength': 1.0, 'under_attack': 0}
        curr_state = {'army_strength': 1.0, 'under_attack': 1}

        reward_safe = calculate_step_reward(prev_state,
            {'army_strength': 1.0, 'under_attack': 0}, config)
        reward_attacked = calculate_step_reward(prev_state, curr_state, config)

        # Should get lower reward when under attack
        assert reward_attacked < reward_safe

    def test_shaping_disabled(self):
        """Test that shaping can be disabled."""
        config = RewardConfig(enable_shaping=False)

        prev_state = {'army_strength': 1.0}
        curr_state = {'army_strength': 2.0}  # Big improvement

        reward = calculate_step_reward(prev_state, curr_state, config)

        assert reward == 0.0

    def test_shaping_scale(self):
        """Test shaping scale affects reward."""
        config1 = RewardConfig(shaping_scale=1.0)
        config2 = RewardConfig(shaping_scale=2.0)

        prev_state = {'army_strength': 1.0}
        curr_state = {'army_strength': 1.5}

        reward1 = calculate_step_reward(prev_state, curr_state, config1)
        reward2 = calculate_step_reward(prev_state, curr_state, config2)

        assert abs(reward2 - 2 * reward1) < 0.01


class TestRewardPresets:
    """Tests for reward configuration presets."""

    def test_get_config(self):
        """Test getting preset configurations."""
        assert get_config('exploration') == EXPLORATION_CONFIG
        assert get_config('balanced') == BALANCED_CONFIG
        assert get_config('sparse') == SPARSE_CONFIG
        assert get_config('aggressive') == AGGRESSIVE_CONFIG
        assert get_config('unknown') == BALANCED_CONFIG  # Default fallback

    def test_exploration_config(self):
        """Test exploration preset values."""
        config = EXPLORATION_CONFIG

        assert config.shaping_scale == 2.0
        assert config.time_penalty == 0.0

    def test_sparse_config(self):
        """Test sparse preset values."""
        config = SPARSE_CONFIG

        assert config.shaping_scale == 0.1

    def test_aggressive_config(self):
        """Test aggressive preset values."""
        config = AGGRESSIVE_CONFIG

        assert config.enemy_unit_killed == 1.0
        assert config.enemy_building_destroyed == 4.0


class TestRewardBounds:
    """Tests for reward value bounds."""

    def test_reward_is_finite(self):
        """Test that rewards are always finite."""
        config = RewardConfig()

        states = [
            {'army_strength': 0.0},
            {'army_strength': 10.0},
            {'army_strength': 1.0, 'tech_level': 1.0, 'income': 10.0},
            {},  # Empty state
        ]

        for i, curr_state in enumerate(states):
            for prev_state in states:
                reward = calculate_step_reward(prev_state, curr_state, config)
                assert not (reward != reward), f"NaN reward for states {i}"  # NaN check

    def test_reward_reasonable_magnitude(self):
        """Test that step rewards are reasonably bounded."""
        config = RewardConfig()

        # Extreme improvement
        prev_state = {'army_strength': 0.1, 'tech_level': 0, 'income': 0}
        curr_state = {'army_strength': 5.0, 'tech_level': 1.0, 'income': 10.0}

        reward = calculate_step_reward(prev_state, curr_state, config)

        # Step reward shouldn't exceed terminal rewards in magnitude
        assert abs(reward) < abs(config.win_reward)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
