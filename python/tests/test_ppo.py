"""Tests for PPO algorithm and RolloutBuffer."""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ppo import PPOAgent, PPOConfig, RolloutBuffer
from training.model import STATE_DIM, TOTAL_ACTION_DIM


class TestRolloutBuffer:
    """Tests for the RolloutBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create a fresh buffer for each test."""
        return RolloutBuffer()

    def test_add_transition(self, buffer):
        """Test adding transitions to buffer."""
        state = torch.randn(STATE_DIM)
        action = torch.rand(TOTAL_ACTION_DIM)
        reward = 1.0
        value = torch.tensor([0.5])
        log_prob = torch.tensor(-1.0)
        done = False

        buffer.add(state, action, reward, value, log_prob, done)

        assert len(buffer) == 1
        assert len(buffer.states) == 1
        assert len(buffer.rewards) == 1

    def test_compute_returns_and_advantages(self, buffer):
        """Test GAE computation."""
        # Add several transitions
        for i in range(10):
            buffer.add(
                state=torch.randn(STATE_DIM),
                action=torch.rand(TOTAL_ACTION_DIM),
                reward=float(i) * 0.1,
                value=torch.tensor([0.5]),
                log_prob=torch.tensor(-1.0),
                done=(i == 9)
            )

        last_value = torch.tensor([0.0])
        buffer.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert len(buffer.advantages) == 10
        assert len(buffer.returns) == 10
        assert torch.all(torch.isfinite(buffer.advantages))
        assert torch.all(torch.isfinite(buffer.returns))

    def test_get_batches(self, buffer):
        """Test minibatch generation."""
        # Add transitions
        for i in range(64):
            buffer.add(
                state=torch.randn(STATE_DIM),
                action=torch.rand(TOTAL_ACTION_DIM),
                reward=0.1,
                value=torch.tensor([0.5]),
                log_prob=torch.tensor(-1.0),
                done=False
            )

        buffer.compute_returns_and_advantages(torch.tensor([0.0]), 0.99, 0.95)

        # Generate batches
        batches = list(buffer.get_batches(batch_size=16, num_minibatches=4))

        assert len(batches) == 4  # 64 / 4 = 16 per batch

        for batch in batches:
            assert 'states' in batch
            assert 'actions' in batch
            assert 'old_log_probs' in batch
            assert 'old_values' in batch
            assert 'advantages' in batch
            assert 'returns' in batch

    def test_clear(self, buffer):
        """Test buffer clearing."""
        buffer.add(
            torch.randn(STATE_DIM),
            torch.rand(TOTAL_ACTION_DIM),
            1.0,
            torch.tensor([0.5]),
            torch.tensor(-1.0),
            False
        )

        assert len(buffer) == 1

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.advantages is None
        assert buffer.returns is None


class TestPPOAgent:
    """Tests for the PPOAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        config = PPOConfig(num_epochs=2, num_minibatches=2)
        return PPOAgent(config, device='cpu')

    def test_select_action(self, agent):
        """Test action selection."""
        state = torch.randn(STATE_DIM)
        action, log_prob, value = agent.select_action(state)

        assert action.shape == (TOTAL_ACTION_DIM,)
        assert torch.all(action >= 0) and torch.all(action <= 1)
        assert torch.isfinite(log_prob)

    def test_store_transition(self, agent):
        """Test storing transitions."""
        state = torch.randn(STATE_DIM)
        action, log_prob, value = agent.select_action(state)

        agent.store_transition(state, action, 1.0, value, log_prob, done=False)

        assert len(agent.buffer) == 1
        assert agent.total_steps == 1

    def test_update(self, agent):
        """Test PPO update."""
        # Collect some transitions
        for i in range(100):
            state = torch.randn(STATE_DIM)
            action, log_prob, value = agent.select_action(state)
            agent.store_transition(state, action, 0.1, value, log_prob, done=(i == 99))

        # Perform update
        stats = agent.update(torch.tensor([0.0]))

        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats
        assert len(agent.buffer) == 0  # Buffer should be cleared

    def test_save_load(self, agent, tmp_path):
        """Test agent save and load."""
        path = str(tmp_path / "test_agent.pt")

        # Collect and update to change weights
        for i in range(50):
            state = torch.randn(STATE_DIM)
            action, log_prob, value = agent.select_action(state)
            agent.store_transition(state, action, 0.1, value, log_prob, done=False)
        agent.update(torch.tensor([0.0]))

        # Get action with trained agent
        test_state = torch.randn(STATE_DIM)
        action1, _, _ = agent.select_action(test_state)

        # Save
        agent.save(path)

        # Load into new agent
        new_agent = PPOAgent(PPOConfig(), device='cpu')
        new_agent.load(path)

        # Get action with loaded agent (deterministic for comparison)
        action2, _, _ = new_agent.policy.get_action(test_state, deterministic=True)
        action1_det, _, _ = agent.policy.get_action(test_state, deterministic=True)

        assert torch.allclose(action1_det, action2)

    def test_episode_counting(self, agent):
        """Test episode tracking."""
        # One complete episode
        for i in range(10):
            state = torch.randn(STATE_DIM)
            action, log_prob, value = agent.select_action(state)
            agent.store_transition(state, action, 0.1, value, log_prob, done=(i == 9))

        assert agent.total_steps == 10
        assert agent.total_episodes == 1


class TestPPOConfig:
    """Tests for PPO configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PPOConfig()

        assert config.lr == 3e-4
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.entropy_coef == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = PPOConfig(lr=1e-3, gamma=0.95, entropy_coef=0.02)

        assert config.lr == 1e-3
        assert config.gamma == 0.95
        assert config.entropy_coef == 0.02


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
