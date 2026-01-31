"""Tests for PolicyNetwork model."""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import (
    PolicyNetwork, STATE_DIM, TOTAL_ACTION_DIM,
    state_dict_to_tensor, action_tensor_to_dict
)


class TestPolicyNetwork:
    """Tests for the PolicyNetwork class."""

    @pytest.fixture
    def model(self):
        """Create a fresh model for each test."""
        return PolicyNetwork()

    def test_model_creation(self, model):
        """Test model initializes correctly."""
        assert model.state_dim == STATE_DIM
        assert model.action_dim == TOTAL_ACTION_DIM
        # Check parameter count is reasonable
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 10000  # Should have substantial parameters

    def test_action_bounds(self, model):
        """Test that all actions are in [0, 1] range."""
        for _ in range(100):
            state = torch.randn(STATE_DIM)
            action, log_prob, value = model.get_action(state)

            assert action.shape == (TOTAL_ACTION_DIM,)
            assert torch.all(action >= 0), f"Action below 0: {action}"
            assert torch.all(action <= 1), f"Action above 1: {action}"

    def test_action_bounds_batch(self, model):
        """Test action bounds with batch input."""
        states = torch.randn(32, STATE_DIM)
        for i in range(32):
            action, _, _ = model.get_action(states[i])
            assert torch.all(action >= 0) and torch.all(action <= 1)

    def test_log_prob_finite(self, model):
        """Test that log probabilities are always finite."""
        for _ in range(100):
            state = torch.randn(STATE_DIM)
            action, log_prob, _ = model.get_action(state)

            assert torch.isfinite(log_prob), f"Log prob not finite: {log_prob}"
            # Log probs should be negative (probabilities < 1)
            # For Beta distribution, can be positive for peaked distributions

    def test_deterministic_action(self, model):
        """Test deterministic action selection."""
        state = torch.randn(STATE_DIM)

        action1, _, _ = model.get_action(state, deterministic=True)
        action2, _, _ = model.get_action(state, deterministic=True)

        assert torch.allclose(action1, action2), "Deterministic actions should be identical"

    def test_value_estimate(self, model):
        """Test value estimate is a scalar."""
        state = torch.randn(STATE_DIM)
        _, _, value = model.get_action(state)

        assert value.dim() == 0 or (value.dim() == 1 and value.size(0) == 1)

    def test_evaluate_actions(self, model):
        """Test action evaluation for PPO update."""
        states = torch.randn(32, STATE_DIM)
        actions = torch.rand(32, TOTAL_ACTION_DIM)  # Random valid actions

        log_probs, values, entropy = model.evaluate_actions(states, actions)

        assert log_probs.shape == (32,)
        assert values.shape == (32,)
        assert entropy.shape == (32,)
        assert torch.all(torch.isfinite(log_probs))
        assert torch.all(torch.isfinite(values))
        # Note: Beta distribution entropy can be negative for peaked distributions (alpha,beta > 1)
        assert torch.all(torch.isfinite(entropy))

    def test_forward_output_shapes(self, model):
        """Test forward pass output shapes."""
        state = torch.randn(1, STATE_DIM)
        alpha, beta, value = model.forward(state)

        assert alpha.shape == (1, TOTAL_ACTION_DIM)
        assert beta.shape == (1, TOTAL_ACTION_DIM)
        assert value.shape == (1, 1)

        # Beta distribution parameters should be positive
        assert torch.all(alpha > 0)
        assert torch.all(beta > 0)

    def test_save_load(self, model, tmp_path):
        """Test model save and load."""
        path = str(tmp_path / "test_model.pt")

        # Get action with original model
        state = torch.randn(STATE_DIM)
        action1, _, _ = model.get_action(state, deterministic=True)

        # Save and load
        model.save(path)
        loaded_model = PolicyNetwork.load(path)

        # Get action with loaded model
        action2, _, _ = loaded_model.get_action(state, deterministic=True)

        assert torch.allclose(action1, action2), "Loaded model should produce same actions"


class TestStateConversion:
    """Tests for state conversion functions."""

    def test_state_dict_to_tensor(self):
        """Test conversion of game state dict to tensor."""
        state = {
            'money': 3.0,
            'power': 50.0,
            'income': 2.0,
            'supply': 0.5,
            'own_infantry': [0.5, 0.8, 0.0],
            'own_vehicles': [0.3, 0.6, 0.0],
            'own_aircraft': [0.0, 0.0, 0.0],
            'own_structures': [0.7, 0.9, 0.0],
            'enemy_infantry': [0.4, 0.7, 0.0],
            'enemy_vehicles': [0.3, 0.5, 0.0],
            'enemy_aircraft': [0.0, 0.0, 0.0],
            'enemy_structures': [0.6, 0.8, 0.0],
            'game_time': 5.0,
            'tech_level': 0.3,
            'base_threat': 0.1,
            'army_strength': 1.2,
            'under_attack': 0,
            'distance_to_enemy': 0.5,
            'is_usa': 1,
            'is_china': 0,
            'is_gla': 0,
        }

        tensor = state_dict_to_tensor(state)

        assert tensor.shape == (STATE_DIM,)
        assert tensor.dtype == torch.float32
        assert torch.all(torch.isfinite(tensor))

    def test_state_dict_missing_keys(self):
        """Test handling of missing state keys."""
        # Minimal state
        state = {}
        tensor = state_dict_to_tensor(state)

        assert tensor.shape == (STATE_DIM,)
        assert torch.all(torch.isfinite(tensor))

    def test_action_tensor_to_dict(self):
        """Test conversion of action tensor to recommendation dict."""
        action = torch.tensor([0.3, 0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.7])

        rec = action_tensor_to_dict(action)

        # Check all required keys exist
        required_keys = [
            'priority_economy', 'priority_defense', 'priority_military', 'priority_tech',
            'prefer_infantry', 'prefer_vehicles', 'prefer_aircraft', 'aggression'
        ]
        for key in required_keys:
            assert key in rec
            assert isinstance(rec[key], float)

        # Check priorities sum to ~1
        priority_sum = (rec['priority_economy'] + rec['priority_defense'] +
                       rec['priority_military'] + rec['priority_tech'])
        assert abs(priority_sum - 1.0) < 0.01

        # Check army prefs sum to ~1
        army_sum = rec['prefer_infantry'] + rec['prefer_vehicles'] + rec['prefer_aircraft']
        assert abs(army_sum - 1.0) < 0.01

        # Check aggression in [0, 1]
        assert 0 <= rec['aggression'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
