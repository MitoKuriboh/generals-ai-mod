"""Tests for game-Python communication protocol."""

import pytest
import json
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import state_dict_to_tensor, action_tensor_to_dict


class TestJsonProtocol:
    """Tests for JSON message serialization."""

    def test_state_json_parsing(self):
        """Test parsing of game state JSON."""
        # Example JSON from C++ MLBridge::stateToJson()
        json_str = '''
        {
            "player": 3,
            "money": 3.50,
            "power": 50.00,
            "income": 2.50,
            "supply": 0.35,
            "own_infantry": [0.70, 0.85, 0.00],
            "own_vehicles": [0.48, 0.78, 0.00],
            "own_aircraft": [0.00, 0.00, 0.00],
            "own_structures": [0.70, 0.95, 0.00],
            "enemy_infantry": [0.60, 0.80, 0.00],
            "enemy_vehicles": [0.48, 0.75, 0.00],
            "enemy_aircraft": [0.00, 0.00, 0.00],
            "enemy_structures": [0.60, 0.90, 0.00],
            "game_time": 5.50,
            "tech_level": 0.40,
            "base_threat": 0.10,
            "army_strength": 1.20,
            "under_attack": 0.0,
            "distance_to_enemy": 0.65,
            "is_usa": 1.0,
            "is_china": 0.0,
            "is_gla": 0.0
        }
        '''

        state = json.loads(json_str)

        # Should parse without error
        assert state['player'] == 3
        assert abs(state['money'] - 3.50) < 0.01
        assert len(state['own_infantry']) == 3

    def test_state_to_tensor_roundtrip(self):
        """Test state dict to tensor conversion."""
        state = {
            'player': 3,
            'money': 3.5,
            'power': 50.0,
            'income': 2.5,
            'supply': 0.35,
            'own_infantry': [0.7, 0.85, 0.0],
            'own_vehicles': [0.48, 0.78, 0.0],
            'own_aircraft': [0.0, 0.0, 0.0],
            'own_structures': [0.7, 0.95, 0.0],
            'enemy_infantry': [0.6, 0.8, 0.0],
            'enemy_vehicles': [0.48, 0.75, 0.0],
            'enemy_aircraft': [0.0, 0.0, 0.0],
            'enemy_structures': [0.6, 0.9, 0.0],
            'game_time': 5.5,
            'tech_level': 0.4,
            'base_threat': 0.1,
            'army_strength': 1.2,
            'under_attack': 0.0,
            'distance_to_enemy': 0.65,
            'is_usa': 1.0,
            'is_china': 0.0,
            'is_gla': 0.0,
        }

        tensor = state_dict_to_tensor(state)

        assert tensor.shape[0] == 44  # STATE_DIM
        assert tensor.dtype.is_floating_point

    def test_recommendation_json_format(self):
        """Test recommendation dictionary format for C++ parsing."""
        import torch

        action = torch.tensor([0.3, 0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.7])
        rec = action_tensor_to_dict(action)

        # Serialize to JSON
        json_str = json.dumps(rec)

        # Parse back
        parsed = json.loads(json_str)

        # Check expected keys (C++ parseJsonFloat looks for these)
        expected_keys = [
            'priority_economy', 'priority_defense', 'priority_military', 'priority_tech',
            'prefer_infantry', 'prefer_vehicles', 'prefer_aircraft', 'aggression'
        ]
        for key in expected_keys:
            assert key in parsed
            assert isinstance(parsed[key], (int, float))

    def test_game_end_message_parsing(self):
        """Test parsing of game end message from C++."""
        # Format from MLBridge::sendGameEnd()
        json_str = '''
        {
            "type": "game_end",
            "victory": true,
            "game_time": 12.50,
            "army_strength": 1.85
        }
        '''

        msg = json.loads(json_str)

        assert msg['type'] == 'game_end'
        assert msg['victory'] is True
        assert abs(msg['game_time'] - 12.5) < 0.01


class TestMessageFraming:
    """Tests for message length-prefix framing."""

    def test_length_prefix_format(self):
        """Test 4-byte little-endian length prefix."""
        message = b'{"test": "data"}'
        length = len(message)

        # Pack as unsigned int, little endian (matches C++ WriteFile)
        prefix = struct.pack('<I', length)

        assert len(prefix) == 4

        # Unpack and verify
        unpacked_length = struct.unpack('<I', prefix)[0]
        assert unpacked_length == length

    def test_message_framing_roundtrip(self):
        """Test full message framing roundtrip."""
        original = {"priority_economy": 0.25, "aggression": 0.5}

        # Serialize
        json_bytes = json.dumps(original).encode('utf-8')
        length_prefix = struct.pack('<I', len(json_bytes))
        framed_message = length_prefix + json_bytes

        # Deserialize
        received_length = struct.unpack('<I', framed_message[:4])[0]
        received_json = framed_message[4:4 + received_length].decode('utf-8')
        parsed = json.loads(received_json)

        assert parsed == original


class TestEdgeCases:
    """Tests for protocol edge cases."""

    def test_missing_optional_fields(self):
        """Test handling of missing optional state fields."""
        # Minimal state (C++ might send incomplete data on error)
        minimal_state = {
            'player': 0,
            'money': 0.0,
        }

        tensor = state_dict_to_tensor(minimal_state)

        # Should not crash, should pad missing fields
        assert tensor.shape[0] == 44

    def test_extra_unknown_fields(self):
        """Test handling of extra unknown fields in state."""
        state = {
            'money': 3.0,
            'unknown_future_field': 999,
            'another_unknown': [1, 2, 3],
        }

        # Should ignore unknown fields
        tensor = state_dict_to_tensor(state)
        assert tensor.shape[0] == 44

    def test_json_special_characters(self):
        """Test JSON with special float values."""
        # C++ sprintf might produce these in edge cases
        state = {
            'money': 0.0,
            'power': -0.0,
            'income': 0.00001,  # Very small
            'supply': 0.99999,  # Nearly 1
        }

        # Should handle gracefully
        tensor = state_dict_to_tensor(state)
        import torch
        assert torch.all(torch.isfinite(tensor))


class TestBatchedProtocol:
    """Tests for batched protocol format (teams/units in single request)."""

    def test_batched_request_parsing(self):
        """Test parsing of batched request from C++."""
        # Format from MLBridge::batchedRequestToJson()
        json_str = '''
        {
            "type": "batched_request",
            "frame": 1000,
            "strategic": {
                "money": 3.5,
                "power": 50.0,
                "army_strength": 1.2
            },
            "teams": {
                "1": [0.5, 0.5, 0.8, 0.7, 0.3, 0.2, 0.1, 0.0,
                      0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "2": [0.6, 0.4, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1,
                      0.5, 0.4, 0.3, 0.2, 0.6, 0.5, 0.4, 0.3,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "units": {
                "101": [0.3, 0.5, 0.0, 0.2, 0.8, 0.6, 0.4, 0.3,
                        0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
        '''

        msg = json.loads(json_str)

        assert msg['type'] == 'batched_request'
        assert msg['frame'] == 1000
        assert 'strategic' in msg
        assert 'teams' in msg
        assert 'units' in msg

        # Verify team data is 64-float array
        assert '1' in msg['teams']
        assert len(msg['teams']['1']) == 64

        # Verify unit data is 32-float array
        assert '101' in msg['units']
        assert len(msg['units']['101']) == 32

    def test_batched_response_format(self):
        """Test batched response format sent to C++."""
        # Format expected by MLBridge::parseBatchedResponse()
        response = {
            'strategic': {
                'priority_economy': 0.25,
                'priority_defense': 0.15,
                'priority_military': 0.45,
                'priority_tech': 0.15,
                'prefer_infantry': 0.33,
                'prefer_vehicles': 0.34,
                'prefer_aircraft': 0.33,
                'aggression': 0.6,
                'target_player': -1
            },
            'teams': [
                {'id': 1, 'action': 0, 'x': 0.7, 'y': 0.5, 'attitude': 2},
                {'id': 2, 'action': 1, 'x': 0.3, 'y': 0.6, 'attitude': 1}
            ],
            'units': [
                {'id': 101, 'action': 3, 'angle': 1.57, 'dist': 0.5}
            ]
        }

        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        # Verify strategic has required keys
        strategic_keys = ['priority_economy', 'priority_defense', 'priority_military',
                         'priority_tech', 'prefer_infantry', 'prefer_vehicles',
                         'prefer_aircraft', 'aggression']
        for key in strategic_keys:
            assert key in parsed['strategic']

        # Verify team commands have required keys
        assert len(parsed['teams']) == 2
        for team_cmd in parsed['teams']:
            assert 'id' in team_cmd
            assert 'action' in team_cmd
            assert 'x' in team_cmd
            assert 'y' in team_cmd
            assert 'attitude' in team_cmd

        # Verify unit commands have required keys
        assert len(parsed['units']) == 1
        for unit_cmd in parsed['units']:
            assert 'id' in unit_cmd
            assert 'action' in unit_cmd
            assert 'angle' in unit_cmd
            assert 'dist' in unit_cmd

    def test_batched_empty_teams_and_units(self):
        """Test batched response with empty teams/units lists."""
        response = {
            'strategic': {
                'priority_economy': 0.25,
                'aggression': 0.5
            },
            'teams': [],
            'units': []
        }

        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        assert parsed['teams'] == []
        assert parsed['units'] == []

    def test_batched_request_empty_collections(self):
        """Test parsing batched request with no teams or units."""
        json_str = '''
        {
            "type": "batched_request",
            "frame": 500,
            "strategic": {"money": 2.0},
            "teams": {},
            "units": {}
        }
        '''

        msg = json.loads(json_str)

        assert msg['type'] == 'batched_request'
        assert len(msg['teams']) == 0
        assert len(msg['units']) == 0

    def test_batched_team_command_action_types(self):
        """Test all valid tactical action types in batched response."""
        # TacticalAction enum values: 0-7
        for action_type in range(8):
            response = {
                'teams': [
                    {'id': 1, 'action': action_type, 'x': 0.5, 'y': 0.5, 'attitude': 0}
                ]
            }
            json_str = json.dumps(response)
            parsed = json.loads(json_str)
            assert parsed['teams'][0]['action'] == action_type

    def test_batched_unit_command_action_types(self):
        """Test all valid micro action types in batched response."""
        # MicroAction enum values: 0-10
        for action_type in range(11):
            response = {
                'units': [
                    {'id': 100, 'action': action_type, 'angle': 0.0, 'dist': 0.5}
                ]
            }
            json_str = json.dumps(response)
            parsed = json.loads(json_str)
            assert parsed['units'][0]['action'] == action_type


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
