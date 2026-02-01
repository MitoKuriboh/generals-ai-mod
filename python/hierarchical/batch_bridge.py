"""
Batched ML Bridge

Efficient batched communication protocol for hierarchical inference.

Protocol format:
- Game sends batched state for strategic + teams + units
- Python responds with batched recommendations for all layers

This reduces pipe communication overhead by combining messages.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from training.config import PROTOCOL_VERSION


@dataclass
class BatchedRequest:
    """Batched request from game."""

    frame: int
    player_id: int

    # Strategic state (always present)
    strategic_state: Dict

    # Team states (optional, only teams needing decisions)
    team_states: Dict[int, Dict]  # team_id -> state dict

    # Unit states (optional, only units needing micro)
    unit_states: Dict[int, Dict]  # unit_id -> state dict


@dataclass
class BatchedResponse:
    """Batched response to game."""

    frame: int
    version: int

    # Strategic recommendation (always present)
    strategic: Dict

    # Tactical commands (per team)
    teams: List[Dict]

    # Micro commands (per unit)
    units: List[Dict]


class BatchedMLBridge:
    """
    Handles batched communication with the game.

    Optimizes for low latency by:
    - Single message per frame
    - Compact JSON format
    - Optional fields only when needed
    """

    def __init__(self, tactical_enabled: bool = False, micro_enabled: bool = False):
        self.version = PROTOCOL_VERSION
        self.frame_count = 0
        self.message_count = 0

        # Capability flags - set based on what models are loaded
        self.hierarchical_enabled = True  # Always true for this bridge
        self.tactical_enabled = tactical_enabled
        self.micro_enabled = micro_enabled

        # Stats
        self.total_teams_processed = 0
        self.total_units_processed = 0

    def parse_request(self, json_str: str) -> BatchedRequest:
        """
        Parse batched request from game.

        Expected format:
        {
            "frame": 1234,
            "player_id": 3,
            "strategic": { ... },
            "teams": [
                {"id": 1, "state": { ... }},
                ...
            ],
            "units": [
                {"id": 101, "state": { ... }},
                ...
            ]
        }
        """
        data = json.loads(json_str)

        frame = data.get('frame', 0)
        player_id = data.get('player_id', 0)

        # Parse strategic state
        strategic_state = data.get('strategic', {})

        # Parse team states
        team_states = {}
        for team_data in data.get('teams', []):
            team_id = team_data.get('id')
            if team_id is not None:
                team_states[team_id] = team_data.get('state', {})

        # Parse unit states
        unit_states = {}
        for unit_data in data.get('units', []):
            unit_id = unit_data.get('id')
            if unit_id is not None:
                unit_states[unit_id] = unit_data.get('state', {})

        self.frame_count = frame
        self.message_count += 1

        return BatchedRequest(
            frame=frame,
            player_id=player_id,
            strategic_state=strategic_state,
            team_states=team_states,
            unit_states=unit_states,
        )

    def build_response(self,
                       frame: int,
                       strategic: Dict,
                       teams: List[Dict],
                       units: List[Dict]) -> str:
        """
        Build batched response JSON.

        Output format:
        {
            "frame": 1234,
            "version": 2,
            "capabilities": {
                "hierarchical": true,
                "tactical": true,
                "micro": false
            },
            "strategic": {
                "priority_economy": 0.25,
                ...
            },
            "teams": [
                {"id": 1, "action": 0, "x": 0.5, "y": 0.6, "attitude": 0.8},
                ...
            ],
            "units": [
                {"id": 101, "action": 5, "angle": 1.2, "dist": 0.3},
                ...
            ]
        }
        """
        response = {
            'frame': frame,
            'version': self.version,
            'capabilities': {
                'hierarchical': self.hierarchical_enabled,
                'tactical': self.tactical_enabled,
                'micro': self.micro_enabled,
            },
            'strategic': strategic,
        }

        # Only include teams/units if present
        if teams:
            response['teams'] = teams
            self.total_teams_processed += len(teams)

        if units:
            response['units'] = units
            self.total_units_processed += len(units)

        return json.dumps(response, separators=(',', ':'))  # Compact JSON

    def build_game_state_from_request(self, request: BatchedRequest) -> Dict:
        """
        Build unified game state dict from batched request.

        Combines strategic, team, and unit data into format expected
        by the coordinator.
        """
        game_state = dict(request.strategic_state)

        # Add teams as nested dict
        game_state['teams'] = {
            str(team_id): state
            for team_id, state in request.team_states.items()
        }

        # Add units as nested dict
        game_state['units'] = {
            str(unit_id): state
            for unit_id, state in request.unit_states.items()
        }

        return game_state

    def build_response_from_result(self, frame: int, result: Dict) -> str:
        """
        Build response from coordinator result.

        Args:
            frame: Current frame number
            result: Output from HierarchicalCoordinator.process_frame()

        Returns:
            JSON string for transmission
        """
        strategic = result.get('strategic', self._default_strategic())
        teams = result.get('teams', [])
        units = result.get('units', [])

        return self.build_response(frame, strategic, teams, units)

    def _default_strategic(self) -> Dict:
        """Default strategic recommendation when none computed."""
        return {
            'priority_economy': 0.25,
            'priority_defense': 0.15,
            'priority_military': 0.45,
            'priority_tech': 0.15,
            'prefer_infantry': 0.33,
            'prefer_vehicles': 0.34,
            'prefer_aircraft': 0.33,
            'aggression': 0.5,
            'target_player': -1,
        }

    def get_stats(self) -> Dict:
        """Get communication statistics."""
        return {
            'messages': self.message_count,
            'last_frame': self.frame_count,
            'total_teams': self.total_teams_processed,
            'total_units': self.total_units_processed,
        }


def validate_request(json_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validate request JSON format.

    Returns:
        (is_valid, error_message)
    """
    try:
        data = json.loads(json_str)

        # Check required fields
        if 'strategic' not in data:
            return False, "Missing 'strategic' field"

        if 'frame' not in data:
            return False, "Missing 'frame' field"

        # Validate strategic state
        strategic = data['strategic']
        required_strategic = ['money', 'power']
        for field in required_strategic:
            if field not in strategic:
                return False, f"Missing strategic field: {field}"

        # Validate team states if present
        for team_data in data.get('teams', []):
            if 'id' not in team_data:
                return False, "Team missing 'id' field"

        # Validate unit states if present
        for unit_data in data.get('units', []):
            if 'id' not in unit_data:
                return False, "Unit missing 'id' field"

        return True, None

    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"


def validate_response(json_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validate response JSON format.

    Returns:
        (is_valid, error_message)
    """
    try:
        data = json.loads(json_str)

        # Check required fields
        if 'strategic' not in data:
            return False, "Missing 'strategic' field"

        if 'version' not in data:
            return False, "Missing 'version' field"

        # Validate strategic response
        strategic = data['strategic']
        required_fields = [
            'priority_economy', 'priority_defense', 'priority_military',
            'priority_tech', 'prefer_infantry', 'prefer_vehicles',
            'prefer_aircraft', 'aggression'
        ]
        for field in required_fields:
            if field not in strategic:
                return False, f"Missing strategic field: {field}"
            if not isinstance(strategic[field], (int, float)):
                return False, f"Invalid type for {field}"

        # Validate team commands if present
        for team_cmd in data.get('teams', []):
            if 'id' not in team_cmd or 'action' not in team_cmd:
                return False, "Team command missing required fields"

        # Validate unit commands if present
        for unit_cmd in data.get('units', []):
            if 'id' not in unit_cmd or 'action' not in unit_cmd:
                return False, "Unit command missing required fields"

        return True, None

    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"


if __name__ == '__main__':
    print("Testing BatchedMLBridge...")

    bridge = BatchedMLBridge()

    # Test request parsing
    request_json = json.dumps({
        'frame': 1234,
        'player_id': 3,
        'strategic': {
            'money': 3.5,
            'power': 50,
            'income': 5,
            'supply': 0.6,
            'own_infantry': [1.0, 0.8, 0],
            'own_vehicles': [0.7, 0.9, 0],
            'own_aircraft': [0.3, 1.0, 0],
            'own_structures': [1.0, 0.95, 0],
            'enemy_infantry': [0.8, 0.7, 0],
            'enemy_vehicles': [0.5, 0.8, 0],
            'enemy_aircraft': [0.2, 1.0, 0],
            'enemy_structures': [0.9, 0.9, 0],
            'game_time': 5.0,
            'tech_level': 0.7,
            'base_threat': 0.3,
            'army_strength': 1.2,
            'under_attack': 0,
            'distance_to_enemy': 0.6,
            'is_usa': 1, 'is_china': 0, 'is_gla': 0,
        },
        'teams': [
            {'id': 1, 'state': {'health': 0.8, 'composition': {'infantry': 10}}},
            {'id': 2, 'state': {'health': 0.9, 'composition': {'vehicles': 5}}},
        ],
        'units': [
            {'id': 101, 'state': {'health': 0.75, 'type': 'tank', 'situational': {'under_fire': True}}},
            {'id': 102, 'state': {'health': 0.5, 'type': 'infantry'}},
        ],
    })

    # Validate request
    valid, error = validate_request(request_json)
    assert valid, f"Request validation failed: {error}"
    print("Request validation passed")

    # Parse request
    request = bridge.parse_request(request_json)
    print(f"\nParsed request:")
    print(f"  Frame: {request.frame}")
    print(f"  Player: {request.player_id}")
    print(f"  Teams: {len(request.team_states)}")
    print(f"  Units: {len(request.unit_states)}")

    # Build game state
    game_state = bridge.build_game_state_from_request(request)
    print(f"\nGame state teams: {list(game_state['teams'].keys())}")
    print(f"Game state units: {list(game_state['units'].keys())}")

    # Build response
    strategic = {
        'priority_economy': 0.2,
        'priority_defense': 0.1,
        'priority_military': 0.5,
        'priority_tech': 0.2,
        'prefer_infantry': 0.3,
        'prefer_vehicles': 0.4,
        'prefer_aircraft': 0.3,
        'aggression': 0.7,
        'target_player': -1,
    }

    teams = [
        {'id': 1, 'action': 0, 'x': 0.6, 'y': 0.5, 'attitude': 0.8},
    ]

    units = [
        {'id': 101, 'action': 5, 'angle': -0.5, 'dist': 0.3},
    ]

    response_json = bridge.build_response(request.frame, strategic, teams, units)
    print(f"\nResponse: {response_json[:100]}...")

    # Validate response
    valid, error = validate_response(response_json)
    assert valid, f"Response validation failed: {error}"
    print("Response validation passed")

    # Stats
    print(f"\nBridge stats: {bridge.get_stats()}")

    print("\nBatchedMLBridge test passed!")
