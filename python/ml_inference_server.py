#!/usr/bin/env python3
"""
ML Inference Server - Hierarchical AI for C&C Generals Zero Hour

Provides:
- Strategic layer: Build priorities, army composition, aggression
- Tactical layer: Team-level commands (attack, defend, retreat, etc.)

The AI adapts to what's happening in game, not just following a fixed script.

Usage:
    python ml_inference_server.py --model checkpoints/best_agent.pt --verbose
"""

import struct
import json
import time
import sys
import os
import argparse
from datetime import datetime
from typing import Optional, Dict, List

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.model import PolicyNetwork, state_dict_to_tensor, action_tensor_to_dict, STATE_DIM
from training.ppo import PPOAgent, PPOConfig
from training.config import PROTOCOL_VERSION
from training.tactical import TacticalDecisionMaker, TacticalCommand, parse_tactical_state
from training.micro import MicroDecisionMaker, MicroCommand, parse_micro_state

# Windows named pipe support
if sys.platform == 'win32':
    import win32pipe
    import win32file
    import pywintypes
    HAS_WIN32 = True
else:
    HAS_WIN32 = False
    print("Warning: Named pipes require Windows. Running in simulation mode.")


PIPE_NAME = r'\\.\pipe\generals_ml_bridge'


class HierarchicalInferenceServer:
    """
    Hierarchical AI server with strategic and tactical layers.

    Strategic: ML model for build priorities, army composition, aggression
    Tactical: Rule-based adaptive decisions for team commands
    """

    def __init__(self, model_path: Optional[str] = None, verbose: bool = False,
                 log_file: Optional[str] = None, deterministic: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.deterministic = deterministic

        self.pipe = None
        self.connected = False
        self.state_count = 0
        self._protocol_validated = False

        # Logging
        self.log_file = None
        if log_file:
            self.log_file = open(log_file, 'a')
            self._log_session_start()

        # Strategic layer: ML model
        self.agent: Optional[PPOAgent] = None
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        # Tactical layer: Rule-based for now
        self.tactical = TacticalDecisionMaker(verbose=verbose)

        # Micro layer: Rule-based adaptive control
        self.micro = MicroDecisionMaker(verbose=verbose)

    def _load_model(self, path: str):
        """Load trained strategic model."""
        try:
            print(f"[Server] Loading strategic model from {path}...")
            self.agent = PPOAgent(PPOConfig(), device='cpu')
            self.agent.load(path)
            self.agent.policy.eval()
            self.model_loaded = True
            print(f"[Server] Model loaded!")
            print(f"[Server]   Training steps: {self.agent.total_steps}")
        except Exception as e:
            print(f"[Server] Model load failed: {e}")
            self.model_loaded = False

    def _log_session_start(self):
        if self.log_file:
            entry = {
                'type': 'session_start',
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'hierarchical': True,
            }
            self.log_file.write(json.dumps(entry) + '\n')
            self.log_file.flush()

    def create_pipe(self) -> bool:
        if not HAS_WIN32:
            print("[Simulation] Would create pipe:", PIPE_NAME)
            return True
        try:
            self.pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1, 65536, 65536, 0, None  # Larger buffers for batched data
            )
            print(f"[Server] Created pipe: {PIPE_NAME}")
            return True
        except Exception as e:
            print(f"[Server] Failed to create pipe: {e}")
            return False

    def wait_for_connection(self) -> bool:
        if not HAS_WIN32:
            time.sleep(1)
            self.connected = True
            return True
        print("[Server] Waiting for game to connect...")
        try:
            win32pipe.ConnectNamedPipe(self.pipe, None)
            self.connected = True
            print("[Server] Game connected!")
            return True
        except Exception as e:
            print(f"[Server] Connection failed: {e}")
            return False

    def read_message(self) -> Optional[str]:
        if not HAS_WIN32:
            return None
        try:
            length_bytes = b''
            while len(length_bytes) < 4:
                result, chunk = win32file.ReadFile(self.pipe, 4 - len(length_bytes))
                if not chunk:
                    return None
                length_bytes += chunk

            length = struct.unpack('<I', length_bytes)[0]
            if length == 0 or length > 1024 * 1024:
                return None

            data = b''
            while len(data) < length:
                result, chunk = win32file.ReadFile(self.pipe, length - len(data))
                if not chunk:
                    return None
                data += chunk

            return data.decode('utf-8')
        except Exception as e:
            if hasattr(e, 'args') and e.args[0] == 109:
                self.connected = False
            return None

    def write_message(self, data: str) -> bool:
        if not HAS_WIN32:
            return True
        try:
            encoded = data.encode('utf-8')
            length = struct.pack('<I', len(encoded))
            win32file.WriteFile(self.pipe, length + encoded)
            return True
        except Exception as e:
            self.connected = False
            return False

    def process_request(self, request: Dict) -> Dict:
        """
        Process a request from the game.

        Can be:
        - Simple strategic request (legacy)
        - Batched request with strategic + tactical states
        """
        self.state_count += 1

        # Check for batched format
        if 'strategic' in request or 'teams' in request:
            return self._process_batched(request)
        else:
            # Legacy simple strategic request
            return self._process_simple_strategic(request)

    def _process_simple_strategic(self, state: Dict) -> Dict:
        """Process legacy simple strategic request."""
        rec = self._generate_strategic(state)

        if self.verbose:
            self._print_strategic(state, rec)

        # Include capabilities to enable hierarchical mode
        rec['capabilities'] = {
            'hierarchical': True,
            'tactical': True,
            'micro': True  # Micro layer enabled
        }

        return rec

    def _process_batched(self, request: Dict) -> Dict:
        """Process batched request with strategic, tactical, and micro states."""
        response = {
            'version': PROTOCOL_VERSION,
            'capabilities': {
                'hierarchical': True,
                'tactical': True,
                'micro': True
            }
        }

        # Strategic layer
        if 'strategic' in request:
            strategic_rec = self._generate_strategic(request['strategic'])
            response['strategic'] = strategic_rec

            if self.verbose:
                self._print_strategic(request['strategic'], strategic_rec)

        # Tactical layer - process each team
        if 'teams' in request and isinstance(request['teams'], list):
            tactical_commands = []

            for team_data in request['teams']:
                team_id = team_data.get('id', 0)
                state_array = team_data.get('state', [])

                if len(state_array) >= 64:
                    try:
                        tac_state = parse_tactical_state(team_id, state_array)
                        cmd = self.tactical.decide(tac_state)
                        tactical_commands.append({
                            'id': cmd.team_id,
                            'action': cmd.action,
                            'x': cmd.target_x,
                            'y': cmd.target_y,
                            'attitude': cmd.attitude
                        })
                    except Exception as e:
                        if self.verbose:
                            print(f"  [!] Team {team_id} tactical error: {e}")

            if tactical_commands:
                response['tactical'] = tactical_commands

                if self.verbose:
                    print(f"  Tactical: {len(tactical_commands)} team commands")

        # Micro layer - process each unit
        if 'units' in request and isinstance(request['units'], list):
            micro_commands = []

            for unit_data in request['units']:
                unit_id = unit_data.get('id', 0)
                state_array = unit_data.get('state', [])

                if len(state_array) >= 32:
                    try:
                        micro_state = parse_micro_state(unit_id, state_array)
                        cmd = self.micro.decide(micro_state)
                        micro_commands.append({
                            'id': cmd.unit_id,
                            'action': cmd.action,
                            'angle': cmd.move_angle / 3.14159265359,  # Normalize to -1..1 for C++
                            'dist': cmd.move_distance
                        })
                    except Exception as e:
                        if self.verbose:
                            print(f"  [!] Unit {unit_id} micro error: {e}")

            if micro_commands:
                response['micro'] = micro_commands

                if self.verbose:
                    print(f"  Micro: {len(micro_commands)} unit commands")

        return response

    def _generate_strategic(self, state: Dict) -> Dict:
        """Generate strategic recommendation."""
        # Try ML model
        if self.model_loaded and self.agent:
            try:
                state_tensor = state_dict_to_tensor(state)
                with torch.no_grad():
                    action, _, _ = self.agent.select_action(
                        state_tensor, deterministic=self.deterministic
                    )
                return action_tensor_to_dict(action)
            except Exception as e:
                if self.verbose:
                    print(f"[!] Model inference failed: {e}")

        # Fallback: adaptive rules
        return self._rule_based_strategic(state)

    def _rule_based_strategic(self, state: Dict) -> Dict:
        """Adaptive rule-based strategic decisions."""
        game_time = state.get('game_time', 0)
        money = state.get('money', 0)
        army_strength = state.get('army_strength', 1.0)
        under_attack = state.get('under_attack', 0)
        base_threat = state.get('base_threat', 0)
        tech_level = state.get('tech_level', 0)

        # Get enemy force info
        enemy_inf = state.get('enemy_infantry', [0])[0]
        enemy_veh = state.get('enemy_vehicles', [0])[0]
        enemy_air = state.get('enemy_aircraft', [0])[0]

        rec = {
            'priority_economy': 0.25,
            'priority_defense': 0.25,
            'priority_military': 0.25,
            'priority_tech': 0.25,
            'prefer_infantry': 0.33,
            'prefer_vehicles': 0.34,
            'prefer_aircraft': 0.33,
            'aggression': 0.5,
            'target_player': -1
        }

        # === ADAPTIVE DECISIONS ===

        # Phase-based baseline
        if game_time < 3.0:  # Early game
            rec['priority_economy'] = 0.45
            rec['priority_military'] = 0.35
            rec['priority_tech'] = 0.1
            rec['priority_defense'] = 0.1
            rec['aggression'] = 0.25
        elif game_time < 8.0:  # Mid game
            rec['priority_economy'] = 0.2
            rec['priority_military'] = 0.4
            rec['priority_tech'] = 0.25
            rec['priority_defense'] = 0.15
            rec['aggression'] = 0.55
        else:  # Late game
            rec['priority_economy'] = 0.15
            rec['priority_military'] = 0.45
            rec['priority_tech'] = 0.2
            rec['priority_defense'] = 0.2
            rec['aggression'] = 0.7

        # === SITUATIONAL ADAPTATIONS ===

        # Under attack -> prioritize defense, reduce aggression
        if under_attack > 0.5 or base_threat > 0.5:
            rec['priority_defense'] = max(rec['priority_defense'], 0.35)
            rec['priority_military'] = max(rec['priority_military'], 0.35)
            rec['aggression'] = min(rec['aggression'], 0.3)

        # Low money -> prioritize economy
        if money < 3.0:  # log10($1000) = 3
            rec['priority_economy'] = max(rec['priority_economy'], 0.4)

        # Rich -> spend on military
        elif money > 3.7:  # log10($5000) = 3.7
            rec['priority_military'] = max(rec['priority_military'], 0.45)
            rec['aggression'] = min(1.0, rec['aggression'] + 0.1)

        # Strong army -> be aggressive
        if army_strength > 1.5:
            rec['aggression'] = max(rec['aggression'], 0.75)

        # Weak army -> turtle up
        elif army_strength < 0.7:
            rec['aggression'] = min(rec['aggression'], 0.25)
            rec['priority_defense'] = max(rec['priority_defense'], 0.3)

        # Low tech -> invest in tech if stable
        if tech_level < 0.3 and not under_attack:
            rec['priority_tech'] = max(rec['priority_tech'], 0.3)

        # === COUNTER ENEMY COMPOSITION ===

        total_enemy = enemy_inf + enemy_veh + enemy_air + 0.001
        if enemy_air > 0.5:
            # Enemy has air -> build anti-air (vehicles/infantry)
            rec['prefer_aircraft'] = 0.15
            rec['prefer_vehicles'] = 0.5
            rec['prefer_infantry'] = 0.35
        elif enemy_veh > enemy_inf * 1.5:
            # Enemy heavy on vehicles -> counter with aircraft
            rec['prefer_aircraft'] = 0.45
            rec['prefer_vehicles'] = 0.35
            rec['prefer_infantry'] = 0.2
        elif enemy_inf > enemy_veh * 1.5:
            # Enemy infantry spam -> vehicles
            rec['prefer_vehicles'] = 0.5
            rec['prefer_infantry'] = 0.25
            rec['prefer_aircraft'] = 0.25

        # Normalize
        total_pri = sum([rec['priority_economy'], rec['priority_defense'],
                        rec['priority_military'], rec['priority_tech']])
        if total_pri > 0:
            rec['priority_economy'] /= total_pri
            rec['priority_defense'] /= total_pri
            rec['priority_military'] /= total_pri
            rec['priority_tech'] /= total_pri

        total_army = rec['prefer_infantry'] + rec['prefer_vehicles'] + rec['prefer_aircraft']
        if total_army > 0:
            rec['prefer_infantry'] /= total_army
            rec['prefer_vehicles'] /= total_army
            rec['prefer_aircraft'] /= total_army

        return rec

    def _print_strategic(self, state: Dict, rec: Dict):
        """Print strategic state and recommendation."""
        src = "MODEL" if self.model_loaded else "RULES"
        print(f"\n[{self.state_count:4d}] Strategic ({src})")
        print(f"  Time: {state.get('game_time', 0):.1f}m | "
              f"Money: ${10**state.get('money', 0):.0f} | "
              f"Army: {state.get('army_strength', 1):.2f}x | "
              f"Threat: {state.get('base_threat', 0):.0%}")
        print(f"  -> eco={rec['priority_economy']:.0%} "
              f"def={rec['priority_defense']:.0%} "
              f"mil={rec['priority_military']:.0%} "
              f"tech={rec['priority_tech']:.0%} "
              f"| agg={rec['aggression']:.0%}")

    def validate_protocol_version(self, state: Dict) -> bool:
        if self._protocol_validated:
            return True
        version = state.get('version', 1)
        if version != PROTOCOL_VERSION:
            # M4 FIX: Fail fast on protocol mismatch instead of continuing silently
            error_msg = f"Protocol mismatch: expected v{PROTOCOL_VERSION}, got v{version}"
            print(f"[!] FATAL: {error_msg}")
            raise ValueError(error_msg)  # Will be caught by exception handler and send error response
        else:
            print(f"[OK] Protocol v{version}")
        self._protocol_validated = True
        return True

    def close(self):
        if self.pipe and HAS_WIN32:
            win32file.CloseHandle(self.pipe)
            self.pipe = None
        if self.log_file:
            self.log_file.write(json.dumps({
                'type': 'session_end',
                'total_states': self.state_count
            }) + '\n')
            self.log_file.close()
            self.log_file = None
        self.connected = False
        self.state_count = 0
        self._protocol_validated = False


def main():
    parser = argparse.ArgumentParser(description='Hierarchical ML Server for Generals Zero Hour')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to trained strategic model')
    parser.add_argument('--log', type=str, default=None,
                        help='Log file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--deterministic', action='store_true',
                        help='Deterministic actions (no sampling)')
    args = parser.parse_args()

    print("=" * 60)
    print("C&C Generals - Hierarchical AI Server")
    print("=" * 60)
    print(f"Strategic: {'ML Model' if args.model else 'Rule-based (adaptive)'}")
    print(f"Tactical:  Rule-based (adaptive)")
    print(f"Micro:     Rule-based (adaptive)")
    print()

    server = HierarchicalInferenceServer(
        model_path=args.model,
        verbose=args.verbose,
        log_file=args.log,
        deterministic=args.deterministic,
    )

    while True:
        server.close()

        if not server.create_pipe():
            print("[Server] Retrying in 5s...")
            time.sleep(5)
            continue

        if not server.wait_for_connection():
            server.close()
            continue

        print("[Server] Running hierarchical inference...")
        print("[Server] AI will now ADAPT to game situations!\n")

        while server.connected:
            msg = server.read_message()
            if msg is None:
                time.sleep(0.01)
                continue

            try:
                request = json.loads(msg)
                server.validate_protocol_version(request)
                response = server.process_request(request)
                server.write_message(json.dumps(response))
            except json.JSONDecodeError as e:
                print(f"[!] Bad JSON: {e}")
                # H7 FIX: Send error response instead of silently disconnecting
                error_response = {
                    'error': True,
                    'message': f'Invalid JSON: {str(e)}',
                    'version': PROTOCOL_VERSION
                }
                server.write_message(json.dumps(error_response))
                server.connected = False
                break
            except Exception as e:
                print(f"[!] Error: {e}")
                import traceback
                traceback.print_exc()
                # H7 FIX: Always send error response - don't leave C++ waiting
                error_response = {
                    'error': True,
                    'message': f'Server error: {str(e)}',
                    'version': PROTOCOL_VERSION
                }
                if not server.write_message(json.dumps(error_response)):
                    # Write failed, connection probably dead
                    server.connected = False
                    break

        print(f"\n[Server] Disconnected after {server.state_count} requests")
        server.close()
        print("[Server] Waiting for reconnection...\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
