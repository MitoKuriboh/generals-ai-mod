#!/usr/bin/env python3
"""
ML Inference Server - Serves trained model recommendations

This server loads a trained PPO model and uses it for inference instead of
the rule-based strategy. Use this after training to evaluate model performance.

Usage:
    # Use trained model
    python ml_inference_server.py --model checkpoints/best_agent.pt

    # With logging for evaluation
    python ml_inference_server.py --model checkpoints/best_agent.pt --log eval.jsonl --verbose

    # Fallback to rules if no model
    python ml_inference_server.py --model checkpoints/agent.pt --fallback-rules
"""

import struct
import json
import time
import sys
import os
import argparse
from datetime import datetime
from typing import Optional, Dict

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.model import PolicyNetwork, state_dict_to_tensor, action_tensor_to_dict, STATE_DIM
from training.ppo import PPOAgent, PPOConfig

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


class InferenceServer:
    """Server that uses trained model for inference."""

    def __init__(self, model_path: Optional[str] = None, fallback_rules: bool = True,
                 verbose: bool = False, log_file: Optional[str] = None,
                 deterministic: bool = False):
        self.model_path = model_path
        self.fallback_rules = fallback_rules
        self.verbose = verbose
        self.deterministic = deterministic

        self.pipe = None
        self.connected = False
        self.state_count = 0

        # Logging
        self.log_file = None
        if log_file:
            self.log_file = open(log_file, 'a')
            self._log_session_start()

        # Load model
        self.agent: Optional[PPOAgent] = None
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load trained model."""
        try:
            print(f"[Server] Loading model from {path}...")
            self.agent = PPOAgent(PPOConfig(), device='cpu')
            self.agent.load(path)
            self.agent.policy.eval()  # Set to evaluation mode
            self.model_loaded = True
            print(f"[Server] Model loaded successfully!")
            print(f"[Server]   Total training steps: {self.agent.total_steps}")
            print(f"[Server]   Total episodes: {self.agent.total_episodes}")
        except Exception as e:
            print(f"[Server] Failed to load model: {e}")
            self.model_loaded = False

    def _log_session_start(self):
        """Log session start."""
        if self.log_file:
            entry = {
                'type': 'session_start',
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'model_loaded': self.model_loaded,
            }
            self.log_file.write(json.dumps(entry) + '\n')
            self.log_file.flush()

    def _log_state(self, state: Dict, recommendation: Dict, source: str):
        """Log state and recommendation."""
        if self.log_file:
            entry = {
                'type': 'inference',
                'seq': self.state_count,
                'timestamp': datetime.now().isoformat(),
                'state': state,
                'recommendation': recommendation,
                'source': source,  # 'model' or 'rules'
            }
            self.log_file.write(json.dumps(entry) + '\n')
            self.log_file.flush()

    def create_pipe(self) -> bool:
        """Create named pipe server."""
        if not HAS_WIN32:
            print("[Simulation] Would create pipe:", PIPE_NAME)
            return True

        try:
            self.pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1, 4096, 4096, 0, None
            )
            print(f"[Server] Created pipe: {PIPE_NAME}")
            return True
        except Exception as e:
            print(f"[Server] Failed to create pipe: {e}")
            return False

    def wait_for_connection(self) -> bool:
        """Wait for game connection."""
        if not HAS_WIN32:
            print("[Simulation] Waiting for connection...")
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
        """Read message from pipe."""
        if not HAS_WIN32:
            return None

        try:
            result, length_bytes = win32file.ReadFile(self.pipe, 4)
            if len(length_bytes) < 4:
                return None
            length = struct.unpack('<I', length_bytes)[0]
            result, data = win32file.ReadFile(self.pipe, length)
            return data.decode('utf-8')
        except Exception as e:
            if hasattr(e, 'args') and e.args[0] == 109:
                self.connected = False
            return None

    def write_message(self, data: str) -> bool:
        """Write message to pipe."""
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

    def generate_recommendation(self, state: Dict) -> tuple[Dict, str]:
        """
        Generate recommendation using model or fallback rules.

        Returns:
            (recommendation, source) where source is 'model' or 'rules'
        """
        # Try model inference first
        if self.model_loaded and self.agent:
            try:
                state_tensor = state_dict_to_tensor(state)
                with torch.no_grad():
                    action, _, _ = self.agent.select_action(
                        state_tensor, deterministic=self.deterministic
                    )
                rec = action_tensor_to_dict(action)
                return rec, 'model'
            except Exception as e:
                if self.verbose:
                    print(f"[Server] Model inference failed: {e}")

        # Fallback to rules
        if self.fallback_rules:
            rec = self._rule_based_recommendation(state)
            return rec, 'rules'

        # Default balanced recommendation
        return {
            'priority_economy': 0.25,
            'priority_defense': 0.25,
            'priority_military': 0.25,
            'priority_tech': 0.25,
            'prefer_infantry': 0.33,
            'prefer_vehicles': 0.34,
            'prefer_aircraft': 0.33,
            'aggression': 0.5,
            'target_player': -1
        }, 'default'

    def _rule_based_recommendation(self, state: Dict) -> Dict:
        """Rule-based fallback (same as ml_bridge_server.py)."""
        game_time = state.get('game_time', 0)
        money = state.get('money', 0)
        army_strength = state.get('army_strength', 1.0)
        under_attack = state.get('under_attack', 0)
        base_threat = state.get('base_threat', 0)
        enemy_aircraft = state.get('enemy_aircraft', [0, 0, 0])[0]

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

        # Phase-based
        if game_time < 3.0:
            rec['priority_economy'] = 0.5
            rec['priority_military'] = 0.3
            rec['priority_tech'] = 0.1
            rec['priority_defense'] = 0.1
            rec['aggression'] = 0.2
            rec['prefer_infantry'] = 0.5
            rec['prefer_vehicles'] = 0.35
            rec['prefer_aircraft'] = 0.15
        elif game_time < 8.0:
            rec['priority_economy'] = 0.2
            rec['priority_military'] = 0.35
            rec['priority_tech'] = 0.3
            rec['priority_defense'] = 0.15
            rec['aggression'] = 0.5
            rec['prefer_infantry'] = 0.25
            rec['prefer_vehicles'] = 0.5
            rec['prefer_aircraft'] = 0.25
        else:
            rec['priority_economy'] = 0.15
            rec['priority_military'] = 0.45
            rec['priority_tech'] = 0.25
            rec['priority_defense'] = 0.15
            rec['aggression'] = 0.7
            rec['prefer_infantry'] = 0.2
            rec['prefer_vehicles'] = 0.4
            rec['prefer_aircraft'] = 0.4

        # Situational
        if under_attack > 0.5 or base_threat > 0.5:
            rec['priority_defense'] = max(rec['priority_defense'], 0.35)
            rec['aggression'] = min(rec['aggression'], 0.3)

        if money < 3.0:
            rec['priority_economy'] = max(rec['priority_economy'], 0.4)
        elif money > 3.7:
            rec['priority_military'] = max(rec['priority_military'], 0.4)

        if army_strength > 1.5:
            rec['aggression'] = max(rec['aggression'], 0.75)
        elif army_strength < 0.7:
            rec['aggression'] = min(rec['aggression'], 0.25)

        if enemy_aircraft > 0.5:
            rec['prefer_aircraft'] = max(rec['prefer_aircraft'], 0.4)

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

    def print_state(self, state: Dict, rec: Dict, source: str):
        """Print state and recommendation."""
        self.state_count += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[State #{self.state_count}] Source: {source.upper()}")
            print(f"{'='*60}")
            print(f"  Time: {state.get('game_time', 0):.1f}m | "
                  f"Money: ${10**state.get('money', 0):.0f} | "
                  f"Army: {state.get('army_strength', 1):.2f}x")
            print(f"  Build: eco={rec['priority_economy']:.0%} "
                  f"def={rec['priority_defense']:.0%} "
                  f"mil={rec['priority_military']:.0%} "
                  f"tech={rec['priority_tech']:.0%}")
            print(f"  Army:  inf={rec['prefer_infantry']:.0%} "
                  f"veh={rec['prefer_vehicles']:.0%} "
                  f"air={rec['prefer_aircraft']:.0%}")
            print(f"  Aggression: {rec['aggression']:.0%}")
        else:
            marker = "🤖" if source == 'model' else "📋"
            print(f"[{self.state_count:4d}] {marker} t={state.get('game_time', 0):5.1f}m "
                  f"${10**state.get('money', 0):6.0f} "
                  f"army={state.get('army_strength', 1):4.2f}x "
                  f"agg={rec['aggression']:.0%}")

    def close(self):
        """Clean up."""
        if self.pipe and HAS_WIN32:
            win32file.CloseHandle(self.pipe)
            self.pipe = None
        if self.log_file:
            entry = {
                'type': 'session_end',
                'timestamp': datetime.now().isoformat(),
                'total_states': self.state_count,
            }
            self.log_file.write(json.dumps(entry) + '\n')
            self.log_file.close()
        self.connected = False
        self.state_count = 0


def main():
    parser = argparse.ArgumentParser(description='ML Inference Server for Generals Zero Hour')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--log', type=str, default=None,
                        help='Log file for evaluation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--fallback-rules', action='store_true',
                        help='Use rules if model fails')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions (no sampling)')
    args = parser.parse_args()

    print("=" * 60)
    print("Generals Zero Hour - ML Inference Server")
    print("=" * 60)

    if args.model:
        print(f"Model: {args.model}")
    else:
        print("Model: None (using rules)")

    if args.log:
        print(f"Logging to: {args.log}")
    print()

    server = InferenceServer(
        model_path=args.model,
        fallback_rules=args.fallback_rules or not args.model,
        verbose=args.verbose,
        log_file=args.log,
        deterministic=args.deterministic,
    )

    while True:
        if not server.create_pipe():
            print("[Server] Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if not server.wait_for_connection():
            server.close()
            continue

        print("[Server] Starting inference loop...")

        while server.connected:
            msg = server.read_message()
            if msg is None:
                time.sleep(0.01)
                continue

            try:
                state = json.loads(msg)
                rec, source = server.generate_recommendation(state)
                server._log_state(state, rec, source)
                server.print_state(state, rec, source)
                server.write_message(json.dumps(rec))
            except json.JSONDecodeError as e:
                print(f"[Server] Invalid JSON: {e}")
            except Exception as e:
                print(f"[Server] Error: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n[Server] Connection closed after {server.state_count} messages")
        server.close()
        print("[Server] Waiting for new connection...\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
