#!/usr/bin/env python3
"""
ML Bridge Server - Test client for Generals Zero Hour Learning AI

This script creates a named pipe server that receives game state from the
Learning AI and sends back recommendations. For testing purposes, it sends
simple rule-based recommendations.

Usage:
    python ml_bridge_server.py [--log states.jsonl] [--verbose]

Options:
    --log FILE      Log all states to a JSONL file for analysis
    --verbose       Print detailed state information
    --no-respond    Don't send recommendations (observation only)

The server must be running BEFORE starting the game with Learning AI.
"""

import struct
import json
import time
import sys
import os
import argparse
from datetime import datetime

# Windows named pipe support
if sys.platform == 'win32':
    import win32pipe
    import win32file
    import pywintypes
else:
    print("Warning: Named pipes require Windows. Running in simulation mode.")


PIPE_NAME = r'\\.\pipe\generals_ml_bridge'


class StateLogger:
    """Logs game states to a JSONL file for analysis."""

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.state_count = 0
        self.session_start = datetime.now().isoformat()

    def open(self):
        self.file = open(self.filename, 'a')
        # Write session header
        header = {
            'type': 'session_start',
            'timestamp': self.session_start,
            'pipe': PIPE_NAME
        }
        self.file.write(json.dumps(header) + '\n')
        self.file.flush()

    def log_state(self, state, recommendation=None):
        if not self.file:
            return

        self.state_count += 1
        entry = {
            'type': 'state',
            'seq': self.state_count,
            'timestamp': datetime.now().isoformat(),
            'state': state
        }
        if recommendation:
            entry['recommendation'] = recommendation

        self.file.write(json.dumps(entry) + '\n')
        self.file.flush()

    def close(self):
        if self.file:
            footer = {
                'type': 'session_end',
                'timestamp': datetime.now().isoformat(),
                'total_states': self.state_count
            }
            self.file.write(json.dumps(footer) + '\n')
            self.file.close()
            self.file = None


class MLBridgeServer:
    """Named pipe server for ML Bridge communication."""

    def __init__(self, logger=None, verbose=False, respond=True):
        self.pipe = None
        self.connected = False
        self.logger = logger
        self.verbose = verbose
        self.respond = respond
        self.state_count = 0

    def create_pipe(self):
        """Create the named pipe server."""
        if sys.platform != 'win32':
            print("[Simulation] Would create pipe:", PIPE_NAME)
            return True

        try:
            self.pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1,  # Max instances
                4096,  # Out buffer size
                4096,  # In buffer size
                0,  # Default timeout
                None  # Security attributes
            )
            print(f"[Server] Created pipe: {PIPE_NAME}")
            return True
        except pywintypes.error as e:
            print(f"[Server] Failed to create pipe: {e}")
            return False

    def wait_for_connection(self):
        """Wait for the game to connect."""
        if sys.platform != 'win32':
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
        except pywintypes.error as e:
            print(f"[Server] Connection failed: {e}")
            return False

    def read_message(self):
        """Read a length-prefixed message from the pipe."""
        if sys.platform != 'win32':
            return None

        try:
            # Read 4-byte length prefix
            result, length_bytes = win32file.ReadFile(self.pipe, 4)
            if len(length_bytes) < 4:
                return None

            length = struct.unpack('<I', length_bytes)[0]

            # Read message data
            result, data = win32file.ReadFile(self.pipe, length)
            return data.decode('utf-8')
        except pywintypes.error as e:
            if e.args[0] == 109:  # ERROR_BROKEN_PIPE
                print("[Server] Pipe broken, client disconnected")
                self.connected = False
            return None

    def write_message(self, data):
        """Write a length-prefixed message to the pipe."""
        if sys.platform != 'win32':
            if self.verbose:
                print(f"[Simulation] Would send: {data[:100]}...")
            return True

        try:
            encoded = data.encode('utf-8')
            length = struct.pack('<I', len(encoded))
            win32file.WriteFile(self.pipe, length + encoded)
            return True
        except pywintypes.error as e:
            print(f"[Server] Write failed: {e}")
            self.connected = False
            return False

    def generate_recommendation(self, state):
        """Generate a recommendation based on game state.

        This is an improved rule-based system for testing.
        Replace with ML model inference for actual training.
        """
        # Extract state values
        money = state.get('money', 0)  # log10 scale
        power = state.get('power', 0)
        base_threat = state.get('base_threat', 0)
        army_strength = state.get('army_strength', 1.0)
        game_time = state.get('game_time', 0)
        tech_level = state.get('tech_level', 0)
        under_attack = state.get('under_attack', 0)

        # Extract force counts (log10 scale, so 1.0 = 9 units)
        own_infantry = state.get('own_infantry', [0, 0, 0])[0]
        own_vehicles = state.get('own_vehicles', [0, 0, 0])[0]
        own_aircraft = state.get('own_aircraft', [0, 0, 0])[0]
        enemy_aircraft = state.get('enemy_aircraft', [0, 0, 0])[0]

        # Initialize with balanced defaults
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

        # === PHASE-BASED STRATEGY ===

        # Early game (0-3 minutes): Economy focus, low aggression
        if game_time < 3.0:
            rec['priority_economy'] = 0.5
            rec['priority_military'] = 0.3
            rec['priority_tech'] = 0.1
            rec['priority_defense'] = 0.1
            rec['aggression'] = 0.2
            # Prefer infantry early (cheap, fast)
            rec['prefer_infantry'] = 0.5
            rec['prefer_vehicles'] = 0.35
            rec['prefer_aircraft'] = 0.15

        # Mid game (3-8 minutes): Balanced, tech up
        elif game_time < 8.0:
            rec['priority_economy'] = 0.2
            rec['priority_military'] = 0.35
            rec['priority_tech'] = 0.3
            rec['priority_defense'] = 0.15
            rec['aggression'] = 0.5
            # Shift to vehicles
            rec['prefer_infantry'] = 0.25
            rec['prefer_vehicles'] = 0.5
            rec['prefer_aircraft'] = 0.25

        # Late game (8+ minutes): Military focus, high aggression
        else:
            rec['priority_economy'] = 0.15
            rec['priority_military'] = 0.45
            rec['priority_tech'] = 0.25
            rec['priority_defense'] = 0.15
            rec['aggression'] = 0.7
            # Mixed high-tech army
            rec['prefer_infantry'] = 0.2
            rec['prefer_vehicles'] = 0.4
            rec['prefer_aircraft'] = 0.4

        # === SITUATIONAL ADJUSTMENTS ===

        # Under attack: defensive stance
        if under_attack > 0.5 or base_threat > 0.5:
            rec['priority_defense'] = max(rec['priority_defense'], 0.35)
            rec['priority_military'] = max(rec['priority_military'], 0.35)
            rec['priority_economy'] = min(rec['priority_economy'], 0.15)
            rec['aggression'] = min(rec['aggression'], 0.3)

        # Low money: focus on economy
        if money < 3.0:  # Less than $1000
            rec['priority_economy'] = max(rec['priority_economy'], 0.4)
            rec['aggression'] = min(rec['aggression'], 0.3)

        # High money: spend on military and tech
        elif money > 3.7:  # More than $5000
            rec['priority_military'] = max(rec['priority_military'], 0.4)
            rec['priority_tech'] = max(rec['priority_tech'], 0.25)
            rec['priority_economy'] = min(rec['priority_economy'], 0.15)

        # Strong army: be aggressive
        if army_strength > 1.5:
            rec['aggression'] = max(rec['aggression'], 0.75)
            rec['priority_military'] = max(rec['priority_military'], 0.4)

        # Weak army: defensive, build up
        elif army_strength < 0.7:
            rec['aggression'] = min(rec['aggression'], 0.25)
            rec['priority_defense'] = max(rec['priority_defense'], 0.3)

        # Counter enemy air with own air
        if enemy_aircraft > 0.5:  # Enemy has 3+ aircraft
            rec['prefer_aircraft'] = max(rec['prefer_aircraft'], 0.4)

        # Low power: prioritize economy (power plants)
        if power < 0:
            rec['priority_economy'] = max(rec['priority_economy'], 0.4)

        # Normalize priorities to sum to 1.0
        total_pri = (rec['priority_economy'] + rec['priority_defense'] +
                     rec['priority_military'] + rec['priority_tech'])
        if total_pri > 0:
            rec['priority_economy'] /= total_pri
            rec['priority_defense'] /= total_pri
            rec['priority_military'] /= total_pri
            rec['priority_tech'] /= total_pri

        # Normalize army preferences to sum to 1.0
        total_army = rec['prefer_infantry'] + rec['prefer_vehicles'] + rec['prefer_aircraft']
        if total_army > 0:
            rec['prefer_infantry'] /= total_army
            rec['prefer_vehicles'] /= total_army
            rec['prefer_aircraft'] /= total_army

        return rec

    def print_state(self, state, rec=None):
        """Print state information."""
        self.state_count += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[State #{self.state_count}] Player {state.get('player', '?')}")
            print(f"{'='*60}")

            # Economy
            print(f"\n  ECONOMY:")
            print(f"    Money:    {state.get('money', 0):.2f} (log10) = ${10**state.get('money', 0):.0f}")
            print(f"    Power:    {state.get('power', 0):.0f}")
            print(f"    Income:   {state.get('income', 0):.1f}/s")
            print(f"    Supply:   {state.get('supply', 0):.1%}")

            # Forces
            print(f"\n  OWN FORCES:")
            print(f"    Infantry:   count={10**state.get('own_infantry', [0])[0]-1:.0f}, health={state.get('own_infantry', [0,0])[1]:.0%}")
            print(f"    Vehicles:   count={10**state.get('own_vehicles', [0])[0]-1:.0f}, health={state.get('own_vehicles', [0,0])[1]:.0%}")
            print(f"    Aircraft:   count={10**state.get('own_aircraft', [0])[0]-1:.0f}, health={state.get('own_aircraft', [0,0])[1]:.0%}")
            print(f"    Structures: count={10**state.get('own_structures', [0])[0]-1:.0f}, health={state.get('own_structures', [0,0])[1]:.0%}")

            print(f"\n  ENEMY FORCES (visible):")
            print(f"    Infantry:   count={10**state.get('enemy_infantry', [0])[0]-1:.0f}")
            print(f"    Vehicles:   count={10**state.get('enemy_vehicles', [0])[0]-1:.0f}")
            print(f"    Aircraft:   count={10**state.get('enemy_aircraft', [0])[0]-1:.0f}")
            print(f"    Structures: count={10**state.get('enemy_structures', [0])[0]-1:.0f}")

            # Strategic
            print(f"\n  STRATEGIC:")
            print(f"    Game time:    {state.get('game_time', 0):.1f} min")
            print(f"    Tech level:   {state.get('tech_level', 0):.0%}")
            print(f"    Base threat:  {state.get('base_threat', 0):.0%}")
            print(f"    Army strength:{state.get('army_strength', 0):.2f}x enemy")
            print(f"    Under attack: {'YES' if state.get('under_attack', 0) > 0.5 else 'no'}")
            print(f"    Distance:     {state.get('distance_to_enemy', 0):.0%} of map")

            if rec:
                print(f"\n  RECOMMENDATION:")
                print(f"    Build: eco={rec['priority_economy']:.0%} def={rec['priority_defense']:.0%} mil={rec['priority_military']:.0%} tech={rec['priority_tech']:.0%}")
                print(f"    Army:  inf={rec['prefer_infantry']:.0%} veh={rec['prefer_vehicles']:.0%} air={rec['prefer_aircraft']:.0%}")
                print(f"    Aggression: {rec['aggression']:.0%}")
        else:
            # Compact output
            print(f"[{self.state_count:4d}] t={state.get('game_time', 0):5.1f}m "
                  f"${10**state.get('money', 0):6.0f} "
                  f"army={state.get('army_strength', 0):4.2f}x "
                  f"threat={state.get('base_threat', 0):3.0%} "
                  f"{'ATTACK!' if state.get('under_attack', 0) > 0.5 else ''}")

    def close(self):
        """Close the pipe."""
        if self.pipe and sys.platform == 'win32':
            win32file.CloseHandle(self.pipe)
            self.pipe = None
        self.connected = False
        self.state_count = 0


def main():
    """Main server loop."""
    parser = argparse.ArgumentParser(description='ML Bridge Server for Generals Zero Hour')
    parser.add_argument('--log', metavar='FILE', help='Log states to JSONL file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose state output')
    parser.add_argument('--no-respond', action='store_true', help='Don\'t send recommendations')
    args = parser.parse_args()

    print("=" * 60)
    print("Generals Zero Hour - ML Bridge Server")
    print("=" * 60)
    if args.log:
        print(f"Logging to: {args.log}")
    if args.verbose:
        print("Verbose mode: ON")
    if args.no_respond:
        print("Response mode: OFF (observation only)")
    print()

    logger = None
    if args.log:
        logger = StateLogger(args.log)

    server = MLBridgeServer(logger=logger, verbose=args.verbose, respond=not args.no_respond)

    while True:
        # Create pipe and wait for connection
        if not server.create_pipe():
            print("[Server] Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if not server.wait_for_connection():
            server.close()
            continue

        if logger:
            logger.open()

        print("[Server] Starting message loop...")

        # Message loop
        while server.connected:
            msg = server.read_message()
            if msg is None:
                time.sleep(0.01)  # Small sleep to avoid busy-wait
                continue

            try:
                state = json.loads(msg)

                # Generate recommendation
                rec = server.generate_recommendation(state) if server.respond else None

                # Log state
                if logger:
                    logger.log_state(state, rec)

                # Print state
                server.print_state(state, rec)

                # Send recommendation
                if server.respond and rec:
                    rec_json = json.dumps(rec)
                    server.write_message(rec_json)

            except json.JSONDecodeError as e:
                print(f"[Server] Invalid JSON: {e}")
            except Exception as e:
                print(f"[Server] Error: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n[Server] Connection closed after {server.state_count} messages")
        server.close()
        if logger:
            logger.close()
        print("[Server] Waiting for new connection...\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
