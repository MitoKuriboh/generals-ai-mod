#!/usr/bin/env python3
"""
Manual training mode.

User manually starts skirmish games; this handles the ML training.
Can also be auto-launched by the game when Learning AI is selected.
"""

import sys
import json
import struct
import time
import logging
from typing import Optional, Dict

import torch

from .base import (
    BaseTrainer, EpisodeResult,
    wrap_recommendation_with_capabilities, validate_protocol_version
)
from ..model import state_dict_to_tensor

# Windows named pipe support
if sys.platform == 'win32':
    import win32pipe
    import win32file
    import win32event
    import win32api
    import pywintypes
    HAS_WIN32 = True
else:
    HAS_WIN32 = False

PIPE_NAME = r'\\.\pipe\generals_ml_bridge'


class ManualTrainer(BaseTrainer):
    """
    Training handler for manual game starts.

    Creates a named pipe server and waits for the game to connect.
    User starts games manually; training happens during gameplay.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe = None
        self.connected = False

    def setup(self) -> bool:
        """Create the named pipe server."""
        if not HAS_WIN32:
            print("[Error] Named pipes require Windows")
            return False

        try:
            self.pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1,  # Max instances
                4096,  # Out buffer
                4096,  # In buffer
                0,  # Default timeout
                None  # Security attributes
            )
            print(f"[Pipe] Created: {PIPE_NAME}")
            print(f"\nInstructions:")
            print(f"  1. Launch C&C Generals Zero Hour")
            print(f"  2. Start Skirmish -> Select Learning AI opponent")
            print(f"  3. Play the game (training happens automatically)")
            print(f"  4. When game ends, start another skirmish")
            print(f"  5. Press Ctrl+C to stop early\n")
            return True
        except pywintypes.error as e:
            print(f"[Error] Failed to create pipe: {e}")
            return False

    def cleanup(self):
        """Close the pipe completely."""
        if self.pipe:
            try:
                win32pipe.DisconnectNamedPipe(self.pipe)
                win32file.CloseHandle(self.pipe)
            except pywintypes.error:
                # Pipe may already be disconnected or closed
                pass
            self.pipe = None
        self.connected = False

    def _wait_for_connection(self, timeout: float = None) -> bool:
        """Wait for game to connect."""
        if not self.pipe:
            return False

        print("[Pipe] Waiting for game to connect...")
        print("       Start a skirmish with Learning AI opponent")

        event_handle = None
        try:
            overlapped = pywintypes.OVERLAPPED()
            event_handle = win32event.CreateEvent(None, True, False, None)
            if event_handle is None:
                print("[Error] Failed to create event handle")
                return False
            overlapped.hEvent = event_handle

            try:
                win32pipe.ConnectNamedPipe(self.pipe, overlapped)
            except pywintypes.error as e:
                if e.args[0] != 997:  # ERROR_IO_PENDING
                    raise

            # Wait (indefinitely if timeout is None)
            wait_time = int(timeout * 1000) if timeout else 0xFFFFFFFF  # INFINITE
            result = win32event.WaitForSingleObject(overlapped.hEvent, wait_time)

            if result == 0:  # WAIT_OBJECT_0
                self.connected = True
                print("[Pipe] Game connected!")
                return True
            else:
                print("[Pipe] Connection timeout")
                return False

        except pywintypes.error as e:
            print(f"[Error] Connection failed: {e}")
            return False
        finally:
            # Always close event handle to prevent resource leak
            if event_handle is not None:
                try:
                    win32api.CloseHandle(event_handle)
                except pywintypes.error:
                    pass  # Handle may already be closed

    def _disconnect_pipe(self):
        """Disconnect and prepare for next connection."""
        if self.pipe:
            try:
                win32pipe.DisconnectNamedPipe(self.pipe)
            except pywintypes.error:
                # Pipe may already be disconnected
                pass
        self.connected = False

    def _read_message(self) -> Optional[Dict]:
        """Read a message from the pipe."""
        if not self.connected:
            return None

        try:
            # Check if data available
            _, bytes_avail, _ = win32pipe.PeekNamedPipe(self.pipe, 0)

            if bytes_avail < 4:
                return None

            # Read length prefix
            result, length_data = win32file.ReadFile(self.pipe, 4)
            if len(length_data) < 4:
                return None

            msg_length = struct.unpack('<I', length_data)[0]

            if msg_length > 65536:
                print(f"[Error] Message too large: {msg_length}")
                return None

            # Read message data
            result, msg_data = win32file.ReadFile(self.pipe, msg_length)

            return json.loads(msg_data.decode('utf-8'))

        except pywintypes.error as e:
            if e.args[0] in (109, 232):  # Broken pipe, pipe being closed
                self.connected = False
                print("[Pipe] Disconnected")
            return None
        except json.JSONDecodeError as e:
            print(f"[Error] JSON parse error: {e}")
            return None

    def _write_message(self, data: Dict) -> bool:
        """Write a message to the pipe."""
        if not self.connected:
            return False

        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            # Write length prefix + data atomically (single call)
            length_bytes = struct.pack('<I', len(json_bytes))
            win32file.WriteFile(self.pipe, length_bytes + json_bytes)

            return True
        except pywintypes.error as e:
            print(f"[Error] Write failed: {e}")
            self.connected = False
            return False

    def run_episode(self) -> Optional[EpisodeResult]:
        """Run a single training episode."""
        # Wait for connection if not connected
        if not self.connected:
            if not self._wait_for_connection():
                return None

        states = []
        rewards = []
        self._prev_state = None

        victory = False
        game_time = 0.0
        final_army = 0.0
        game_ended = False

        print(f"\n--- Episode {self.stats.episodes_completed + 1} Started ---")

        while self.connected and not game_ended:
            # Read state from game
            state = self._read_message()

            if state is None:
                time.sleep(0.05)
                continue

            # Check for game end message
            if state.get('type') == 'game_end':
                game_ended = True
                victory = state.get('victory', False)
                game_time = state.get('game_time', 0.0)
                final_army = state.get('army_strength', 0.0)
                break

            # Validate protocol version on first state
            if len(states) == 0:
                validate_protocol_version(state)

            # Store state
            states.append(state)

            # Calculate reward
            reward = self.calculate_reward(self._prev_state, state)
            rewards.append(reward)

            # Get recommendation
            recommendation = self.get_recommendation(state)

            if self.verbose and len(states) % 50 == 0:
                print(f"  Step {len(states)}: "
                      f"army={state.get('army_strength', 1.0):.2f}x, "
                      f"time={state.get('game_time', 0):.1f}m")

            # Store transition for PPO
            self.store_transition(state, reward, done=False)

            # Send recommendation to game
            wrapped = wrap_recommendation_with_capabilities(recommendation)
            self._write_message(wrapped)

            # PPO update every 256 steps
            loss = self.maybe_update(state, threshold=256)
            if loss is not None and self.verbose:
                print(f"  PPO update: loss={loss:.4f}")

            self._prev_state = state

        # Disconnect for next game
        self._disconnect_pipe()

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
