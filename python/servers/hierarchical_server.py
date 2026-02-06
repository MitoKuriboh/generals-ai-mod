"""
Hierarchical Inference Server

Serves all three layers of the hierarchical architecture:
- Strategic: Macro decisions
- Tactical: Team-level decisions
- Micro: Unit-level decisions

Uses batched communication for efficiency.
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from training.model import PolicyNetwork
from training.config import STATE_DIM, ACTION_DIM, HIDDEN_DIM, PROTOCOL_VERSION
from tactical.model import TacticalNetwork
from tactical.state import TacticalState
from micro.model import MicroNetwork
from micro.state import MicroState
from hierarchical.coordinator import HierarchicalCoordinator, InferenceConfig
from hierarchical.batch_bridge import BatchedMLBridge, validate_request

# Configure logging - DEBUG level to see coordinator internals
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also set coordinator logger to DEBUG
logging.getLogger('hierarchical.coordinator').setLevel(logging.DEBUG)


class HierarchicalServer:
    """
    Server for hierarchical inference across all three layers.

    Handles batched communication with the game and coordinates
    inference across strategic, tactical, and micro layers.
    """

    def __init__(
        self,
        strategic_path: Optional[str] = None,
        tactical_path: Optional[str] = None,
        micro_path: Optional[str] = None,
        device: str = 'cpu',
        config: Optional[InferenceConfig] = None
    ):
        self.device = device
        self.config = config or InferenceConfig(device=device)

        # Load models first to determine capabilities
        logger.info("Loading models...")

        # Strategic model (required)
        self.strategic_model = PolicyNetwork(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM
        ).to(device)

        if strategic_path and Path(strategic_path).exists():
            logger.info(f"Loading strategic model from {strategic_path}")
            self.strategic_model = PolicyNetwork.load(strategic_path).to(device)
        else:
            logger.warning("Using untrained strategic model")

        self.strategic_model.eval()

        # Tactical model (optional)
        self.tactical_model = None
        if self.config.tactical_enabled:
            self.tactical_model = TacticalNetwork().to(device)
            if tactical_path and Path(tactical_path).exists():
                logger.info(f"Loading tactical model from {tactical_path}")
                self.tactical_model = TacticalNetwork.load(tactical_path).to(device)
            self.tactical_model.eval()

        # Micro model (optional)
        self.micro_model = None
        if self.config.micro_enabled:
            self.micro_model = MicroNetwork().to(device)
            if micro_path and Path(micro_path).exists():
                logger.info(f"Loading micro model from {micro_path}")
                self.micro_model = MicroNetwork.load(micro_path).to(device)
            self.micro_model.eval()

        # Initialize bridge with capabilities based on loaded models
        self.bridge = BatchedMLBridge(
            tactical_enabled=self.tactical_model is not None,
            micro_enabled=self.micro_model is not None
        )

        logger.info(f"Capabilities: hierarchical=True, tactical={self.tactical_model is not None}, "
                   f"micro={self.micro_model is not None}")

        # Create coordinator
        self.coordinator = HierarchicalCoordinator(
            strategic_model=self.strategic_model,
            tactical_model=self.tactical_model,
            micro_model=self.micro_model,
            config=self.config
        )

        # State tracking
        self.frame_count = 0
        self.message_count = 0
        self.running = False

        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'strategic_inferences': 0,
            'tactical_inferences': 0,
            'micro_inferences': 0,
            'total_latency_ms': 0.0,
            'max_latency_ms': 0.0,
        }

        logger.info("Hierarchical server initialized")

    def process_message(self, json_str: str) -> str:
        """
        Process a batched request and return batched response.

        Args:
            json_str: JSON string with batched request

        Returns:
            JSON string with batched response
        """
        t0 = time.perf_counter()

        # Handle game_end messages
        try:
            data = json.loads(json_str)
            if data.get('type') == 'game_end':
                victory = data.get('victory', False)
                game_time = data.get('game_time', 0)
                logger.info(f"Game ended: {'Victory' if victory else 'Defeat'} at {game_time:.1f}s")
                self.stats['games_played'] = self.stats.get('games_played', 0) + 1
                return json.dumps({"ack": True})
        except json.JSONDecodeError:
            pass

        # Validate request
        valid, error = validate_request(json_str)
        if not valid:
            # Log first 200 chars of message for debugging
            preview = json_str[:200] if len(json_str) > 200 else json_str
            logger.warning(f"Invalid request: {error} - Message: {preview}")
            return self._error_response(error)

        # Parse request
        try:
            request = self.bridge.parse_request(json_str)
        except Exception as e:
            logger.error(f"Failed to parse request: {e}")
            return self._error_response(str(e))

        self.stats['messages_received'] += 1
        self.frame_count = request.frame

        # Build unified game state
        game_state = self.bridge.build_game_state_from_request(request)

        # Process through coordinator
        result = self.coordinator.process_frame(game_state, request.frame)

        # Update stats
        if result['strategic']:
            self.stats['strategic_inferences'] += 1
        self.stats['tactical_inferences'] += len(result.get('teams', []))
        self.stats['micro_inferences'] += len(result.get('units', []))

        # Build response
        if result['teams'] or result['units']:
            logger.info(f"Frame {request.frame}: Returning {len(result['teams'])} tactical, {len(result['units'])} micro commands")
        response_json = self.bridge.build_response_from_result(request.frame, result)

        # Track latency
        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats['total_latency_ms'] += latency_ms
        self.stats['max_latency_ms'] = max(self.stats['max_latency_ms'], latency_ms)
        self.stats['messages_sent'] += 1

        return response_json

    def _error_response(self, error: str) -> str:
        """Generate error response JSON."""
        return json.dumps({
            'frame': self.frame_count,
            'version': PROTOCOL_VERSION,
            'error': error,
            'strategic': self.bridge._default_strategic(),
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        stats = dict(self.stats)

        if stats['messages_sent'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['messages_sent']
        else:
            stats['avg_latency_ms'] = 0.0

        # Add coordinator stats
        stats['coordinator'] = self.coordinator.get_latency_stats()
        stats['bridge'] = self.bridge.get_stats()

        return stats

    def print_stats(self):
        """Print server statistics."""
        stats = self.get_stats()
        logger.info("=" * 60)
        logger.info("Hierarchical Server Statistics")
        logger.info("=" * 60)
        logger.info(f"Messages received: {stats['messages_received']}")
        logger.info(f"Messages sent: {stats['messages_sent']}")
        logger.info(f"Strategic inferences: {stats['strategic_inferences']}")
        logger.info(f"Tactical inferences: {stats['tactical_inferences']}")
        logger.info(f"Micro inferences: {stats['micro_inferences']}")
        logger.info(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
        logger.info(f"Max latency: {stats['max_latency_ms']:.2f}ms")

        if 'coordinator' in stats:
            for layer, layer_stats in stats['coordinator'].items():
                if layer_stats.get('samples', 0) > 0:
                    logger.info(f"  {layer}: {layer_stats['mean_ms']:.3f}ms avg, "
                               f"{layer_stats.get('p99_ms', 0):.3f}ms p99")


def run_pipe_server(server: HierarchicalServer, pipe_name: str = r'\\.\pipe\generals_ml_bridge'):
    """
    Run the server using Windows named pipes.

    Args:
        server: HierarchicalServer instance
        pipe_name: Name of the Windows named pipe
    """
    try:
        import win32pipe
        import win32file
        import pywintypes
    except ImportError:
        logger.error("pywin32 not installed. Run: pip install pywin32")
        return

    logger.info(f"Starting pipe server on {pipe_name}")
    server.running = True

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        server.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while server.running:
        pipe = None
        try:
            # Create named pipe
            pipe = win32pipe.CreateNamedPipe(
                pipe_name,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1,  # Max instances
                65536,  # Out buffer size
                65536,  # In buffer size
                0,  # Default timeout
                None  # Security attributes
            )

            logger.info("Waiting for client connection...")
            win32pipe.ConnectNamedPipe(pipe, None)
            logger.info("Client connected")

            # Message loop
            while server.running:
                try:
                    # Read message length (4 bytes)
                    result, length_bytes = win32file.ReadFile(pipe, 4)
                    if result != 0 or len(length_bytes) != 4:
                        break

                    length = int.from_bytes(length_bytes, 'little')
                    if length > 1024 * 1024:  # 1MB max
                        logger.error(f"Message too large: {length}")
                        break

                    # Read message
                    result, data = win32file.ReadFile(pipe, length)
                    if result != 0:
                        break

                    # Decode with error handling to prevent crash on malformed UTF-8
                    try:
                        message = data.decode('utf-8')
                    except UnicodeDecodeError as e:
                        logger.warning(f"UTF-8 decode error: {e}, using replacement chars")
                        message = data.decode('utf-8', errors='replace')

                    # Process message
                    response = server.process_message(message)

                    # Write response atomically (length + data in single call)
                    response_bytes = response.encode('utf-8')
                    length_bytes = len(response_bytes).to_bytes(4, 'little')
                    win32file.WriteFile(pipe, length_bytes + response_bytes)

                except pywintypes.error as e:
                    if e.args[0] == 109:  # Broken pipe
                        logger.info("Client disconnected")
                    else:
                        logger.error(f"Pipe error: {e}")
                    break

        except Exception as e:
            logger.error(f"Server error: {e}")
            time.sleep(1)
        finally:
            # Cleanup pipe handle - always runs even if exception occurs
            if pipe is not None:
                try:
                    win32pipe.DisconnectNamedPipe(pipe)
                except pywintypes.error:
                    pass  # Already disconnected
                try:
                    win32file.CloseHandle(pipe)
                except pywintypes.error:
                    pass  # Already closed

    server.print_stats()
    logger.info("Server stopped")


def run_socket_server(server: HierarchicalServer, host: str = 'localhost', port: int = 5555):
    """
    Run the server using TCP sockets.

    Args:
        server: HierarchicalServer instance
        host: Host to bind to
        port: Port to listen on
    """
    import socket
    import struct

    logger.info(f"Starting socket server on {host}:{port}")
    server.running = True

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        server.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    sock.settimeout(1.0)  # Allow checking for shutdown

    while server.running:
        try:
            client, addr = sock.accept()
            logger.info(f"Client connected from {addr}")
            client.settimeout(5.0)

            while server.running:
                try:
                    # Read message length
                    length_bytes = client.recv(4)
                    if not length_bytes:
                        break

                    length = struct.unpack('<I', length_bytes)[0]
                    if length > 1024 * 1024:
                        logger.error(f"Message too large: {length}")
                        break

                    # Read message
                    data = b''
                    while len(data) < length:
                        chunk = client.recv(length - len(data))
                        if not chunk:
                            break
                        data += chunk

                    if len(data) != length:
                        break

                    # Decode with error handling to prevent crash on malformed UTF-8
                    try:
                        message = data.decode('utf-8')
                    except UnicodeDecodeError as e:
                        logger.warning(f"UTF-8 decode error: {e}, using replacement chars")
                        message = data.decode('utf-8', errors='replace')

                    # Process message
                    response = server.process_message(message)

                    # Write response atomically (length + data in single call)
                    response_bytes = response.encode('utf-8')
                    client.sendall(struct.pack('<I', len(response_bytes)) + response_bytes)

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Connection error: {e}")
                    break

            client.close()
            logger.info("Client disconnected")

        except socket.timeout:
            continue
        except Exception as e:
            logger.error(f"Accept error: {e}")

    sock.close()
    server.print_stats()
    logger.info("Server stopped")


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Inference Server')
    parser.add_argument('--strategic', type=str, default=None,
                       help='Path to strategic model checkpoint')
    parser.add_argument('--tactical', type=str, default=None,
                       help='Path to tactical model checkpoint')
    parser.add_argument('--micro', type=str, default=None,
                       help='Path to micro model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--mode', type=str, default='pipe',
                       choices=['pipe', 'socket'],
                       help='Communication mode')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host for socket mode')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port for socket mode')
    parser.add_argument('--no-tactical', action='store_true',
                       help='Disable tactical layer')
    parser.add_argument('--no-micro', action='store_true',
                       help='Disable micro layer')

    args = parser.parse_args()

    # Configure inference
    config = InferenceConfig(
        tactical_enabled=not args.no_tactical,
        micro_enabled=not args.no_micro,
        device=args.device
    )

    # Create server
    server = HierarchicalServer(
        strategic_path=args.strategic,
        tactical_path=args.tactical,
        micro_path=args.micro,
        device=args.device,
        config=config
    )

    # Run server
    if args.mode == 'pipe':
        run_pipe_server(server)
    else:
        run_socket_server(server, args.host, args.port)


if __name__ == '__main__':
    main()
