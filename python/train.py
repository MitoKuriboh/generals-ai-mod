#!/usr/bin/env python3
"""
Unified Training Script for C&C Generals AI

Modes:
  manual    - Wait for user to start games manually (default)
  auto      - Auto-launch game with -autoSkirmish
  simulated - Use SimulatedEnv for fast testing

Usage:
  python train.py --mode manual --episodes 100
  python train.py --mode auto --episodes 100 --headless
  python train.py --mode simulated --episodes 100
  python train.py --mode manual --resume checkpoints/best_agent.pt
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_dir: str = "logs"):
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    try:
        if sys.stdout is not None and hasattr(sys.stdout, 'write'):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    except (AttributeError, OSError) as e:
        # stdout may not be available in certain environments (e.g., daemon mode)
        logging.debug(f"Could not add console handler: {e}")

    return log_file


def main():
    parser = argparse.ArgumentParser(
        description='Unified Training for C&C Generals AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  manual    - Create named pipe, wait for game connection
              User manually starts skirmishes with Learning AI
  auto      - Auto-launch game with -autoSkirmish flags
              Supports headless mode for faster training
  simulated - Use SimulatedEnv for fast testing
              No game required, good for pipeline validation

Examples:
  python train.py --mode manual --episodes 100
  python train.py --mode auto --episodes 500 --headless
  python train.py --mode simulated --episodes 100
  python train.py --mode manual --resume checkpoints/best_agent.pt
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='manual',
                        choices=['manual', 'auto', 'simulated'],
                        help='Training mode (default: manual)')

    # Common options
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train (default: 100)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N episodes (default: 10)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output during training')

    # Auto mode options
    parser.add_argument('--game-path', type=str, default=None,
                        help='Path to generals.exe (auto mode)')
    parser.add_argument('--map', type=str, default='Alpine Assault',
                        help='Map name (auto mode, default: Alpine Assault)')
    parser.add_argument('--ai', type=int, default=0, choices=[0, 1, 2],
                        help='Enemy AI: 0=Easy, 1=Medium, 2=Hard (auto mode)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without graphics (auto mode)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (auto mode)')

    # Simulated mode options
    parser.add_argument('--sim-length', type=int, default=100,
                        help='Steps per simulated episode (default: 100)')

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging()
    logging.info(f"Training starting, log file: {log_file}")

    # Create trainer based on mode
    if args.mode == 'manual':
        # Check for Windows
        if sys.platform != 'win32':
            print("[Error] Manual mode requires Windows (for named pipes)")
            print("        Use --mode simulated for testing on Linux/WSL")
            sys.exit(1)

        from training.modes import ManualTrainer
        trainer = ManualTrainer(
            learning_rate=args.lr,
            verbose=args.verbose,
        )

    elif args.mode == 'auto':
        if sys.platform != 'win32':
            print("[Error] Auto mode requires Windows (to launch game)")
            print("        Use --mode simulated for testing on Linux/WSL")
            sys.exit(1)

        from training.modes import AutoTrainer
        trainer = AutoTrainer(
            game_path=args.game_path,
            map_name=args.map,
            ai_difficulty=args.ai,
            headless=args.headless,
            seed=args.seed,
            learning_rate=args.lr,
            verbose=args.verbose,
        )

    else:  # simulated
        from training.modes import SimulatedTrainer
        trainer = SimulatedTrainer(
            episode_length=args.sim_length,
            learning_rate=args.lr,
            verbose=args.verbose,
        )

    # Run training
    try:
        trainer.train(
            num_episodes=args.episodes,
            checkpoint_interval=args.checkpoint_interval,
            resume_path=args.resume,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
