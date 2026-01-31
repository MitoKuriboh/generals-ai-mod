#!/usr/bin/env python3
"""
Health Check Script for Generals AI Training Environment

Verifies that all required dependencies and paths are properly configured
before starting training. Run this first if you encounter issues.

Usage:
    python health_check.py
    python health_check.py --verbose
    python health_check.py --fix  # Attempt to fix common issues
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Callable

# Check results
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"


class HealthChecker:
    """Runs health checks and reports results."""

    def __init__(self, verbose: bool = False, fix: bool = False):
        self.verbose = verbose
        self.fix = fix
        self.results: List[Tuple[str, str, str]] = []  # (status, name, details)

    def check(self, name: str, check_fn: Callable[[], Tuple[bool, str]],
              fix_fn: Callable[[], bool] = None):
        """Run a single check."""
        try:
            success, details = check_fn()
            status = PASS if success else FAIL

            if not success and fix_fn and self.fix:
                if fix_fn():
                    success, details = check_fn()
                    if success:
                        status = PASS
                        details += " (fixed)"

            self.results.append((status, name, details))

            if self.verbose or not success:
                print(f"{status} {name}")
                if details and (self.verbose or not success):
                    print(f"       {details}")

        except Exception as e:
            self.results.append((FAIL, name, str(e)))
            print(f"{FAIL} {name}")
            print(f"       Error: {e}")

    def summary(self) -> bool:
        """Print summary and return True if all passed."""
        passed = sum(1 for s, _, _ in self.results if s == PASS)
        failed = sum(1 for s, _, _ in self.results if s == FAIL)
        warned = sum(1 for s, _, _ in self.results if s == WARN)

        print("\n" + "=" * 50)
        print(f"Health Check Summary: {passed} passed, {failed} failed, {warned} warnings")
        print("=" * 50)

        return failed == 0


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.8+."""
    version = sys.version_info
    ok = version >= (3, 8)
    return ok, f"Python {version.major}.{version.minor}.{version.micro}"


def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch is installed and functional."""
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return True, f"PyTorch {torch.__version__} ({device})"
    except ImportError:
        return False, "PyTorch not installed. Run: pip install torch"


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability (optional but recommended)."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
        else:
            return True, "CUDA not available (CPU training will be slower)"
    except Exception as e:
        return True, f"CUDA check skipped: {e}"


def check_pywin32() -> Tuple[bool, str]:
    """Check pywin32 is installed (Windows only)."""
    if sys.platform != 'win32':
        return True, "Not on Windows, pywin32 not needed"
    try:
        import win32pipe
        import win32file
        return True, "pywin32 installed"
    except ImportError:
        return False, "pywin32 not installed. Run: pip install pywin32"


def check_training_module() -> Tuple[bool, str]:
    """Check training module can be imported."""
    try:
        from training.model import PolicyNetwork, STATE_DIM
        from training.ppo import PPOAgent
        from training.rewards import calculate_step_reward
        return True, f"Training module OK (STATE_DIM={STATE_DIM})"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_checkpoint_dir() -> Tuple[bool, str]:
    """Check checkpoint directory is writable."""
    try:
        from training.config import CHECKPOINT_DIR
        path = Path(CHECKPOINT_DIR)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        # Test write
        test_file = path / ".health_check_test"
        test_file.write_text("test")
        test_file.unlink()

        return True, f"Checkpoint dir OK: {path}"
    except Exception as e:
        return False, f"Checkpoint dir error: {e}"


def check_log_dir() -> Tuple[bool, str]:
    """Check log directory is writable."""
    try:
        from training.config import LOG_DIR
        path = Path(LOG_DIR)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        # Test write
        test_file = path / ".health_check_test"
        test_file.write_text("test")
        test_file.unlink()

        return True, f"Log dir OK: {path}"
    except Exception as e:
        return False, f"Log dir error: {e}"


def check_game_exe() -> Tuple[bool, str]:
    """Check if game executable exists (Windows only)."""
    if sys.platform != 'win32':
        return True, "Not on Windows, skipping game check"

    possible_paths = [
        r"C:\Program Files (x86)\Steam\steamapps\common\Command and Conquer Generals - Zero Hour\generalszh.exe",
        r"C:\Games\Command and Conquer Generals - Zero Hour\generalszh.exe",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return True, f"Game found: {path}"

    return False, "Game executable not found. Set GENERALS_GAME_PATH environment variable."


def check_model_creation() -> Tuple[bool, str]:
    """Check model can be created and run inference."""
    try:
        import torch
        from training.model import PolicyNetwork, STATE_DIM

        model = PolicyNetwork()
        state = torch.randn(STATE_DIM)
        action, log_prob, value = model.get_action(state)

        # Verify action bounds
        if not (torch.all(action >= 0) and torch.all(action <= 1)):
            return False, f"Action out of bounds: {action}"

        if not torch.isfinite(log_prob):
            return False, f"Log prob not finite: {log_prob}"

        param_count = sum(p.numel() for p in model.parameters())
        return True, f"Model OK ({param_count:,} parameters)"
    except Exception as e:
        return False, f"Model error: {e}"


def check_pipe_name() -> Tuple[bool, str]:
    """Check pipe name is valid format."""
    try:
        from training.config import PIPE_NAME

        if not PIPE_NAME.startswith(r'\\.\pipe\\'):
            return False, f"Invalid pipe name format: {PIPE_NAME}"

        return True, f"Pipe name: {PIPE_NAME}"
    except Exception as e:
        return False, f"Config error: {e}"


def main():
    parser = argparse.ArgumentParser(description='Health check for Generals AI training')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all check details')
    parser.add_argument('--fix', action='store_true',
                        help='Attempt to fix common issues')
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Generals AI Training - Health Check")
    print("=" * 50 + "\n")

    checker = HealthChecker(verbose=args.verbose, fix=args.fix)

    # Core checks
    print("Core Dependencies:")
    checker.check("Python version", check_python_version)
    checker.check("PyTorch", check_pytorch)
    checker.check("CUDA (optional)", check_cuda)
    checker.check("pywin32 (Windows)", check_pywin32)

    print("\nTraining Module:")
    checker.check("Training imports", check_training_module)
    checker.check("Model creation", check_model_creation)

    print("\nPaths and Directories:")
    checker.check("Checkpoint directory", check_checkpoint_dir)
    checker.check("Log directory", check_log_dir)
    checker.check("Pipe name format", check_pipe_name)

    print("\nGame Integration:")
    checker.check("Game executable", check_game_exe)

    # Summary
    success = checker.summary()

    if success:
        print("\nAll checks passed! You're ready to train.")
        print("\nQuick start:")
        print("  python train_with_game.py --episodes 100")
    else:
        print("\nSome checks failed. Please fix the issues above.")
        if not args.fix:
            print("Tip: Run with --fix to attempt automatic fixes.")

    return 0 if success else 1


if __name__ == '__main__':
    # Add parent to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
