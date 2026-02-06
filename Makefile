.PHONY: train train-sim train-auto test sync status clean help

PYTHON := python3
EPISODES ?= 100

help:
	@echo "C&C Generals AI - Available commands:"
	@echo "  make train      - Start manual training (EPISODES=100)"
	@echo "  make train-sim  - Train with simulated environment"
	@echo "  make train-auto - Train with auto-launch game"
	@echo "  make test       - Run Python tests"
	@echo "  make sync       - Sync Python to Windows"
	@echo "  make status     - Show training status"
	@echo "  make clean      - Clean cache files"

train:
	cd python && $(PYTHON) train.py --mode manual --episodes $(EPISODES)

train-sim:
	cd python && $(PYTHON) train.py --mode simulated --episodes $(EPISODES)

train-auto:
	cd python && $(PYTHON) train.py --mode auto --episodes $(EPISODES)

test:
	cd python && $(PYTHON) -m pytest tests/ -v

sync:
	@if [ ! -d "/mnt/c" ]; then \
		echo "Error: /mnt/c is not mounted. Are you running in WSL with Windows drives mounted?"; \
		exit 1; \
	fi
	@if [ ! -d "/mnt/c/Users/Public/generals-ai-mod" ]; then \
		echo "Error: Target directory /mnt/c/Users/Public/generals-ai-mod does not exist."; \
		echo "Create it first: mkdir -p /mnt/c/Users/Public/generals-ai-mod/python"; \
		exit 1; \
	fi
	rsync -av --exclude __pycache__ --exclude .pytest_cache \
		--exclude "*.pyc" --exclude .coverage \
		python/ /mnt/c/Users/Public/generals-ai-mod/python/

status:
	@echo "=== Training Status ==="
	@ls -la python/checkpoints/*.pt 2>/dev/null | tail -5 || echo "No checkpoints"
	@echo ""
	@head -30 .claude-context.md 2>/dev/null || echo "No context file"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
