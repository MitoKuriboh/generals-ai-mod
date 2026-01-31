"""
Experiment Management for Generals Zero Hour Learning AI Training

Track and manage training experiments with different hyperparameters.

Usage:
    from training.experiments import ExperimentTracker

    tracker = ExperimentTracker('experiments/')
    exp_id = tracker.create_experiment(
        name='baseline_v1',
        config={...},
        description='Initial baseline with default parameters'
    )
    tracker.log_metric(exp_id, 'win_rate', 0.65, step=100)
    tracker.finish_experiment(exp_id)
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import shutil


@dataclass
class Experiment:
    """Experiment metadata and results."""
    id: str
    name: str
    description: str
    config: Dict[str, Any]
    created_at: str
    finished_at: Optional[str] = None
    status: str = 'running'  # running, completed, failed, cancelled
    metrics: Dict[str, List[tuple]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    best_checkpoint: Optional[str] = None
    final_metrics: Dict[str, float] = field(default_factory=dict)


class ExperimentTracker:
    """Track and manage training experiments."""

    def __init__(self, experiments_dir: str = 'experiments'):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.experiments: Dict[str, Experiment] = {}
        self._load_experiments()

    def _load_experiments(self):
        """Load existing experiments from disk."""
        for name in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, name)
            if os.path.isdir(exp_path):
                meta_path = os.path.join(exp_path, 'experiment.json')
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        data = json.load(f)
                        # Convert metrics list back to dict of lists
                        metrics = {}
                        for k, v in data.get('metrics', {}).items():
                            metrics[k] = [tuple(x) for x in v]
                        data['metrics'] = metrics
                        self.experiments[name] = Experiment(**data)

    def _save_experiment(self, exp: Experiment):
        """Save experiment metadata to disk."""
        exp_dir = os.path.join(self.experiments_dir, exp.id)
        os.makedirs(exp_dir, exist_ok=True)

        meta_path = os.path.join(exp_dir, 'experiment.json')
        data = asdict(exp)
        # Convert metrics to JSON-serializable format
        data['metrics'] = {k: list(v) for k, v in data['metrics'].items()}

        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)

    def create_experiment(self, name: str, config: Dict, description: str = '') -> str:
        """Create a new experiment."""
        # Generate unique ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{name}_{timestamp}"

        exp = Experiment(
            id=exp_id,
            name=name,
            description=description,
            config=config,
            created_at=datetime.now().isoformat(),
        )

        self.experiments[exp_id] = exp
        self._save_experiment(exp)

        # Create subdirectories
        exp_dir = os.path.join(self.experiments_dir, exp_id)
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)

        print(f"[Experiment] Created: {exp_id}")
        return exp_id

    def log_metric(self, exp_id: str, name: str, value: float, step: int):
        """Log a metric value."""
        if exp_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")

        exp = self.experiments[exp_id]
        if name not in exp.metrics:
            exp.metrics[name] = []
        exp.metrics[name].append((step, value, time.time()))
        self._save_experiment(exp)

    def log_metrics(self, exp_id: str, metrics: Dict[str, float], step: int):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(exp_id, name, value, step)

    def add_note(self, exp_id: str, note: str):
        """Add a note to experiment."""
        if exp_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")

        exp = self.experiments[exp_id]
        timestamped_note = f"[{datetime.now().isoformat()}] {note}"
        exp.notes.append(timestamped_note)
        self._save_experiment(exp)

    def save_checkpoint(self, exp_id: str, checkpoint_path: str, is_best: bool = False):
        """Save checkpoint reference."""
        if exp_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")

        exp = self.experiments[exp_id]
        exp_dir = os.path.join(self.experiments_dir, exp_id)
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')

        # Copy checkpoint to experiment directory
        dest = os.path.join(checkpoint_dir, os.path.basename(checkpoint_path))
        if os.path.exists(checkpoint_path):
            shutil.copy2(checkpoint_path, dest)

        if is_best:
            exp.best_checkpoint = dest
            self._save_experiment(exp)

    def finish_experiment(self, exp_id: str, status: str = 'completed',
                          final_metrics: Optional[Dict[str, float]] = None):
        """Mark experiment as finished."""
        if exp_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")

        exp = self.experiments[exp_id]
        exp.finished_at = datetime.now().isoformat()
        exp.status = status
        if final_metrics:
            exp.final_metrics = final_metrics
        self._save_experiment(exp)

        print(f"[Experiment] Finished: {exp_id} ({status})")

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.experiments.get(exp_id)

    def list_experiments(self, status: Optional[str] = None) -> List[Experiment]:
        """List all experiments, optionally filtered by status."""
        exps = list(self.experiments.values())
        if status:
            exps = [e for e in exps if e.status == status]
        return sorted(exps, key=lambda e: e.created_at, reverse=True)

    def get_best_experiment(self, metric: str = 'win_rate') -> Optional[Experiment]:
        """Get experiment with best final metric value."""
        best = None
        best_value = float('-inf')

        for exp in self.experiments.values():
            if exp.status == 'completed' and metric in exp.final_metrics:
                if exp.final_metrics[metric] > best_value:
                    best_value = exp.final_metrics[metric]
                    best = exp

        return best

    def compare_experiments(self, exp_ids: List[str], metrics: List[str]) -> Dict:
        """Compare metrics across experiments."""
        comparison = {'experiments': [], 'metrics': {m: [] for m in metrics}}

        for exp_id in exp_ids:
            exp = self.experiments.get(exp_id)
            if not exp:
                continue

            comparison['experiments'].append(exp_id)
            for metric in metrics:
                if metric in exp.final_metrics:
                    comparison['metrics'][metric].append(exp.final_metrics[metric])
                else:
                    comparison['metrics'][metric].append(None)

        return comparison

    def print_summary(self, exp_id: str):
        """Print experiment summary."""
        exp = self.experiments.get(exp_id)
        if not exp:
            print(f"Unknown experiment: {exp_id}")
            return

        print(f"\n{'='*60}")
        print(f" Experiment: {exp.name}")
        print(f"{'='*60}")
        print(f"  ID:          {exp.id}")
        print(f"  Status:      {exp.status}")
        print(f"  Created:     {exp.created_at}")
        if exp.finished_at:
            print(f"  Finished:    {exp.finished_at}")
        print(f"  Description: {exp.description or 'N/A'}")

        if exp.final_metrics:
            print(f"\n  Final Metrics:")
            for k, v in exp.final_metrics.items():
                print(f"    {k}: {v:.4f}")

        if exp.best_checkpoint:
            print(f"\n  Best Checkpoint: {exp.best_checkpoint}")

        if exp.notes:
            print(f"\n  Notes:")
            for note in exp.notes[-5:]:  # Last 5 notes
                print(f"    {note}")

        print(f"{'='*60}\n")

    def print_all_experiments(self):
        """Print summary of all experiments."""
        print(f"\n{'='*70}")
        print(" Experiments")
        print(f"{'='*70}")

        if not self.experiments:
            print("  No experiments found.")
            print(f"{'='*70}\n")
            return

        print(f"\n  {'ID':<30} {'Status':<12} {'Win Rate':<10} {'Created':<20}")
        print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*20}")

        for exp in self.list_experiments():
            win_rate = exp.final_metrics.get('win_rate', None)
            win_str = f"{win_rate:.1%}" if win_rate is not None else "N/A"
            created = exp.created_at[:19].replace('T', ' ')
            print(f"  {exp.id:<30} {exp.status:<12} {win_str:<10} {created:<20}")

        print(f"{'='*70}\n")


# Predefined experiment configurations
EXPERIMENT_PRESETS = {
    'baseline': {
        'description': 'Baseline configuration with default parameters',
        'ppo': {
            'lr': 3e-4,
            'clip_epsilon': 0.2,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'entropy_coef': 0.01,
        },
        'training': {
            'episodes': 1000,
            'steps_per_update': 512,
        },
        'reward': 'balanced',
    },
    'exploration': {
        'description': 'High exploration for initial training',
        'ppo': {
            'lr': 5e-4,
            'clip_epsilon': 0.3,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'entropy_coef': 0.05,  # Higher entropy
        },
        'training': {
            'episodes': 500,
            'steps_per_update': 256,
        },
        'reward': 'exploration',
    },
    'fine_tune': {
        'description': 'Fine-tuning with lower learning rate',
        'ppo': {
            'lr': 1e-4,
            'clip_epsilon': 0.1,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'entropy_coef': 0.005,
        },
        'training': {
            'episodes': 500,
            'steps_per_update': 512,
        },
        'reward': 'balanced',
    },
    'aggressive': {
        'description': 'Training for aggressive playstyle',
        'ppo': {
            'lr': 3e-4,
            'clip_epsilon': 0.2,
            'gamma': 0.995,  # Higher gamma for longer-term thinking
            'gae_lambda': 0.95,
            'entropy_coef': 0.01,
        },
        'training': {
            'episodes': 1000,
            'steps_per_update': 512,
        },
        'reward': 'aggressive',
    },
}


def get_preset(name: str) -> Dict:
    """Get experiment preset by name."""
    return EXPERIMENT_PRESETS.get(name, EXPERIMENT_PRESETS['baseline'])


if __name__ == '__main__':
    # Test experiment tracking
    print("Testing ExperimentTracker...")

    tracker = ExperimentTracker('/tmp/test_experiments')

    # Create experiment
    exp_id = tracker.create_experiment(
        name='test_run',
        config=get_preset('baseline'),
        description='Test experiment'
    )

    # Log some metrics
    for step in range(0, 100, 10):
        tracker.log_metrics(exp_id, {
            'reward': step * 0.1 + 5,
            'win_rate': min(0.5 + step * 0.005, 0.95),
        }, step)

    # Add note
    tracker.add_note(exp_id, 'Training started successfully')

    # Finish
    tracker.finish_experiment(exp_id, final_metrics={
        'reward': 15.0,
        'win_rate': 0.95,
    })

    # Print summary
    tracker.print_summary(exp_id)
    tracker.print_all_experiments()

    print("ExperimentTracker test passed!")
