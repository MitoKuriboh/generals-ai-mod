#!/usr/bin/env python3
"""
Model Evaluation Tool for Generals Zero Hour Learning AI

Analyzes inference logs and training runs to evaluate model performance.

Usage:
    # Analyze inference logs
    python evaluate_model.py eval.jsonl --summary

    # Compare multiple runs
    python evaluate_model.py run1.jsonl run2.jsonl --compare

    # Plot performance over time
    python evaluate_model.py training.jsonl --plot

    # Export to CSV
    python evaluate_model.py eval.jsonl --export results.csv
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np


def load_logs(path: str) -> List[Dict]:
    """Load logs from JSONL file."""
    logs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs


def analyze_inference_run(logs: List[Dict]) -> Dict:
    """Analyze a single inference run."""
    inference_logs = [l for l in logs if l.get('type') == 'inference']

    if not inference_logs:
        return {}

    # Extract data
    game_times = []
    army_strengths = []
    aggressions = []
    model_count = 0
    rules_count = 0

    for log in inference_logs:
        state = log.get('state', {})
        rec = log.get('recommendation', {})

        game_times.append(state.get('game_time', 0))
        army_strengths.append(state.get('army_strength', 1.0))
        aggressions.append(rec.get('aggression', 0.5))

        if log.get('source') == 'model':
            model_count += 1
        else:
            rules_count += 1

    return {
        'total_states': len(inference_logs),
        'duration_minutes': max(game_times) if game_times else 0,
        'model_inferences': model_count,
        'rules_inferences': rules_count,
        'model_ratio': model_count / len(inference_logs) if inference_logs else 0,
        'avg_army_strength': np.mean(army_strengths) if army_strengths else 0,
        'final_army_strength': army_strengths[-1] if army_strengths else 0,
        'avg_aggression': np.mean(aggressions) if aggressions else 0,
        'max_army_strength': max(army_strengths) if army_strengths else 0,
        'min_army_strength': min(army_strengths) if army_strengths else 0,
    }


def analyze_training_run(logs: List[Dict]) -> Dict:
    """Analyze a training run."""
    episode_logs = [l for l in logs if l.get('total_reward') is not None]

    if not episode_logs:
        return {}

    rewards = [l['total_reward'] for l in episode_logs]
    wins = [l.get('won') for l in episode_logs if l.get('won') is not None]
    steps = [l.get('steps', 0) for l in episode_logs]
    game_times = [l.get('game_time', 0) for l in episode_logs]

    # Calculate rolling statistics
    window = min(50, len(rewards))
    recent_rewards = rewards[-window:] if len(rewards) >= window else rewards
    recent_wins = [w for w in wins[-window:] if w is not None] if len(wins) >= window else [w for w in wins if w is not None]

    return {
        'total_episodes': len(episode_logs),
        'total_wins': sum(1 for w in wins if w),
        'total_losses': sum(1 for w in wins if w is False),
        'win_rate': sum(1 for w in wins if w) / len(wins) if wins else 0,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': max(rewards),
        'min_reward': min(rewards),
        'recent_avg_reward': np.mean(recent_rewards),
        'recent_win_rate': sum(recent_wins) / len(recent_wins) if recent_wins else 0,
        'avg_steps': np.mean(steps),
        'avg_game_time': np.mean(game_times),
    }


def detect_outcome(logs: List[Dict]) -> Optional[bool]:
    """Try to detect game outcome from inference logs."""
    inference_logs = [l for l in logs if l.get('type') == 'inference']

    if not inference_logs:
        return None

    # Check last few states
    last_states = [l.get('state', {}) for l in inference_logs[-5:]]

    # If we have very low structures at the end, we probably lost
    final_own_struct = last_states[-1].get('own_structures', [1, 1, 0])[0] if last_states else 1
    final_enemy_struct = last_states[-1].get('enemy_structures', [1, 1, 0])[0] if last_states else 1

    if final_own_struct < 0.3:  # Less than 1 structure
        return False  # We lost
    elif final_enemy_struct < 0.3:
        return True  # We won

    # Check army strength trend
    army_strengths = [s.get('army_strength', 1.0) for s in last_states]
    if army_strengths:
        final_strength = army_strengths[-1]
        if final_strength > 2.0:
            return True  # Probably winning
        elif final_strength < 0.3:
            return False  # Probably losing

    return None  # Can't determine


def print_summary(stats: Dict, title: str = "Summary"):
    """Print statistics summary."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    if 'total_episodes' in stats:
        # Training stats
        print(f"\n  Episodes:    {stats['total_episodes']}")
        print(f"  Win Rate:    {stats['win_rate']:.1%} ({stats['total_wins']}W/{stats['total_losses']}L)")
        print(f"\n  Rewards:")
        print(f"    Average:   {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"    Range:     [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"    Recent:    {stats['recent_avg_reward']:.2f}")
        print(f"\n  Recent Win Rate: {stats['recent_win_rate']:.1%}")
        print(f"  Avg Steps:   {stats['avg_steps']:.1f}")
        print(f"  Avg Time:    {stats['avg_game_time']:.1f} min")
    else:
        # Inference stats
        print(f"\n  States:      {stats.get('total_states', 0)}")
        print(f"  Duration:    {stats.get('duration_minutes', 0):.1f} minutes")
        print(f"\n  Inference Source:")
        print(f"    Model:     {stats.get('model_inferences', 0)} ({stats.get('model_ratio', 0):.0%})")
        print(f"    Rules:     {stats.get('rules_inferences', 0)}")
        print(f"\n  Army Strength:")
        print(f"    Average:   {stats.get('avg_army_strength', 0):.2f}x")
        print(f"    Final:     {stats.get('final_army_strength', 0):.2f}x")
        print(f"    Range:     [{stats.get('min_army_strength', 0):.2f}, {stats.get('max_army_strength', 0):.2f}]")
        print(f"\n  Avg Aggression: {stats.get('avg_aggression', 0):.0%}")

    print(f"{'='*60}\n")


def compare_runs(run_stats: List[tuple[str, Dict]]):
    """Compare multiple runs side by side."""
    print(f"\n{'='*70}")
    print(" Run Comparison")
    print(f"{'='*70}")

    # Determine columns
    headers = ['Metric'] + [name for name, _ in run_stats]
    col_width = max(15, max(len(h) for h in headers) + 2)

    # Print header
    print(f"\n{'Metric':<20}", end='')
    for name, _ in run_stats:
        print(f"{name:>{col_width}}", end='')
    print()
    print("-" * (20 + col_width * len(run_stats)))

    # Metrics to compare
    if 'total_episodes' in run_stats[0][1]:
        metrics = [
            ('Episodes', 'total_episodes', '{:.0f}'),
            ('Win Rate', 'win_rate', '{:.1%}'),
            ('Avg Reward', 'avg_reward', '{:.2f}'),
            ('Recent Reward', 'recent_avg_reward', '{:.2f}'),
            ('Recent Win%', 'recent_win_rate', '{:.1%}'),
        ]
    else:
        metrics = [
            ('States', 'total_states', '{:.0f}'),
            ('Duration', 'duration_minutes', '{:.1f}m'),
            ('Model %', 'model_ratio', '{:.0%}'),
            ('Avg Army', 'avg_army_strength', '{:.2f}x'),
            ('Final Army', 'final_army_strength', '{:.2f}x'),
        ]

    for label, key, fmt in metrics:
        print(f"{label:<20}", end='')
        for _, stats in run_stats:
            value = stats.get(key, 0)
            print(f"{fmt.format(value):>{col_width}}", end='')
        print()

    print(f"{'='*70}\n")


def plot_training_progress(logs: List[Dict], save_path: Optional[str] = None):
    """Plot training progress over episodes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    episode_logs = [l for l in logs if l.get('total_reward') is not None]
    if not episode_logs:
        print("No episode data to plot")
        return

    episodes = list(range(1, len(episode_logs) + 1))
    rewards = [l['total_reward'] for l in episode_logs]
    wins = [1 if l.get('won') else 0 for l in episode_logs]

    # Compute rolling averages
    window = min(50, len(rewards) // 4)
    if window > 1:
        rolling_reward = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rolling_win = np.convolve(wins, np.ones(window)/window, mode='valid')
        rolling_eps = episodes[window-1:]
    else:
        rolling_reward = rewards
        rolling_win = wins
        rolling_eps = episodes

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Training Progress')

    # Rewards
    axes[0].plot(episodes, rewards, 'b-', alpha=0.3, label='Episode Reward')
    axes[0].plot(rolling_eps, rolling_reward, 'b-', linewidth=2, label=f'Rolling Avg ({window})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Win rate
    axes[1].plot(rolling_eps, rolling_win, 'g-', linewidth=2)
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% Target')
    axes[1].axhline(y=0.5, color='orange', linestyle='--', label='50% Baseline')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Rate')
    axes[1].set_title(f'Rolling Win Rate (window={window})')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def export_csv(logs: List[Dict], output_path: str):
    """Export logs to CSV."""
    episode_logs = [l for l in logs if l.get('total_reward') is not None or l.get('type') == 'inference']

    if not episode_logs:
        print("No data to export")
        return

    with open(output_path, 'w') as f:
        # Determine columns based on log type
        if episode_logs[0].get('type') == 'inference':
            headers = ['seq', 'game_time', 'money', 'army_strength', 'under_attack',
                      'aggression', 'priority_military', 'source']
            f.write(','.join(headers) + '\n')

            for log in episode_logs:
                state = log.get('state', {})
                rec = log.get('recommendation', {})
                row = [
                    log.get('seq', 0),
                    state.get('game_time', 0),
                    10 ** state.get('money', 0),
                    state.get('army_strength', 1),
                    state.get('under_attack', 0),
                    rec.get('aggression', 0),
                    rec.get('priority_military', 0),
                    log.get('source', 'unknown'),
                ]
                f.write(','.join(str(x) for x in row) + '\n')
        else:
            headers = ['episode', 'reward', 'won', 'steps', 'game_time',
                      'units_killed', 'units_lost', 'army_strength']
            f.write(','.join(headers) + '\n')

            for i, log in enumerate(episode_logs):
                row = [
                    log.get('episode', i + 1),
                    log.get('total_reward', 0),
                    1 if log.get('won') else 0,
                    log.get('steps', 0),
                    log.get('game_time', 0),
                    log.get('units_killed', 0),
                    log.get('units_lost', 0),
                    log.get('final_army_strength', 0),
                ]
                f.write(','.join(str(x) for x in row) + '\n')

    print(f"Exported {len(episode_logs)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Learning AI performance')
    parser.add_argument('logs', nargs='+', help='Log files to analyze')
    parser.add_argument('--summary', '-s', action='store_true', help='Print summary statistics')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare multiple runs')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot training progress')
    parser.add_argument('--plot-save', type=str, help='Save plot to file')
    parser.add_argument('--export', '-e', type=str, help='Export to CSV')
    args = parser.parse_args()

    # Load all logs
    all_runs = []
    for path in args.logs:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        logs = load_logs(path)
        name = os.path.basename(path).replace('.jsonl', '')
        all_runs.append((name, logs))

    if not all_runs:
        print("No valid log files found")
        return

    # Analyze
    run_stats = []
    for name, logs in all_runs:
        # Detect log type
        if any(l.get('type') == 'inference' for l in logs):
            stats = analyze_inference_run(logs)
        else:
            stats = analyze_training_run(logs)
        run_stats.append((name, stats))

    # Summary
    if args.summary or (not args.compare and not args.plot and not args.export):
        for name, stats in run_stats:
            if stats:
                print_summary(stats, title=name)

    # Compare
    if args.compare and len(run_stats) > 1:
        compare_runs(run_stats)

    # Plot
    if args.plot or args.plot_save:
        for name, logs in all_runs:
            plot_training_progress(logs, save_path=args.plot_save)

    # Export
    if args.export:
        # Export first file
        export_csv(all_runs[0][1], args.export)


if __name__ == '__main__':
    main()
