#!/usr/bin/env python3
"""
State Analysis Tool - Analyze logged game states from ML Bridge

Usage:
    python analyze_states.py states.jsonl [--plot] [--summary] [--export CSV]

This tool helps verify that state extraction is working correctly by:
1. Showing summary statistics
2. Plotting state values over time
3. Identifying anomalies or issues
"""

import json
import argparse
import sys
from collections import defaultdict
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_states(filename):
    """Load states from a JSONL file."""
    states = []
    sessions = []
    current_session = None

    with open(filename, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entry_type = entry.get('type', 'state')

                if entry_type == 'session_start':
                    current_session = {
                        'start': entry.get('timestamp'),
                        'states': []
                    }
                elif entry_type == 'session_end':
                    if current_session:
                        current_session['end'] = entry.get('timestamp')
                        current_session['total'] = entry.get('total_states', 0)
                        sessions.append(current_session)
                        current_session = None
                elif entry_type == 'state':
                    state = entry.get('state', {})
                    state['_seq'] = entry.get('seq', len(states) + 1)
                    state['_timestamp'] = entry.get('timestamp')
                    if entry.get('recommendation'):
                        state['_recommendation'] = entry.get('recommendation')
                    states.append(state)
                    if current_session:
                        current_session['states'].append(state)
            except json.JSONDecodeError:
                continue

    return states, sessions


def print_summary(states, sessions):
    """Print summary statistics."""
    if not states:
        print("No states found in file.")
        return

    print("=" * 60)
    print("STATE ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nSessions: {len(sessions)}")
    print(f"Total states: {len(states)}")

    if sessions:
        for i, sess in enumerate(sessions):
            print(f"\n  Session {i+1}:")
            print(f"    Start: {sess.get('start', 'unknown')}")
            print(f"    End:   {sess.get('end', 'unknown')}")
            print(f"    States: {len(sess.get('states', []))}")

    # Analyze state values
    print("\n" + "-" * 60)
    print("STATE VALUE RANGES")
    print("-" * 60)

    # Collect all numeric values
    value_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0})

    for state in states:
        for key, value in state.items():
            if key.startswith('_'):
                continue
            if isinstance(value, (int, float)):
                value_ranges[key]['min'] = min(value_ranges[key]['min'], value)
                value_ranges[key]['max'] = max(value_ranges[key]['max'], value)
                value_ranges[key]['sum'] += value
                value_ranges[key]['count'] += 1
            elif isinstance(value, list) and len(value) > 0:
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        k = f"{key}[{i}]"
                        value_ranges[k]['min'] = min(value_ranges[k]['min'], v)
                        value_ranges[k]['max'] = max(value_ranges[k]['max'], v)
                        value_ranges[k]['sum'] += v
                        value_ranges[k]['count'] += 1

    print(f"\n{'Field':<25} {'Min':>10} {'Max':>10} {'Avg':>10}")
    print("-" * 57)

    for key in sorted(value_ranges.keys()):
        stats = value_ranges[key]
        if stats['count'] > 0:
            avg = stats['sum'] / stats['count']
            print(f"{key:<25} {stats['min']:>10.2f} {stats['max']:>10.2f} {avg:>10.2f}")

    # Check for potential issues
    print("\n" + "-" * 60)
    print("POTENTIAL ISSUES")
    print("-" * 60)

    issues = []

    # Check if money is always 0
    if value_ranges['money']['max'] == 0:
        issues.append("Money is always 0 - check getMoney() access")

    # Check if game_time never increases
    if value_ranges['game_time']['min'] == value_ranges['game_time']['max'] and len(states) > 1:
        issues.append("Game time is constant - check frame counter")

    # Check if all force counts are 0
    force_keys = ['own_infantry[0]', 'own_vehicles[0]', 'own_structures[0]']
    if all(value_ranges.get(k, {}).get('max', 0) == 0 for k in force_keys):
        issues.append("All own force counts are 0 - check isKindOf detection")

    # Check if army_strength is always the default
    if value_ranges.get('army_strength', {}).get('min') == value_ranges.get('army_strength', {}).get('max'):
        issues.append("Army strength is constant - check enemy detection")

    if issues:
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("  No obvious issues detected.")

    # Show first and last states
    print("\n" + "-" * 60)
    print("FIRST STATE")
    print("-" * 60)
    print_state_details(states[0])

    if len(states) > 1:
        print("\n" + "-" * 60)
        print("LAST STATE")
        print("-" * 60)
        print_state_details(states[-1])


def print_state_details(state):
    """Print detailed state information."""
    print(f"\n  Player: {state.get('player', '?')}")
    print(f"  Sequence: {state.get('_seq', '?')}")

    print(f"\n  Economy:")
    money_log = state.get('money', 0)
    print(f"    Money: {money_log:.2f} (log10) = ${10**money_log:.0f}")
    print(f"    Power: {state.get('power', 0):.0f}")

    print(f"\n  Own Forces:")
    own_inf = state.get('own_infantry', [0, 0, 0])
    own_veh = state.get('own_vehicles', [0, 0, 0])
    own_air = state.get('own_aircraft', [0, 0, 0])
    own_str = state.get('own_structures', [0, 0, 0])
    print(f"    Infantry:   {10**own_inf[0]-1:.0f} units, {own_inf[1]:.0%} health")
    print(f"    Vehicles:   {10**own_veh[0]-1:.0f} units, {own_veh[1]:.0%} health")
    print(f"    Aircraft:   {10**own_air[0]-1:.0f} units, {own_air[1]:.0%} health")
    print(f"    Structures: {10**own_str[0]-1:.0f} units, {own_str[1]:.0%} health")

    print(f"\n  Strategic:")
    print(f"    Game time:    {state.get('game_time', 0):.1f} min")
    print(f"    Tech level:   {state.get('tech_level', 0):.0%}")
    print(f"    Base threat:  {state.get('base_threat', 0):.0%}")
    print(f"    Army strength: {state.get('army_strength', 0):.2f}x")
    print(f"    Under attack: {state.get('under_attack', 0)}")


def plot_states(states, output_file=None):
    """Plot state values over time."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed. Run: pip install matplotlib")
        return

    if not states:
        print("No states to plot.")
        return

    # Extract time series
    times = [s.get('game_time', i) for i, s in enumerate(states)]
    money = [10**s.get('money', 0) for s in states]
    army = [s.get('army_strength', 1) for s in states]
    threat = [s.get('base_threat', 0) for s in states]
    own_units = [
        10**s.get('own_infantry', [0])[0] - 1 +
        10**s.get('own_vehicles', [0])[0] - 1 +
        10**s.get('own_aircraft', [0])[0] - 1
        for s in states
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Game State Over Time', fontsize=14)

    # Money
    axes[0, 0].plot(times, money, 'g-', linewidth=1)
    axes[0, 0].set_ylabel('Money ($)')
    axes[0, 0].set_xlabel('Game Time (min)')
    axes[0, 0].set_title('Economy')
    axes[0, 0].grid(True, alpha=0.3)

    # Army strength
    axes[0, 1].plot(times, army, 'b-', linewidth=1)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal')
    axes[0, 1].set_ylabel('Army Strength (ratio)')
    axes[0, 1].set_xlabel('Game Time (min)')
    axes[0, 1].set_title('Army Strength vs Enemy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Base threat
    axes[1, 0].fill_between(times, threat, alpha=0.5, color='r')
    axes[1, 0].plot(times, threat, 'r-', linewidth=1)
    axes[1, 0].set_ylabel('Base Threat')
    axes[1, 0].set_xlabel('Game Time (min)')
    axes[1, 0].set_title('Base Threat Level')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Unit count
    axes[1, 1].plot(times, own_units, 'm-', linewidth=1)
    axes[1, 1].set_ylabel('Unit Count')
    axes[1, 1].set_xlabel('Game Time (min)')
    axes[1, 1].set_title('Total Combat Units')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def export_csv(states, output_file):
    """Export states to CSV for external analysis."""
    if not states:
        print("No states to export.")
        return

    # Collect all keys
    all_keys = set()
    for state in states:
        for key, value in state.items():
            if key.startswith('_'):
                continue
            if isinstance(value, list):
                for i in range(len(value)):
                    all_keys.add(f"{key}_{i}")
            else:
                all_keys.add(key)

    keys = sorted(all_keys)

    with open(output_file, 'w') as f:
        # Header
        f.write(','.join(['seq', 'timestamp'] + keys) + '\n')

        # Data
        for state in states:
            row = [str(state.get('_seq', '')), state.get('_timestamp', '')]
            for key in keys:
                if '_' in key and key.rsplit('_', 1)[1].isdigit():
                    base_key, idx = key.rsplit('_', 1)
                    value = state.get(base_key, [])
                    if isinstance(value, list) and int(idx) < len(value):
                        row.append(str(value[int(idx)]))
                    else:
                        row.append('')
                else:
                    row.append(str(state.get(key, '')))
            f.write(','.join(row) + '\n')

    print(f"Exported {len(states)} states to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ML Bridge state logs')
    parser.add_argument('file', help='JSONL file to analyze')
    parser.add_argument('--plot', action='store_true', help='Plot state graphs')
    parser.add_argument('--plot-file', metavar='FILE', help='Save plot to file')
    parser.add_argument('--summary', action='store_true', help='Show summary (default)')
    parser.add_argument('--export', metavar='FILE', help='Export to CSV')
    args = parser.parse_args()

    # Load states
    print(f"Loading: {args.file}")
    states, sessions = load_states(args.file)
    print(f"Loaded {len(states)} states from {len(sessions)} sessions")

    # Default to summary if nothing specified
    if not args.plot and not args.export and not args.plot_file:
        args.summary = True

    if args.summary:
        print_summary(states, sessions)

    if args.plot or args.plot_file:
        plot_states(states, args.plot_file)

    if args.export:
        export_csv(states, args.export)


if __name__ == '__main__':
    main()
