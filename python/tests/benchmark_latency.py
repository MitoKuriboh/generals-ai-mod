#!/usr/bin/env python3
"""
Benchmark script for ML Bridge performance.

Measures:
- Decision latency (model inference time)
- Pipe throughput (simulated message passing)
- End-to-end roundtrip time

Usage:
    python python/tests/benchmark_latency.py
    python python/tests/benchmark_latency.py --iterations 1000
"""

import argparse
import json
import time
import statistics
from typing import List, Dict

import torch

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model import PolicyNetwork
from training.config import STATE_DIM, ACTION_DIM


def benchmark_inference(model: PolicyNetwork, iterations: int = 500) -> Dict[str, float]:
    """Benchmark model inference latency."""
    model.eval()

    # Generate random states
    states = [torch.randn(STATE_DIM) for _ in range(iterations)]

    latencies = []
    with torch.no_grad():
        for state in states:
            start = time.perf_counter()
            _ = model(state)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
    }


def benchmark_serialization(iterations: int = 500) -> Dict[str, float]:
    """Benchmark JSON serialization (simulates pipe protocol)."""
    # Sample game state
    sample_state = {
        "player": 1,
        "money": 3.5,
        "power": 50,
        "income": 10,
        "supply": 0.7,
        "own_infantry": [1.2, 0.9, 0],
        "own_vehicles": [0.8, 0.95, 0],
        "own_aircraft": [0.3, 1.0, 0],
        "own_structures": [1.0, 0.8, 0],
        "enemy_infantry": [0.9, 0.8, 0],
        "enemy_vehicles": [0.6, 0.9, 0],
        "enemy_aircraft": [0.0, 0.0, 0],
        "enemy_structures": [0.7, 0.9, 0],
        "game_time": 5.2,
        "tech_level": 0.4,
        "base_threat": 0.1,
        "army_strength": 1.3,
        "under_attack": 0,
        "distance_to_enemy": 0.6
    }

    sample_action = {
        "priority_economy": 0.25,
        "priority_defense": 0.25,
        "priority_military": 0.25,
        "priority_tech": 0.25,
        "prefer_infantry": 0.33,
        "prefer_vehicles": 0.34,
        "prefer_aircraft": 0.33,
        "aggression": 0.5,
        "target_player": -1
    }

    # Benchmark encode (state -> bytes)
    encode_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        data = json.dumps(sample_state).encode('utf-8')
        _ = len(data).to_bytes(4, 'little') + data
        end = time.perf_counter()
        encode_times.append((end - start) * 1000)

    # Benchmark decode (bytes -> dict)
    encoded = json.dumps(sample_state).encode('utf-8')
    decode_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = json.loads(encoded.decode('utf-8'))
        end = time.perf_counter()
        decode_times.append((end - start) * 1000)

    return {
        "encode_mean_ms": statistics.mean(encode_times),
        "encode_median_ms": statistics.median(encode_times),
        "decode_mean_ms": statistics.mean(decode_times),
        "decode_median_ms": statistics.median(decode_times),
        "message_size_bytes": len(encoded),
    }


def benchmark_roundtrip(model: PolicyNetwork, iterations: int = 500) -> Dict[str, float]:
    """Benchmark full roundtrip: deserialize -> infer -> serialize."""
    model.eval()

    sample_state_json = json.dumps({
        "player": 1, "money": 3.5, "power": 50, "income": 10, "supply": 0.7,
        "own_infantry": [1.2, 0.9, 0], "own_vehicles": [0.8, 0.95, 0],
        "own_aircraft": [0.3, 1.0, 0], "own_structures": [1.0, 0.8, 0],
        "enemy_infantry": [0.9, 0.8, 0], "enemy_vehicles": [0.6, 0.9, 0],
        "enemy_aircraft": [0.0, 0.0, 0], "enemy_structures": [0.7, 0.9, 0],
        "game_time": 5.2, "tech_level": 0.4, "base_threat": 0.1,
        "army_strength": 1.3, "under_attack": 0, "distance_to_enemy": 0.6
    }).encode('utf-8')

    roundtrip_times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()

            # Decode
            state_dict = json.loads(sample_state_json.decode('utf-8'))

            # Convert to tensor (simplified - real code extracts all features)
            state_tensor = torch.tensor([
                state_dict["money"], state_dict["power"], state_dict["income"],
                state_dict["supply"], state_dict["game_time"], state_dict["tech_level"],
                state_dict["base_threat"], state_dict["army_strength"],
            ] + [0.0] * (STATE_DIM - 8), dtype=torch.float32)

            # Inference
            action_mean, _, _ = model(state_tensor)

            # Encode response
            action_dict = {
                "priority_economy": float(action_mean[0]),
                "priority_defense": float(action_mean[1]),
                "priority_military": float(action_mean[2]),
                "priority_tech": float(action_mean[3]),
                "prefer_infantry": float(action_mean[4]),
                "prefer_vehicles": float(action_mean[5]),
                "prefer_aircraft": float(action_mean[6]),
                "aggression": float(action_mean[7]),
            }
            _ = json.dumps(action_dict).encode('utf-8')

            end = time.perf_counter()
            roundtrip_times.append((end - start) * 1000)

    return {
        "mean_ms": statistics.mean(roundtrip_times),
        "median_ms": statistics.median(roundtrip_times),
        "std_ms": statistics.stdev(roundtrip_times) if len(roundtrip_times) > 1 else 0,
        "p95_ms": sorted(roundtrip_times)[int(len(roundtrip_times) * 0.95)],
        "p99_ms": sorted(roundtrip_times)[int(len(roundtrip_times) * 0.99)],
        "max_decisions_per_sec": 1000.0 / statistics.mean(roundtrip_times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ML Bridge performance")
    parser.add_argument("--iterations", type=int, default=500, help="Number of iterations per benchmark")
    args = parser.parse_args()

    print(f"ML Bridge Performance Benchmark ({args.iterations} iterations)")
    print("=" * 60)

    # Initialize model
    print("\nInitializing model...")
    model = PolicyNetwork(STATE_DIM, ACTION_DIM, hidden_dim=256)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Inference benchmark
    print("\n1. Model Inference Latency")
    print("-" * 40)
    inference_results = benchmark_inference(model, args.iterations)
    print(f"  Mean:   {inference_results['mean_ms']:.3f} ms")
    print(f"  Median: {inference_results['median_ms']:.3f} ms")
    print(f"  P95:    {inference_results['p95_ms']:.3f} ms")
    print(f"  P99:    {inference_results['p99_ms']:.3f} ms")

    # Serialization benchmark
    print("\n2. JSON Serialization")
    print("-" * 40)
    serial_results = benchmark_serialization(args.iterations)
    print(f"  Encode: {serial_results['encode_mean_ms']:.3f} ms")
    print(f"  Decode: {serial_results['decode_mean_ms']:.3f} ms")
    print(f"  Message size: {serial_results['message_size_bytes']} bytes")

    # Roundtrip benchmark
    print("\n3. Full Roundtrip (decode -> infer -> encode)")
    print("-" * 40)
    roundtrip_results = benchmark_roundtrip(model, args.iterations)
    print(f"  Mean:   {roundtrip_results['mean_ms']:.3f} ms")
    print(f"  Median: {roundtrip_results['median_ms']:.3f} ms")
    print(f"  P95:    {roundtrip_results['p95_ms']:.3f} ms")
    print(f"  P99:    {roundtrip_results['p99_ms']:.3f} ms")
    print(f"  Max throughput: {roundtrip_results['max_decisions_per_sec']:.0f} decisions/sec")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("-" * 40)
    game_interval_ms = 1000 / 30 * 30  # 30 frames at 30 FPS = 1000ms
    headroom = game_interval_ms - roundtrip_results['p99_ms']
    print(f"  Game decision interval: {game_interval_ms:.0f} ms")
    print(f"  P99 roundtrip latency:  {roundtrip_results['p99_ms']:.1f} ms")
    print(f"  Headroom:               {headroom:.0f} ms ({headroom/game_interval_ms*100:.0f}%)")

    if headroom > 500:
        print("\n  ✓ Excellent: >500ms headroom")
    elif headroom > 100:
        print("\n  ✓ Good: >100ms headroom")
    else:
        print("\n  ⚠ Warning: Tight headroom, may cause frame drops")

    print()


if __name__ == "__main__":
    main()
