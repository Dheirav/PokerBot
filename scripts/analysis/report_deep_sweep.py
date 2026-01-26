#!/usr/bin/env python3
"""
Report the best results from deep hyperparameter sweeps.
- Scans all checkpoints/deep_* directories
- Finds the best run (by final fitness) for each config
- Outputs a summary report (deep_sweep_report.txt)

Usage:
    python scripts/report_deep_sweep.py
"""
import os
import json
from pathlib import Path

def find_latest_run_dir(config_dir):
    runs_dir = config_dir / 'runs'
    if not runs_dir.exists():
        return None
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return None
    return max(run_dirs, key=os.path.getmtime)

def get_final_fitness(state_path):
    try:
        with open(state_path) as f:
            state = json.load(f)
        # Try common keys
        for k in ['best_fitness', 'final_best_fitness', 'fitness']:
            if k in state:
                return state[k]
        # Try nested
        if 'stats' in state and 'best_fitness' in state['stats']:
            return state['stats']['best_fitness']
    except Exception as e:
        return None
    return None

def main():
    checkpoints = Path('checkpoints')
    deep_dirs = [d for d in checkpoints.glob('deep_*') if d.is_dir()]
    results = []
    for d in deep_dirs:
        latest_run = find_latest_run_dir(d)
        if not latest_run:
            print(f"[SKIP] No runs found in {d}")
            continue
        state_path = latest_run / 'state.json'
        if not state_path.exists():
            print(f"[SKIP] No state.json in {latest_run}")
            continue
        fitness = get_final_fitness(state_path)
        if fitness is None:
            print(f"[SKIP] Could not read fitness from {state_path}")
            continue
        results.append({
            'config': d.name.replace('deep_', ''),
            'run_dir': str(latest_run),
            'fitness': fitness
        })
    if not results:
        print("No deep sweep results found.")
        return
    results.sort(key=lambda x: x['fitness'], reverse=True)
    report_lines = [
        "# Deep Hyperparameter Sweep Report\n",
        f"Total configs: {len(results)}\n",
        "\nTop configs by final fitness:\n"
    ]
    for i, r in enumerate(results, 1):
        report_lines.append(f"{i}. {r['config']}\n   Fitness: {r['fitness']}\n   Run: {r['run_dir']}\n")
    report_path = Path('deep_sweep_report.txt')
    with open(report_path, 'w') as f:
        f.writelines(line + '\n' for line in report_lines)
    print(f"Report written to {report_path}")
    print('\n'.join(report_lines))

if __name__ == '__main__':
    main()
