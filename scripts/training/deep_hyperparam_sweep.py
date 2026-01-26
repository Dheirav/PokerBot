#!/usr/bin/env python3
"""
Deep Hyperparameter Sweep:
- Loads latest hyperparameter sweep results
- Selects top N best configurations (by final fitness, optionally only plateaued or still improving)
- Runs deeper sweeps (more generations, optionally more trials) for these configs

Usage:
    python scripts/deep_hyperparam_sweep.py --top 5 --generations 100 --min-status PLATEAUED

Options:
    --top N            Number of configs to extend (default: 5)
    --generations N    Number of generations for deeper run (default: 100)
    --min-status S     Minimum convergence status to include (PLATEAUED, SLOW_IMPROVEMENT, IMPROVING, STRONGLY_IMPROVING)
    --dry-run          Only print commands, do not run

This script will launch new training runs for the selected configs with the specified number of generations.
"""
import sys, os, json, argparse
from pathlib import Path
import subprocess

# Convergence status order for filtering
STATUS_ORDER = ["PLATEAUED", "SLOW_IMPROVEMENT", "IMPROVING", "STRONGLY_IMPROVING"]


def analyze_convergence(result):
    best_progress = result['best_progress']
    n_gens = len(best_progress)
    last_5_gens = min(5, n_gens)
    last_improvement = best_progress[-1] - best_progress[-last_5_gens]
    if last_improvement > 50:
        status = "STRONGLY_IMPROVING"
    elif last_improvement > 20:
        status = "IMPROVING"
    elif last_improvement > 5:
        status = "SLOW_IMPROVEMENT"
    else:
        status = "PLATEAUED"
    return status


def main():
    parser = argparse.ArgumentParser(description="Deepen best hyperparameter configs.")
    parser.add_argument('--top', type=int, default=5, help='Number of configs to extend')
    parser.add_argument('--generations', type=int, default=100, help='Generations for deeper run')
    parser.add_argument('--min-status', type=str, default='PLATEAUED', help='Minimum status to include')
    parser.add_argument('--dry-run', action='store_true', help='Only print commands, do not run')
    args = parser.parse_args()

    # Find latest sweep results
    sweep_dir = max((d for d in (Path(__file__).parent.parent / 'hyperparam_results').glob('sweep_*') if d.is_dir()), key=os.path.getmtime)
    results_path = sweep_dir / 'results.json'
    if not results_path.exists():
        print(f"No results.json found in {sweep_dir}")
        sys.exit(1)
    with open(results_path) as f:
        results = json.load(f)

    # Attach convergence status
    for r in results:
        r['convergence_status'] = analyze_convergence(r)

    # Filter by status
    min_status_idx = STATUS_ORDER.index(args.min_status)
    filtered = [r for r in results if STATUS_ORDER.index(r['convergence_status']) >= min_status_idx]
    if not filtered:
        print(f"No configs found with status >= {args.min_status}")
        sys.exit(1)

    # Sort by final_best_fitness
    filtered.sort(key=lambda r: r['final_best_fitness'], reverse=True)
    selected = filtered[:args.top]

    print(f"Selected top {len(selected)} configs for deeper runs:")
    for i, r in enumerate(selected, 1):
        print(f"{i}. {r['name']} (fitness={r['final_best_fitness']:.1f}, status={r['convergence_status']})")

    # Launch deeper runs
    for r in selected:
        params = r['config'].copy()
        params['generations'] = args.generations
        cmd = [
            sys.executable, 'scripts/train.py',
            '--pop', str(params['population_size']),
            '--matchups', str(params['matchups_per_agent']),
            '--hands', str(params['hands_per_matchup']),
            '--gens', str(params['generations'])
        ]
        if 'mutation_sigma' in params:
            cmd += ['--sigma', str(params['mutation_sigma'])]
        out_dir = f"checkpoints/deep_{r['name']}"
        cmd += ['--output', out_dir]
        print(' '.join(map(str, cmd)))
        if not args.dry_run:
            subprocess.run(cmd)

if __name__ == '__main__':
    main()
