#!/usr/bin/env python3
"""
Deep Hyperparameter Sweep:
- Runs extended training for specific hyperparameter configurations
- Can either load from previous sweep results OR specify custom parameter grid

Usage:
    # From previous sweep results (top N configs)
    python scripts/training/deep_hyperparam_sweep.py --top 5 --generations 100
    
    # Custom parameter grid
    python scripts/training/deep_hyperparam_sweep.py --pop 30 40 50 --matchups 8 10 --hands 375 450 --sigma 0.1 --generations 100
    
    # Single config deep dive
    python scripts/training/deep_hyperparam_sweep.py --pop 40 --matchups 8 --hands 375 --sigma 0.1 --generations 200
    
    # Explore around tournament winner
    python scripts/training/deep_hyperparam_sweep.py --pop 40 --matchups 8 10 12 --hands 300 375 450 --sigma 0.08 0.1 0.12 --generations 100

Options:
    --top N            Number of configs from previous sweep to extend (mutually exclusive with --pop/--matchups/--hands/--sigma)
    --generations N [N2 ...] Number of generations for deeper run. Can specify multiple (e.g., --generations 50 200) (default: 100)
    --min-status S     Minimum convergence status when using --top (PLATEAUED, SLOW_IMPROVEMENT, IMPROVING, STRONGLY_IMPROVING)
    --strongly-improving-only  Only train STRONGLY_IMPROVING configs (shortcut for --min-status STRONGLY_IMPROVING)
    --pop              Population sizes to test (overrides --top)
    --matchups         Matchups per agent to test (overrides --top)
    --hands            Hands per matchup to test (overrides --top)
    --sigma            Mutation sigma values to test (overrides --top)
    --dry-run          Only print commands, do not run
"""
import sys, os, json, argparse
from pathlib import Path
import subprocess
from itertools import product

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
    parser = argparse.ArgumentParser(
        description="Deep training runs for best hyperparameter configurations.",
        epilog="""Examples:
  # Extend top 5 from previous sweep
  python scripts/training/deep_hyperparam_sweep.py --top 5 --generations 100
  
  # Train top 3 and strongly improving for both 50 and 200 generations
  python scripts/training/deep_hyperparam_sweep.py --top 3 --generations 50 200
  python scripts/training/deep_hyperparam_sweep.py --strongly-improving-only --generations 50 200
  
  # Custom parameter grid
  python scripts/training/deep_hyperparam_sweep.py --pop 30 40 --matchups 8 10 --hands 375 450 --sigma 0.1 --generations 100
  
  # Single config deep dive
  python scripts/training/deep_hyperparam_sweep.py --pop 40 --matchups 8 --hands 375 --sigma 0.1 --generations 200
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--top', type=int, help='Number of configs from previous sweep to extend. If not specified with --min-status/--strongly-improving-only, trains ALL matching configs')
    parser.add_argument('--sweep-dir', '--results-dir', type=str, help='Specific sweep directory to load results from (e.g., sweep_hof_20260127_133129). If not specified, uses the latest.')
    parser.add_argument('--generations', nargs='+', type=int, default=[100], help='Generations for deeper run. Can specify multiple (e.g., 50 200) (default: 100)')
    parser.add_argument('--min-status', type=str, default='PLATEAUED', help='Minimum status when using sweep results (default: PLATEAUED)')
    parser.add_argument('--strongly-improving-only', action='store_true', help='Only train STRONGLY_IMPROVING configs (shortcut for --min-status STRONGLY_IMPROVING)')
    parser.add_argument('--pop', '--population', nargs='+', type=int,
                       help='Population sizes to test (overrides --top)')
    parser.add_argument('--matchups', nargs='+', type=int,
                       help='Matchups per agent to test (overrides --top)')
    parser.add_argument('--hands', nargs='+', type=int,
                       help='Hands per matchup to test (overrides --top)')
    parser.add_argument('--sigma', '--mutation', nargs='+', type=float,
                       help='Mutation sigma values to test (overrides --top)')
    parser.add_argument('--dry-run', action='store_true', help='Only print commands, do not run')
    parser.add_argument('--disable-tensorflow-logs', '--disable-tf-logs', action='store_true',
                        help='Suppress TensorFlow logs in subprocess runs (saves ~2-3 sec per config)')
    
    # Hall of Fame options
    hof_group = parser.add_argument_group('Hall of Fame Pre-loading (applies to all runs)')
    hof_group.add_argument('--hof-dir', type=str, default=None,
                          help='Directory containing checkpoints to pre-load into Hall of Fame for all runs')
    hof_group.add_argument('--hof-paths', nargs='+', default=None,
                          help='Specific .npy files to pre-load into Hall of Fame for all runs')
    hof_group.add_argument('--hof-count', type=int, default=5,
                          help='Number of models to load from hof-dir (default: 5)')
    args = parser.parse_args()
    
    # Handle strongly-improving-only flag
    if args.strongly_improving_only:
        args.min_status = 'STRONGLY_IMPROVING'

    # Determine if using custom grid or previous sweep results
    using_custom_grid = any([args.pop, args.matchups, args.hands, args.sigma])
    using_sweep_results = args.top is not None or args.strongly_improving_only or args.min_status != 'PLATEAUED' or args.sweep_dir is not None
    
    if using_custom_grid:
        # Custom parameter grid mode
        if args.top:
            print("Warning: --top is ignored when using custom parameter grid")
        
        # Default values if not specified
        pop_sizes = args.pop or [40]
        matchups = args.matchups or [8]
        hands = args.hands or [375]
        sigmas = args.sigma or [0.1]
        
        print("\nCustom Parameter Grid:")
        print(f"  Population sizes: {pop_sizes}")
        print(f"  Matchups per agent: {matchups}")
        print(f"  Hands per matchup: {hands}")
        print(f"  Mutation sigma: {sigmas}")
        print(f"  Generations: {args.generations}\n")
        
        # Generate all combinations (with multiple generation counts)
        selected = []
        for p, m, h, s in product(pop_sizes, matchups, hands, sigmas):
            for gen_count in args.generations:
                name = f"p{p}_m{m}_h{h}_s{s}_g{gen_count}"
                config = {
                    'population_size': p,
                    'matchups_per_agent': m,
                    'hands_per_matchup': h,
                    'mutation_sigma': s,
                    'generations': gen_count
                }
                selected.append({'name': name, 'config': config})
        
        print(f"Testing {len(selected)} configurations:\n")
        for i, r in enumerate(selected, 1):
            print(f"{i}. {r['name']}")
    
    else:
        # Previous sweep results mode
        if not using_sweep_results:
            parser.error("Either --top/--strongly-improving-only/--sweep-dir or custom parameter grid (--pop/--matchups/--hands/--sigma) must be specified")
        
        # Find sweep results
        hyperparam_dir = Path(__file__).parent.parent.parent / 'hyperparam_results'
        if not hyperparam_dir.exists():
            print(f"No hyperparam_results directory found at {hyperparam_dir}")
            sys.exit(1)
        
        if args.sweep_dir:
            # Use specified sweep directory
            sweep_dir = hyperparam_dir / args.sweep_dir
            if not sweep_dir.exists():
                print(f"Sweep directory not found: {sweep_dir}")
                print(f"\nAvailable sweep directories:")
                for d in sorted(hyperparam_dir.glob('sweep_*'), key=os.path.getmtime, reverse=True):
                    print(f"  - {d.name}")
                sys.exit(1)
        else:
            # Find latest sweep results
            sweep_dirs = [d for d in hyperparam_dir.glob('sweep_*') if d.is_dir()]
            if not sweep_dirs:
                print(f"No sweep directories found in {hyperparam_dir}")
                sys.exit(1)
            sweep_dir = max(sweep_dirs, key=os.path.getmtime)
        
        results_path = sweep_dir / 'results.json'
        if not results_path.exists():
            print(f"No results.json found in {sweep_dir}")
            sys.exit(1)
        
        with open(results_path) as f:
            data = json.load(f)
        
        # Handle both old format (list) and new format (dict with sweep_input)
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
        else:
            results = data

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
        
        # Select top N or all if --top not specified
        if args.top:
            selected = filtered[:args.top]
        else:
            selected = filtered

        print(f"\nSelected {len(selected)} configs from {sweep_dir.name} for deeper runs:")
        for i, r in enumerate(selected, 1):
            print(f"{i}. {r['name']} (fitness={r['final_best_fitness']:.1f}, status={r['convergence_status']})")
        
        # Expand to multiple generation counts
        expanded_selected = []
        for r in selected:
            for gen_count in args.generations:
                expanded_config = r.copy()
                expanded_config['name'] = f"{r['name']}_g{gen_count}"
                expanded_config['config'] = r['config'].copy()
                expanded_config['config']['generations'] = gen_count
                expanded_selected.append(expanded_config)
        selected = expanded_selected
        
        print(f"\nExpanded to {len(selected)} runs across {len(args.generations)} generation counts: {args.generations}")

    # Launch deeper runs
    print("\n" + "="*70)
    print("LAUNCHING DEEP TRAINING RUNS")
    if args.hof_dir:
        print(f"With HOF opponents from: {args.hof_dir} (count={args.hof_count})")
    elif args.hof_paths:
        print(f"With HOF opponents: {len(args.hof_paths)} custom paths")
    print("="*70 + "\n")
    
    for idx, r in enumerate(selected, 1):
        params = r['config'].copy()
        
        # Set PYTHONPATH to project root to fix imports
        env = os.environ.copy()
        project_root = str(Path(__file__).parent.parent.parent.absolute())
        env['PYTHONPATH'] = project_root
        
        cmd = [
            sys.executable, 'scripts/training/train.py',
            '--pop', str(params['population_size']),
            '--matchups', str(params['matchups_per_agent']),
            '--hands', str(params['hands_per_matchup']),
            '--gens', str(params['generations']),
            '--checkpoint-interval', '999'  # Disable checkpointing during sweeps for speed
        ]
        if 'mutation_sigma' in params:
            cmd += ['--sigma', str(params['mutation_sigma'])]
        
        # Add TensorFlow logging suppression if requested (saves ~2-3 sec per config)
        if args.disable_tensorflow_logs:
            cmd += ['--disable-tensorflow-logs']
        out_dir = f"checkpoints/deep_{r['name']}"
        cmd += ['--output', out_dir]
        
        # Add HOF arguments if specified
        if args.hof_dir:
            cmd += ['--hof-dir', args.hof_dir, '--hof-count', str(args.hof_count)]
        elif args.hof_paths:
            cmd += ['--hof-paths'] + args.hof_paths
        
        print(f"\n[{idx}/{len(selected)}] {r['name']}")
        print(' '.join(map(str, cmd)))
        if not args.dry_run:
            result = subprocess.run(cmd, env=env, cwd=project_root)
            if result.returncode != 0:
                print(f"⚠️  Warning: Command failed with exit code {result.returncode}")
        else:
            print("  (DRY RUN - not executed)")

if __name__ == '__main__':
    main()
