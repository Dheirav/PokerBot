#!/usr/bin/env python3
"""
Hyperparameter sweep WITH Hall of Fame opponents pre-loaded.
This is critical for small populations (p12) to prevent overfitting to weak self-play opponents.

Usage:
    # P12 sweep with HoF opponents (recommended for small populations)
    python scripts/training/hyperparam_sweep_with_hof.py \
        --pop 12 \
        --matchups 6 8 10 \
        --hands 375 500 750 \
        --sigma 0.08 0.1 0.12 \
        --hof-dir checkpoints/deep_p40_m6_h500_s0.15/evolution_run \
        --gens 50
    
    # Use top tournament winners automatically
    python scripts/training/hyperparam_sweep_with_hof.py \
        --pop 12 \
        --matchups 6 8 10 \
        --hands 375 500 \
        --sigma 0.08 0.1 0.12 \
        --tournament-winners \
        --gens 50
    
    # Load multiple HoF opponents from different configs
    python scripts/training/hyperparam_sweep_with_hof.py \
        --pop 12 20 \
        --matchups 8 10 \
        --hands 375 500 \
        --sigma 0.1 \
        --hof-paths checkpoints/deep_p40_m8_h375_s0.1/evolution_run/best_genome.npy \
                     checkpoints/deep_p40_m6_h500_s0.15/evolution_run/best_genome.npy \
        --gens 50
    
    # Custom scenario with specific HoF count
    python scripts/training/hyperparam_sweep_with_hof.py \
        --pop 30 \
        --matchups 6 8 \
        --hands 500 \
        --sigma 0.1 0.15 \
        --hof-dir checkpoints/tournament_winners \
        --hof-count 5 \
        --gens 30

The HoF opponents provide strong adversaries during training, preventing overfitting
and enabling smaller populations to achieve good generalization.
"""
import sys, os, json, time, numpy as np
from datetime import datetime
from pathlib import Path
from itertools import product
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training import EvolutionTrainer, TrainingConfig, NetworkConfig, EvolutionConfig, FitnessConfig
from training.policy_network import PolicyNetwork


def find_tournament_winners(top_n=3):
    """Find top N tournament winners from tournament reports."""
    tournament_dir = Path(__file__).parent.parent.parent / 'tournament_reports'
    
    if not tournament_dir.exists():
        print(f"Warning: Tournament directory not found at {tournament_dir}")
        return []
    
    # Find all tournament report.json files
    tournament_files = list(tournament_dir.glob('tournament_*/report.json'))
    
    if not tournament_files:
        print("Warning: No tournament reports found")
        return []
    
    # Aggregate results across all tournaments
    agent_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'chips': 0, 'count': 0, 'paths': set(), 'generations': []})
    
    for report_file in tournament_files:
        try:
            with open(report_file) as f:
                data = json.load(f)
            
            for agent in data['agents']:
                name = agent['name']
                # Remove generation suffix for grouping (p40_m8_h375_s0.1_g200 -> p40_m8_h375_s0.1)
                base_name = '_'.join(name.split('_')[:4])
                
                agent_stats[base_name]['wins'] += agent['wins']
                agent_stats[base_name]['losses'] += agent['losses']
                agent_stats[base_name]['chips'] += agent['chips']
                agent_stats[base_name]['count'] += 1
                
                # Track checkpoint path and generation info
                if 'original_path' in agent:
                    agent_stats[base_name]['paths'].add(agent['original_path'])
                if 'generations' in agent:
                    agent_stats[base_name]['generations'].append(agent['generations'])
        
        except Exception as e:
            print(f"Warning: Could not read {report_file}: {e}")
    
    if not agent_stats:
        print("Warning: No agent statistics found in tournaments")
        return []
    
    # Calculate win rates and sort
    for name, stats in agent_stats.items():
        total_games = stats['wins'] + stats['losses']
        stats['win_rate'] = stats['wins'] / total_games if total_games > 0 else 0
        stats['avg_chips'] = stats['chips'] / stats['count'] if stats['count'] > 0 else 0
    
    # Sort by win rate, then by average chips
    sorted_agents = sorted(agent_stats.items(), 
                          key=lambda x: (x[1]['win_rate'], x[1]['avg_chips']), 
                          reverse=True)
    
    # Get top N winners and map to checkpoint paths
    winners = []
    for name, stats in sorted_agents[:top_n]:
        checkpoint_path = None
        
        # Try to find checkpoint from original_path in tournament data
        if stats['paths']:
            for original_path in stats['paths']:
                # Tournament reports store like: "deep_p40_m6_h500_s0.15/run_20260126_042643"
                # Actual structure: checkpoints/deep_p40_m6_h500_s0.15/runs/run_20260126_042643/best_genome.npy
                
                # Try 1: checkpoints/{original_path}/best_genome.npy
                path1 = Path('checkpoints') / original_path / 'best_genome.npy'
                if path1.exists():
                    checkpoint_path = path1
                    break
                
                # Try 2: Insert 'runs' directory - checkpoints/{config}/runs/{run}/best_genome.npy
                parts = original_path.split('/')
                if len(parts) == 2:
                    config_name, run_name = parts
                    path2 = Path('checkpoints') / config_name / 'runs' / run_name / 'best_genome.npy'
                    if path2.exists():
                        checkpoint_path = path2
                        break
        
        # Fallback: try evolution_run directory
        if not checkpoint_path:
            checkpoint_base = f"deep_{name}"
            fallback_path = Path('checkpoints') / checkpoint_base / 'evolution_run' / 'best_genome.npy'
            if fallback_path.exists():
                checkpoint_path = fallback_path
        
        if checkpoint_path:
            winners.append({
                'name': name,
                'path': str(checkpoint_path),
                'win_rate': stats['win_rate'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'avg_chips': stats['avg_chips'],
                'generations': max(stats['generations']) if stats['generations'] else None
            })
    
    return winners


def load_hof_opponents(hof_dir=None, hof_paths=None, hof_count=None):
    """Load Hall of Fame opponents from checkpoints."""
    hof_weights = []
    
    if hof_paths:
        # Load specific paths
        for path in hof_paths:
            path = Path(path)
            if path.exists():
                data = np.load(path, allow_pickle=True)
                # Handle different checkpoint formats
                if data.dtype == object and data.shape == ():
                    # Dictionary stored as object array
                    weights = data.item()
                elif data.dtype == object and len(data) == 1:
                    # Single item array containing dictionary
                    weights = data[0]
                else:
                    # Direct weights array (flat numpy array)
                    weights = data
                
                hof_weights.append(weights)
                print(f"  ✓ Loaded HoF opponent from {path}")
            else:
                print(f"  ✗ Warning: {path} not found")
    
    elif hof_dir:
        # Load from directory (best_genome.npy, population.npy, hall_of_fame.npy)
        hof_dir = Path(hof_dir)
        
        # Try best_genome.npy first
        best_genome = hof_dir / 'best_genome.npy'
        if best_genome.exists():
            data = np.load(best_genome, allow_pickle=True)
            # Handle different formats
            if data.dtype == object and data.shape == ():
                weights = data.item()
            elif data.dtype == object and len(data) == 1:
                weights = data[0]
            else:
                weights = data
            
            hof_weights.append(weights)
            print(f"  ✓ Loaded best_genome from {hof_dir}")
        
        # Try hall_of_fame.npy
        hof_file = hof_dir / 'hall_of_fame.npy'
        if hof_file.exists():
            hof_data = np.load(hof_file, allow_pickle=True)
            for i, genome_data in enumerate(hof_data):
                if hof_count and len(hof_weights) >= hof_count:
                    break
                if isinstance(genome_data, dict):
                    weights = genome_data['weights']
                else:
                    weights = genome_data
                hof_weights.append(weights)
            print(f"  ✓ Loaded {len(hof_data)} opponents from hall_of_fame.npy")
        
        # Try population.npy as fallback
        if not hof_weights:
            pop_file = hof_dir / 'population.npy'
            if pop_file.exists():
                pop_data = np.load(pop_file, allow_pickle=True)
                count = min(hof_count or 5, len(pop_data))
                for i in range(count):
                    if isinstance(pop_data[i], dict):
                        weights = pop_data[i]['weights']
                    else:
                        weights = pop_data[i]
                    hof_weights.append(weights)
                print(f"  ✓ Loaded {count} opponents from population.npy")
    
    if hof_count and len(hof_weights) > hof_count:
        hof_weights = hof_weights[:hof_count]
    
    return hof_weights


def run_training_with_hof(name, params, hof_weights, seed, out_dir):
    """Run single training configuration with HoF opponents pre-loaded."""
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    
    cfg = TrainingConfig(
        network=NetworkConfig(hidden_sizes=[64, 32]),  # Match HoF opponent architecture
        evolution=EvolutionConfig(
            population_size=params['population_size'],
            mutation_sigma=params['mutation_sigma']
        ),
        fitness=FitnessConfig(
            hands_per_matchup=params['hands_per_matchup'],
            matchups_per_agent=params['matchups_per_agent'],
            num_players=2
        ),
        num_generations=params['generations'],
        seed=seed,
        output_dir=str(out_dir),
        experiment_name=name,
        checkpoint_interval=999  # Don't checkpoint during sweep
    )
    
    # Calculate expected genome size for this architecture
    # Architecture: input_size -> hidden_sizes[0] -> hidden_sizes[1] -> output_size
    input_size = cfg.network.input_size
    hidden_sizes = cfg.network.hidden_sizes
    output_size = cfg.network.output_size
    
    # Calculate total parameters
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    expected_size = 0
    for i in range(len(layer_sizes) - 1):
        # Weights + biases for each layer
        expected_size += layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
    
    # Filter HoF opponents to match architecture
    compatible_hof = []
    for i, weights in enumerate(hof_weights):
        if len(weights) == expected_size:
            compatible_hof.append(weights)
        else:
            print(f"  ⚠️  Skipping HoF opponent {i+1}: architecture mismatch ({len(weights)} params vs {expected_size} expected)")
    
    total_hands = cfg.evolution.population_size * cfg.fitness.matchups_per_agent * cfg.fitness.hands_per_matchup
    print(f"Config: pop={cfg.evolution.population_size}, m={cfg.fitness.matchups_per_agent}, "
          f"h={cfg.fitness.hands_per_matchup}, sig={cfg.evolution.mutation_sigma}")
    print(f"Total hands/gen: {total_hands:,}, Compatible HoF opponents: {len(compatible_hof)}/{len(hof_weights)}")
    
    # Initialize trainer with compatible HoF opponents
    trainer = EvolutionTrainer(cfg)
    trainer.initialize(hof_weights=compatible_hof if compatible_hof else None)
    
    times, train_fitness, best_progress = [], [], []
    eval_seeds = trainer.generate_eval_hand_seeds(cfg.fitness.hands_per_matchup)
    
    try:
        for gen in range(cfg.num_generations):
            t0 = time.time()
            stats = trainer.train_generation(eval_hand_seeds=eval_seeds)
            elapsed = time.time() - t0
            
            times.append(elapsed)
            train_fitness.append(stats['mean_fitness'])
            best_progress.append(trainer.best_fitness)
            
            print(f"Gen {gen:3d} | Mean: {stats['mean_fitness']:+7.1f} | "
                  f"Best: {trainer.best_fitness:+7.1f} | {elapsed:.1f}s")
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        return None
    
    avg_time = np.mean(times)
    convergence = best_progress[-1] - best_progress[0] if len(best_progress) > 1 else 0
    efficiency = convergence / sum(times) if sum(times) > 0 else 0
    
    return {
        'name': name,
        'config': params,
        'total_hands_per_gen': total_hands,
        'avg_gen_time': avg_time,
        'final_best_fitness': trainer.best_fitness,
        'final_mean_fitness': train_fitness[-1] if train_fitness else 0,
        'convergence': convergence,
        'efficiency': efficiency,
        'gen_times': times,
        'train_fitness': train_fitness,
        'best_progress': best_progress,
        'hof_opponent_count': len(compatible_hof)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Hyperparameter sweep with Hall of Fame opponents pre-loaded',
        epilog="""Examples:
  # P12 sweep with strong HoF opponents
  python scripts/training/hyperparam_sweep_with_hof.py --pop 12 --matchups 6 8 10 --hands 375 500 --sigma 0.1 --hof-dir checkpoints/deep_p40_m6_h500_s0.15/evolution_run
  
  # Use top tournament winners automatically
  python scripts/training/hyperparam_sweep_with_hof.py --pop 12 --matchups 6 8 10 --hands 375 500 --sigma 0.1 --tournament-winners
  
  # Multiple population sizes with custom HoF paths
  python scripts/training/hyperparam_sweep_with_hof.py --pop 12 20 --matchups 8 --hands 500 --sigma 0.1 --hof-paths checkpoint1/best_genome.npy checkpoint2/best_genome.npy
  
  # Quick test
  python scripts/training/hyperparam_sweep_with_hof.py --pop 12 --matchups 8 --hands 500 --sigma 0.1 --hof-dir checkpoints/latest --gens 20
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pop', '--population', nargs='+', type=int, required=True,
                       help='Population sizes to test (required)')
    parser.add_argument('--matchups', nargs='+', type=int, required=True,
                       help='Matchups per agent to test (required)')
    parser.add_argument('--hands', nargs='+', type=int, required=True,
                       help='Hands per matchup to test (required)')
    parser.add_argument('--sigma', '--mutation', nargs='+', type=float, required=True,
                       help='Mutation sigma values to test (required)')
    parser.add_argument('--gens', '--generations', type=int, default=50,
                       help='Number of generations per config (default: 50)')
    
    # HoF loading options
    hof_group = parser.add_mutually_exclusive_group(required=True)
    hof_group.add_argument('--hof-dir', type=str,
                          help='Directory containing best_genome.npy, hall_of_fame.npy, or population.npy')
    hof_group.add_argument('--hof-paths', nargs='+', type=str,
                          help='Specific .npy files to load as HoF opponents')
    hof_group.add_argument('--tournament-winners', action='store_true',
                          help='Automatically load top tournament winner checkpoints')
    
    parser.add_argument('--hof-count', type=int,
                       help='Maximum number of HoF opponents to load (default: all available)')
    parser.add_argument('--output', default='hyperparam_results',
                       help='Output directory (default: hyperparam_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed base (default: 42)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HYPERPARAMETER SWEEP WITH HALL OF FAME OPPONENTS")
    print("="*70)
    
    # Load HoF opponents
    print("\n📂 Loading Hall of Fame Opponents:")
    
    # Handle --tournament-winners flag
    if args.tournament_winners:
        print("  Finding top tournament winners from tournament reports...")
        winners = find_tournament_winners(top_n=args.hof_count or 3)
        
        if not winners:
            print("\n❌ Error: Could not find tournament winners!")
            print("   Make sure tournament reports exist in tournament_reports/")
            sys.exit(1)
        
        print(f"\n  Top {len(winners)} tournament winners:")
        for i, w in enumerate(winners, 1):
            gens_info = f", {w['generations']}g" if w['generations'] else ""
            print(f"    {i}. {w['name']:<30s} Win rate: {w['win_rate']*100:.1f}% ({w['wins']}W-{w['losses']}L{gens_info})")
            print(f"       Path: {w['path']}")
        print()
        
        args.hof_paths = [w['path'] for w in winners]
        args.hof_dir = None
    
    hof_weights = load_hof_opponents(
        hof_dir=args.hof_dir,
        hof_paths=args.hof_paths,
        hof_count=args.hof_count
    )
    
    if not hof_weights:
        print("\n❌ Error: No HoF opponents loaded!")
        print("   Make sure the checkpoint paths are correct.")
        sys.exit(1)
    
    print(f"\n✅ Loaded {len(hof_weights)} HoF opponents\n")
    
    # Display parameter grid
    print("Parameter Grid:")
    print(f"  Population sizes: {args.pop}")
    print(f"  Matchups per agent: {args.matchups}")
    print(f"  Hands per matchup: {args.hands}")
    print(f"  Mutation sigma: {args.sigma}")
    print(f"  Generations: {args.gens}")
    
    # Generate all combinations
    configs = []
    for p, m, h, s in product(args.pop, args.matchups, args.hands, args.sigma):
        name = f"p{p}_m{m}_h{h}_s{s}_hof{len(hof_weights)}"
        config = {
            'population_size': p,
            'matchups_per_agent': m,
            'hands_per_matchup': h,
            'mutation_sigma': s,
            'generations': args.gens
        }
        configs.append((name, config))
    
    print(f"\nTesting {len(configs)} configurations\n")
    
    # Create output directory
    out_dir = Path(args.output) / f"sweep_hof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run sweep
    results = []
    for i, (name, params) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {name}")
        result = run_training_with_hof(name, params, hof_weights, args.seed + i, out_dir)
        if result:
            results.append(result)
            # Save incrementally
            with open(out_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    if not results:
        print("\n❌ No results collected")
        return
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    by_fitness = sorted(results, key=lambda x: x['final_best_fitness'], reverse=True)
    by_efficiency = sorted(results, key=lambda x: x['efficiency'], reverse=True)
    by_speed = sorted(results, key=lambda x: x['avg_gen_time'])
    
    print("\n🎯 Top 5 by Final Fitness:")
    for r in by_fitness[:5]:
        print(f"  {r['name']:40s} Fitness: {r['final_best_fitness']:+7.1f}  Time: {r['avg_gen_time']:.1f}s")
    
    print("\n🏆 Top 5 by Efficiency (fitness gain per second):")
    for r in by_efficiency[:5]:
        print(f"  {r['name']:40s} Efficiency: {r['efficiency']:.4f}  Fitness: {r['final_best_fitness']:+7.1f}")
    
    print("\n⚡ Top 5 by Speed:")
    for r in by_speed[:5]:
        print(f"  {r['name']:40s} Time: {r['avg_gen_time']:.1f}s  Fitness: {r['final_best_fitness']:+7.1f}")
    
    # Recommendation
    best = by_fitness[0]
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATION")
    print("="*70)
    print(f"\n✅ {best['name']}")
    print(f"\nTo train with this config:")
    print(f"python scripts/training/train.py \\")
    print(f"  --pop {best['config']['population_size']} \\")
    print(f"  --matchups {best['config']['matchups_per_agent']} \\")
    print(f"  --hands {best['config']['hands_per_matchup']} \\")
    print(f"  --sigma {best['config']['mutation_sigma']} \\")
    print(f"  --gens 200 \\")
    if args.hof_dir:
        print(f"  --hof-dir {args.hof_dir}")
    else:
        print(f"  --hof-paths {' '.join(args.hof_paths)}")
    
    print(f"\nExpected: ~{best['avg_gen_time']:.1f}s/gen, "
          f"{best['total_hands_per_gen']:,} hands/gen")
    
    # Save report
    with open(out_dir / 'report.txt', 'w') as f:
        f.write(f"HYPERPARAMETER SWEEP WITH HOF OPPONENTS\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"HoF opponents: {len(hof_weights)}\n")
        f.write(f"Population sizes: {args.pop}\n")
        f.write(f"Matchups: {args.matchups}\n")
        f.write(f"Hands: {args.hands}\n")
        f.write(f"Sigma: {args.sigma}\n")
        f.write(f"Generations: {args.gens}\n")
        f.write(f"Total configs: {len(results)}\n\n")
        f.write(f"RECOMMENDED: {best['name']}\n")
        f.write(f"  Pop: {best['config']['population_size']}\n")
        f.write(f"  Matchups: {best['config']['matchups_per_agent']}\n")
        f.write(f"  Hands: {best['config']['hands_per_matchup']}\n")
        f.write(f"  Sigma: {best['config']['mutation_sigma']}\n")
        f.write(f"  Avg time: {best['avg_gen_time']:.1f}s\n")
        f.write(f"  Best fitness: {best['final_best_fitness']:.1f}\n")
        f.write(f"  Efficiency: {best['efficiency']:.4f}\n")
    
    print(f"\n📊 Results saved to: {out_dir}")
    print(f"   - results.json (detailed data)")
    print(f"   - report.txt (summary)")
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
