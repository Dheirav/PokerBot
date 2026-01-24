#!/usr/bin/env python3
"""
Hyperparameter sweep - finds optimal training config by testing combinations of
population size, matchups, hands, and mutation sigma.

Usage:
    python scripts/hyperparam_sweep.py --quick       # Fast sweep (6 configs, 10 gens each)
    python scripts/hyperparam_sweep.py               # Normal (12 configs, 15 gens)
    python scripts/hyperparam_sweep.py --thorough    # Thorough (many configs, 20 gens)

Analysis Tools:
    After running a sweep, analyze results with:
    
    python scripts/analyze_convergence.py
        - Identifies which configs have plateaued vs still improving
        - Shows improvement rates across different training phases
        - Warns if results are premature (configs still improving)
        - Outputs: convergence_analysis.txt
    
    python scripts/visualize_hyperparam_sweep.py
        - Generates comparison plots, heatmaps, learning curves
        - Identifies best configurations and parameter effects
        - Outputs: visualizations/*.png + analysis_report.txt
        - Requires: matplotlib, seaborn

Output Structure:
    hyperparam_results/
    └── sweep_YYYYMMDD_HHMMSS/
        ├── results.json                  # Raw results data
        ├── convergence_analysis.txt      # Convergence patterns (from analyze_convergence.py)
        └── visualizations/               # Plots (from visualize_hyperparam_sweep.py)
            ├── final_metrics_comparison.png
            ├── fitness_progression.png
            ├── hyperparameter_heatmaps.png
            ├── top_configurations.png
            └── analysis_report.txt
"""
import argparse, sys, os, json, time, numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import EvolutionTrainer, TrainingConfig, NetworkConfig, EvolutionConfig, FitnessConfig

def create_configs(mode):
    if mode == 'quick':
        return [
            (f"pop{p}_m4_h500", {'population_size': p, 'matchups_per_agent': 4, 'hands_per_matchup': 500, 'generations': 10})
            for p in [12, 20, 30]
        ] + [
            (f"p20_m{m}_h{h}", {'population_size': 20, 'matchups_per_agent': m, 'hands_per_matchup': h, 'generations': 10})
            for m, h in [(2, 1000), (4, 500), (8, 250)]
        ]
    elif mode == 'thorough':
        configs = []
        for p in [12, 20, 30, 40]:
            for m, h in [(2,1500), (3,1000), (4,750), (6,500), (8,375), (12,250)]:
                for s in [0.05, 0.1, 0.15]:
                    if 1000 <= m*h <= 6000:
                        configs.append((f"p{p}_m{m}_h{h}_s{s}", {
                            'population_size': p, 'matchups_per_agent': m, 
                            'hands_per_matchup': h, 'mutation_sigma': s, 'generations': 20
                        }))
        return configs
    else:  # normal
        configs = []
        for p in [16, 20, 24, 30]:
            for m, h in [(3,1000), (4,750), (6,500), (8,375), (10,300)]:
                if 2000 <= m*h <= 4500:
                    configs.append((f"p{p}_m{m}_h{h}", {
                        'population_size': p, 'matchups_per_agent': m, 
                        'hands_per_matchup': h, 'generations': 15
                    }))
        return configs

def run_exp(name, params, seed, out_dir):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    cfg = TrainingConfig(
        network=NetworkConfig(hidden_sizes=[64, 32]),
        evolution=EvolutionConfig(
            population_size=params.get('population_size', 20),
            mutation_sigma=params.get('mutation_sigma', 0.1)),
        fitness=FitnessConfig(
            hands_per_matchup=params.get('hands_per_matchup', 500),
            matchups_per_agent=params.get('matchups_per_agent', 4), num_workers=1),
        num_generations=params.get('generations', 15), seed=seed,
        output_dir=out_dir, experiment_name=name, checkpoint_interval=999)
    
    total_h = cfg.evolution.population_size * cfg.fitness.matchups_per_agent * cfg.fitness.hands_per_matchup
    print(f"Config: pop={cfg.evolution.population_size}, m={cfg.fitness.matchups_per_agent}, "
          f"h={cfg.fitness.hands_per_matchup}, sig={cfg.evolution.mutation_sigma}, total={total_h:,}")
    
    trainer = EvolutionTrainer(cfg)
    trainer.initialize()
    
    times, train_f, eval_f, best_p = [], [], [], []
    eval_seeds = trainer.generate_eval_hand_seeds(cfg.fitness.hands_per_matchup)
    
    try:
        for g in range(cfg.num_generations):
            t0 = time.time()
            stats = trainer.train_generation(eval_hand_seeds=eval_seeds)
            t = time.time() - t0
            times.append(t)
            train_f.append(stats['mean_fitness'])
            eval_f.append(stats.get('eval_mean', 0))
            best_p.append(trainer.best_fitness)
            print(f"Gen {g:2d} | Train: {stats['mean_fitness']:+7.1f} | Best: {trainer.best_fitness:+7.1f} | {t:.1f}s")
    except KeyboardInterrupt:
        print("\n[Stopped]")
    except Exception as e:
        print(f"\n[Error] {e}")
        return None
    
    avg_t = np.mean(times)
    conv = best_p[-1] - best_p[0] if len(best_p) > 1 else 0
    eff = conv / sum(times) if sum(times) > 0 else 0
    
    return {
        'name': name, 'config': params, 'total_hands_per_gen': total_h,
        'avg_gen_time': avg_t, 'final_best_fitness': trainer.best_fitness,
        'final_train_fitness': train_f[-1] if train_f else 0,
        'final_eval_fitness': eval_f[-1] if eval_f else 0,
        'overfitting_gap': train_f[-1] - eval_f[-1] if train_f and eval_f else 0,
        'convergence': conv, 'efficiency': eff,
        'gen_times': times, 'train_fitness': train_f, 'eval_fitness': eval_f, 'best_progress': best_p
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--thorough', action='store_true')
    parser.add_argument('--output', default='hyperparam_results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("="*70 + "\nHyperparameter Sweep\n" + "="*70)
    
    out_dir = Path(args.output) / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mode = 'quick' if args.quick else ('thorough' if args.thorough else 'normal')
    configs = create_configs(mode)
    print(f"\nMode: {mode} | Testing {len(configs)} configs | Output: {out_dir}\n")
    
    results = []
    for i, (name, params) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {name}")
        r = run_exp(name, params, args.seed, str(out_dir))
        if r:
            results.append(r)
            with open(out_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    if not results:
        print("No results"); return
    
    print("\n" + "="*70 + "\nANALYSIS\n" + "="*70)
    
    by_eff = sorted(results, key=lambda x: x['efficiency'], reverse=True)
    by_fit = sorted(results, key=lambda x: x['final_best_fitness'], reverse=True)
    by_spd = sorted(results, key=lambda x: x['avg_gen_time'])
    by_gap = sorted(results, key=lambda x: abs(x['overfitting_gap']))
    
    print("\n🏆 Top 3 by Efficiency:")
    for r in by_eff[:3]:
        print(f"  {r['name']:35s} Eff:{r['efficiency']:.4f} Best:{r['final_best_fitness']:+7.1f} Time:{r['avg_gen_time']:.1f}s")
    print("\n🎯 Top 3 by Fitness:")
    for r in by_fit[:3]:
        print(f"  {r['name']:35s} Best:{r['final_best_fitness']:+7.1f} Time:{r['avg_gen_time']:.1f}s Gap:{r['overfitting_gap']:.1f}")
    print("\n⚡ Top 3 by Speed:")
    for r in by_spd[:3]:
        print(f"  {r['name']:35s} Time:{r['avg_gen_time']:.1f}s Best:{r['final_best_fitness']:+7.1f}")
    print("\n✨ Top 3 by Low Overfitting:")
    for r in by_gap[:3]:
        print(f"  {r['name']:35s} Gap:{r['overfitting_gap']:+7.1f} Best:{r['final_best_fitness']:+7.1f}")
    
    # Score and recommend
    max_e = max(x['efficiency'] for x in results)
    max_f = max(x['final_best_fitness'] for x in results)
    min_g = min(abs(x['overfitting_gap']) for x in results)
    scored = [(0.4*(r['efficiency']/max_e) + 0.3*(min_g/max(abs(r['overfitting_gap']),1)) + 
               0.3*(r['final_best_fitness']/max_f if max_f > 0 else 0), r) for r in results]
    best = sorted(scored, reverse=True)[0][1]
    
    print("\n" + "="*70 + "\nRECOMMENDATION\n" + "="*70)
    print(f"\n✅ {best['name']}")
    print(f"\npython scripts/train.py --pop {best['config']['population_size']} "
          f"--matchups {best['config']['matchups_per_agent']} "
          f"--hands {best['config']['hands_per_matchup']} "
          + (f"--sigma {best['config']['mutation_sigma']} " if 'mutation_sigma' in best['config'] else "")
          + "--gens 100")
    print(f"\nExpected: ~{best['avg_gen_time']:.1f}s/gen, 100 gens in ~{best['avg_gen_time']*100/60:.1f} min, "
          f"{best['total_hands_per_gen']:,} hands/gen, gap: {best['overfitting_gap']:.1f}")
    
    with open(out_dir / 'report.txt', 'w') as f:
        f.write(f"SWEEP REPORT\n{'='*70}\nMode: {mode}\nConfigs: {len(results)}\n\n"
                f"Recommended: {best['name']}\nPop: {best['config']['population_size']}\n"
                f"Matchups: {best['config']['matchups_per_agent']}\nHands: {best['config']['hands_per_matchup']}\n"
                f"Avg time: {best['avg_gen_time']:.1f}s\nBest fitness: {best['final_best_fitness']:.1f}\n"
                f"Overfitting gap: {best['overfitting_gap']:.1f}\n")
    
    print(f"\n📊 Results: {out_dir}/results.json\n✅ Done!")

if __name__ == '__main__':
    main()
