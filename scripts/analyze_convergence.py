#!/usr/bin/env python3
"""
Analyze convergence patterns from hyperparameter sweep results.
Identifies which configurations have plateaued vs still improving.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, TextIO


class TeeOutput:
    """Write to both console and file simultaneously."""
    
    def __init__(self, file_path: Path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
        
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.close()


def analyze_convergence(result: Dict) -> Dict:
    """
    Analyze convergence pattern for a single configuration.
    
    Returns dict with convergence metrics.
    """
    best_progress = result['best_progress']
    n_gens = len(best_progress)
    
    # Calculate improvement in different phases
    if n_gens >= 5:
        early_improvement = best_progress[min(4, n_gens-1)] - best_progress[0]
    else:
        early_improvement = 0
    
    if n_gens >= 15:
        mid_improvement = best_progress[min(14, n_gens-1)] - best_progress[10]
    else:
        mid_improvement = 0
    
    if n_gens >= 20:
        late_improvement = best_progress[min(19, n_gens-1)] - best_progress[15]
    else:
        late_improvement = 0
    
    # Check if still improving at end
    last_5_gens = min(5, n_gens)
    last_improvement = best_progress[-1] - best_progress[-last_5_gens]
    
    # Calculate improvement rate (fitness per generation)
    total_improvement = best_progress[-1] - best_progress[0]
    avg_improvement_rate = total_improvement / n_gens if n_gens > 0 else 0
    
    # Recent improvement rate (last 25% of generations)
    recent_start = max(0, n_gens - n_gens // 4)
    recent_improvement = best_progress[-1] - best_progress[recent_start]
    recent_rate = recent_improvement / (n_gens - recent_start) if recent_start < n_gens else 0
    
    # Determine convergence status
    if last_improvement > 50:
        status = "STRONGLY_IMPROVING"
        emoji = "🚀"
    elif last_improvement > 20:
        status = "IMPROVING"
        emoji = "⚠️"
    elif last_improvement > 5:
        status = "SLOW_IMPROVEMENT"
        emoji = "📊"
    else:
        status = "PLATEAUED"
        emoji = "✓"
    
    return {
        'early_improvement': early_improvement,
        'mid_improvement': mid_improvement,
        'late_improvement': late_improvement,
        'last_improvement': last_improvement,
        'total_improvement': total_improvement,
        'avg_rate': avg_improvement_rate,
        'recent_rate': recent_rate,
        'status': status,
        'emoji': emoji,
        'n_generations': n_gens
    }


def print_convergence_analysis(results: List[Dict], top_n: int = None):
    """Print convergence analysis for all configurations."""
    
    print("\n" + "=" * 100)
    print("CONVERGENCE PATTERN ANALYSIS")
    print("=" * 100)
    print(f"\nTotal configurations: {len(results)}")
    print(f"Generations per config: {results[0]['config']['generations']}")
    
    # Analyze all configs
    analyses = []
    for r in results:
        analysis = analyze_convergence(r)
        analyses.append({
            'name': r['name'],
            'final_fitness': r['final_best_fitness'],
            'config': r['config'],
            **analysis
        })
    
    # Sort by final fitness
    analyses.sort(key=lambda x: x['final_fitness'], reverse=True)
    
    # Count by status
    status_counts = {}
    for a in analyses:
        status_counts[a['status']] = status_counts.get(a['status'], 0) + 1
    
    print("\n" + "-" * 100)
    print("CONVERGENCE STATUS SUMMARY")
    print("-" * 100)
    for status, count in sorted(status_counts.items()):
        pct = (count / len(analyses)) * 100
        print(f"  {status}: {count} configs ({pct:.1f}%)")
    
    # Display configurations
    display_count = top_n if top_n else len(analyses)
    
    print("\n" + "-" * 100)
    print(f"TOP {display_count} CONFIGURATIONS (by final fitness)")
    print("-" * 100)
    
    for i, a in enumerate(analyses[:display_count], 1):
        print(f"\n{i}. {a['emoji']} {a['name']} - {a['status']}")
        print(f"   Final Fitness: {a['final_fitness']:.1f}")
        print(f"   Configuration:")
        print(f"     pop={a['config']['population_size']}, "
              f"matchups={a['config']['matchups_per_agent']}, "
              f"hands={a['config']['hands_per_matchup']}, "
              f"sigma={a['config']['mutation_sigma']}")
        print(f"   Improvement Pattern:")
        print(f"     Early (gen 0-4):   {a['early_improvement']:>8.1f}")
        print(f"     Mid (gen 10-14):   {a['mid_improvement']:>8.1f}")
        print(f"     Late (gen 15-19):  {a['late_improvement']:>8.1f}")
        print(f"     Last 5 gens:       {a['last_improvement']:>8.1f}")
        print(f"   Improvement Rates:")
        print(f"     Average rate:      {a['avg_rate']:>8.2f} fitness/gen")
        print(f"     Recent rate:       {a['recent_rate']:>8.2f} fitness/gen")
        
        # Provide recommendation
        if a['status'] in ['STRONGLY_IMPROVING', 'IMPROVING']:
            print(f"   💡 RECOMMENDATION: Run longer! Could reach {a['final_fitness'] + a['last_improvement'] * 2:.0f}+ fitness")
        elif a['status'] == 'SLOW_IMPROVEMENT':
            print(f"   💡 RECOMMENDATION: May benefit from 10-20 more generations")
        else:
            print(f"   💡 RECOMMENDATION: Likely converged, baseline established")
    
    # Identify concerning patterns
    print("\n" + "=" * 100)
    print("RELIABILITY ANALYSIS")
    print("=" * 100)
    
    still_improving = [a for a in analyses if a['status'] in ['STRONGLY_IMPROVING', 'IMPROVING']]
    if still_improving:
        print(f"\n⚠️  WARNING: {len(still_improving)} configurations still improving significantly!")
        print("These results may be premature. Consider:")
        print("  1. Extending these specific configs to 50-100 generations")
        print("  2. Comparing only configs that have plateaued")
        print("  3. Using learning curve extrapolation")
        
        print("\nConfigs to extend:")
        for a in still_improving[:5]:
            expected_gain = a['last_improvement'] * 3  # Conservative estimate
            print(f"  - {a['name']}: "
                  f"current={a['final_fitness']:.0f}, "
                  f"potential={a['final_fitness'] + expected_gain:.0f}+ "
                  f"(+{expected_gain:.0f})")
    else:
        print("\n✓ All configurations have plateaued or nearly converged")
        print("Results are likely reliable for comparison")
    
    # Best practices recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR FUTURE SWEEPS")
    print("=" * 100)
    print("\n1. Adaptive generation count:")
    print("   - Run until improvement rate < 2 fitness/gen for 10 consecutive gens")
    print("   - Or set max generations to 50-100")
    
    print("\n2. Multiple trials:")
    print("   - Run each config 3-5 times to account for variance")
    print("   - Compare means with confidence intervals")
    
    print("\n3. Staged approach:")
    print("   - Phase 1: Quick sweep (20 gens) to eliminate poor configs")
    print("   - Phase 2: Extended runs (50+ gens) on top 5-10 configs")
    print("   - Phase 3: Deep validation (3-5 trials) on top 3 configs")
    
    print("\n" + "=" * 100)


def compare_population_sizes(results: List[Dict]):
    """Compare performance by population size."""
    
    print("\n" + "=" * 100)
    print("POPULATION SIZE COMPARISON")
    print("=" * 100)
    
    by_pop_size = {}
    for r in results:
        pop_size = r['config']['population_size']
        if pop_size not in by_pop_size:
            by_pop_size[pop_size] = []
        by_pop_size[pop_size].append(r)
    
    for pop_size in sorted(by_pop_size.keys()):
        configs = by_pop_size[pop_size]
        fitnesses = [c['final_best_fitness'] for c in configs]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        
        print(f"\nPopulation Size: {pop_size}")
        print(f"  Configs tested: {len(configs)}")
        print(f"  Avg fitness: {avg_fitness:.1f}")
        print(f"  Best fitness: {max_fitness:.1f}")
        print(f"  Worst fitness: {min_fitness:.1f}")
        print(f"  Range: {max_fitness - min_fitness:.1f}")


def main():
    # Find the most recent results file
    script_dir = Path(__file__).parent
    hyperparam_dir = script_dir.parent / 'hyperparam_results'
    
    # Look for all sweep directories
    sweep_dirs = sorted([d for d in hyperparam_dir.glob('sweep_*') if d.is_dir()], reverse=True)
    
    if not sweep_dirs:
        print("No hyperparam sweep results found!")
        sys.exit(1)
    
    # Use specified path or default to latest
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        latest_sweep = sweep_dirs[0]
        json_path = latest_sweep / 'results.json'
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        sys.exit(1)
    
    # Set up output file
    output_path = json_path.parent / 'convergence_analysis.txt'
    
    # Redirect output to both console and file
    with TeeOutput(output_path):
        print(f"Loading results from: {json_path}")
        
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} configurations")
        
        # Perform analysis
        print_convergence_analysis(results, top_n=10)
        compare_population_sizes(results)
    
    # Print completion message to console only (not in file)
    print(f"\n{'=' * 100}")
    print(f"Analysis complete! Full report saved to: {output_path}")
    print(f"{'=' * 100}\n")


if __name__ == '__main__':
    main()
