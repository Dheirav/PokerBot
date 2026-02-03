  #!/usr/bin/env python3
"""
Analyze convergence patterns from hyperparameter sweep results.
Identifies which configurations have plateaued vs still improving.

Usage:
    # Analyze latest sweep
    python scripts/analysis/analyze_convergence.py
    
    # Analyze specific sweep directory
    python scripts/analysis/analyze_convergence.py hyperparam_results/sweep_20260127_123456
    
    # Analyze specific results.json file
    python scripts/analysis/analyze_convergence.py hyperparam_results/sweep_20260127_123456/results.json
"""

import json
import sys
import argparse
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
    Dynamically adjusts analysis windows based on total generations.
    
    Returns dict with convergence metrics.
    """
    best_progress = result.get('best_progress', [])
    n_gens = len(best_progress)
    if n_gens == 0:
        raise ValueError(f"SKIP: {result.get('name', '<unnamed>')} has no best_progress data.")

    # Calculate improvement in different phases (dynamically sized)
    # Early: first 10% or 5 gens, whichever is larger
    early_window = max(5, n_gens // 10)
    if n_gens >= early_window:
        early_improvement = best_progress[early_window - 1] - best_progress[0]
    else:
        early_improvement = 0

    # Mid: middle 10% window
    mid_start = n_gens // 2 - early_window // 2
    mid_end = mid_start + early_window
    if mid_end <= n_gens:
        mid_improvement = best_progress[mid_end - 1] - best_progress[mid_start]
    else:
        mid_improvement = 0

    # Late: last 20% window
    late_window = max(5, n_gens // 5)
    late_start = n_gens - late_window
    if late_start >= 0:
        late_improvement = best_progress[-1] - best_progress[late_start]
    else:
        late_improvement = 0

    # Check if still improving at end (last 10% or min 5 gens)
    last_window = max(5, n_gens // 10)
    last_improvement = best_progress[-1] - best_progress[-last_window]

    # Calculate improvement rate (fitness per generation)
    total_improvement = best_progress[-1] - best_progress[0]
    avg_improvement_rate = total_improvement / n_gens if n_gens > 0 else 0

    # Recent improvement rate (last 25% of generations)
    recent_start = max(0, n_gens - n_gens // 4)
    recent_improvement = best_progress[-1] - best_progress[recent_start]
    recent_gens = n_gens - recent_start
    recent_rate = recent_improvement / recent_gens if recent_gens > 0 else 0

    # Determine convergence status (scaled by total generations)
    # For longer runs, require proportionally more improvement to be "improving"
    scale_factor = min(1.0, n_gens / 50.0)  # Scale thresholds for runs < 50 gens
    
    # Thresholds scale with generation count
    strong_threshold = 50 * scale_factor
    improving_threshold = 20 * scale_factor
    slow_threshold = 5 * scale_factor
    
    if last_improvement > strong_threshold:
        status = "STRONGLY_IMPROVING"
        emoji = "🚀"
    elif last_improvement > improving_threshold:
        status = "IMPROVING"
        emoji = "⚠️"
    elif last_improvement > slow_threshold:
        status = "SLOW_IMPROVEMENT"
        emoji = "📊"
    else:
        status = "PLATEAUED"
        emoji = "✓"

    return {
        'early_improvement': early_improvement,
        'early_window': early_window,
        'mid_improvement': mid_improvement,
        'mid_start': mid_start,
        'mid_end': mid_end,
        'late_improvement': late_improvement,
        'late_window': late_window,
        'last_improvement': last_improvement,
        'last_window': last_window,
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
    skipped = 0
    for r in results:
        try:
            analysis = analyze_convergence(r)
            analyses.append({
                'name': r['name'],
                'final_fitness': r['final_best_fitness'],
                'config': r['config'],
                **analysis
            })
        except Exception as e:
            print(f"[SKIP] {r.get('name', '<unnamed>')}: {e}")
            skipped += 1
    if skipped:
        print(f"\n[INFO] Skipped {skipped} configs with missing or empty progress data.")
    
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
        print(f"   Total Generations: {a['n_generations']}")
        print(f"   Configuration:")
        print(f"     pop={a['config']['population_size']}, "
              f"matchups={a['config']['matchups_per_agent']}, "
              f"hands={a['config']['hands_per_matchup']}, "
              f"sigma={a['config']['mutation_sigma']}")
        print(f"   Improvement Pattern:")
        print(f"     Early (gen 0-{a['early_window']}):   {a['early_improvement']:>8.1f}")
        print(f"     Mid (gen {a['mid_start']}-{a['mid_end']}):   {a['mid_improvement']:>8.1f}")
        print(f"     Late (last {a['late_window']} gens):  {a['late_improvement']:>8.1f}")
        print(f"     Last {a['last_window']} gens:       {a['last_improvement']:>8.1f}")
        print(f"   Improvement Rates:")
        print(f"     Average rate:      {a['avg_rate']:>8.2f} fitness/gen")
        print(f"     Recent rate:       {a['recent_rate']:>8.2f} fitness/gen")
        
        # Provide recommendation based on status
        if a['status'] in ['STRONGLY_IMPROVING', 'IMPROVING']:
            extra_gens = max(50, a['n_generations'])
            expected_gain = a['recent_rate'] * extra_gens
            print(f"   💡 RECOMMENDATION: Train {extra_gens}+ more generations (expected gain: +{expected_gain:.1f})")
        elif a['status'] == 'SLOW_IMPROVEMENT':
            print(f"   💡 RECOMMENDATION: Train 20-30 more generations to confirm plateau")
        else:
            print(f"   💡 RECOMMENDATION: Converged - ready for tournament evaluation")
    
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
    parser = argparse.ArgumentParser(
        description='Analyze convergence patterns from hyperparameter sweep results',
        epilog="""Examples:
  # Analyze latest sweep
  python scripts/analysis/analyze_convergence.py
  
  # Analyze specific sweep directory
  python scripts/analysis/analyze_convergence.py hyperparam_results/sweep_20260127_123456
  
  # Analyze specific results.json file
  python scripts/analysis/analyze_convergence.py hyperparam_results/sweep_20260127_123456/results.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('sweep_path', nargs='?', type=str,
                       help='Path to sweep directory or results.json file (default: latest sweep)')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top configurations to display (default: 10)')
    args = parser.parse_args()
    
    # Find the most recent results file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    hyperparam_dir = project_root / 'hyperparam_results'
    
    if args.sweep_path:
        # User specified a path
        sweep_path = Path(args.sweep_path)
        
        # Handle relative paths
        if not sweep_path.is_absolute():
            sweep_path = project_root / sweep_path
        
        # Check if it's a directory or file
        if sweep_path.is_dir():
            json_path = sweep_path / 'results.json'
        elif sweep_path.is_file() and sweep_path.name == 'results.json':
            json_path = sweep_path
        else:
            print(f"Error: {sweep_path} is not a valid sweep directory or results.json file!")
            sys.exit(1)
    else:
        # Use latest sweep
        sweep_dirs = sorted([d for d in hyperparam_dir.glob('sweep*') if d.is_dir()], 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not sweep_dirs:
            print(f"No hyperparam sweep results found in {hyperparam_dir}!")
            print("\nUsage: python scripts/analysis/analyze_convergence.py [sweep_directory]")
            sys.exit(1)
        
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
        print(f"Sweep directory: {json_path.parent}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (list) and new format (dict with sweep_input)
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
        else:
            results = data
        
        print(f"Loaded {len(results)} configurations")
        
        # Perform analysis
        print_convergence_analysis(results, top_n=args.top)
        compare_population_sizes(results)
    
    # Print completion message to console only (not in file)
    print(f"\n{'=' * 100}")
    print(f"Analysis complete! Full report saved to: {output_path}")
    print(f"{'=' * 100}\n")


if __name__ == '__main__':
    main()
