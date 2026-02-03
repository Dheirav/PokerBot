#!/usr/bin/env python3
"""
Visualize hyperparameter sweep results from JSON data.

Usage:
    python visualize_hyperparam_sweep.py                                    # Analyze latest sweep
    python visualize_hyperparam_sweep.py hyperparam_results/sweep_XXX       # Analyze specific directory
    python visualize_hyperparam_sweep.py hyperparam_results/sweep_XXX/results.json  # Analyze specific file
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import argparse

def load_results(json_path):
    """Load results from JSON file. Handles both old (list) and new (dict with sweep_input) formats."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both old format (list) and new format (dict with sweep_input)
    if isinstance(data, dict) and 'results' in data:
        return data['results'], data.get('sweep_input')
    else:
        return data, None

def plot_final_metrics_comparison(results, output_dir):
    """Compare final metrics across all configurations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Final Metrics Comparison Across Hyperparameter Configurations', fontsize=16, fontweight='bold')
    
    names = [r['name'] for r in results]
    metrics = {
        'Final Best Fitness': [r['final_best_fitness'] for r in results],
        'Final Mean Fitness': [r.get('final_mean_fitness', r.get('final_train_fitness', 0)) for r in results],
        'Overfitting Gap': [r.get('overfitting_gap', r['final_best_fitness'] - r.get('final_mean_fitness', r.get('final_train_fitness', 0))) for r in results],
        'Convergence': [r['convergence'] for r in results],
        'Efficiency': [r['efficiency'] for r in results],
        'Avg Gen Time (s)': [r['avg_gen_time'] for r in results],
    }
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(range(len(names)), values, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight best value
        best_idx = np.argmax(values) if 'Gap' not in metric_name and 'Time' not in metric_name else np.argmin(values)
        bars[best_idx].set_color('red')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'final_metrics_comparison.png'}")
    plt.close()

def plot_fitness_progression(results, output_dir):
    """Plot fitness progression over generations for all configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fitness Progression Over Generations', fontsize=16, fontweight='bold')
    
    # Best progress
    ax = axes[0, 0]
    for r in results:
        if r.get('best_progress') and len(r['best_progress']) > 0:
            ax.plot(r['best_progress'], label=r['name'], marker='o', markersize=3, alpha=0.7)
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Best Fitness', fontweight='bold')
    ax.set_title('Best Fitness Progress', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Train fitness
    ax = axes[0, 1]
    for r in results:
        if r.get('train_fitness') and len(r['train_fitness']) > 0:
            ax.plot(r['train_fitness'], label=r['name'], marker='o', markersize=3, alpha=0.7)
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Train Fitness', fontweight='bold')
    ax.set_title('Training Fitness (Current Gen)', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Generation times
    ax = axes[1, 0]
    for r in results:
        if r.get('gen_times') and len(r['gen_times']) > 0:
            ax.plot(r['gen_times'], label=r['name'], marker='o', markersize=3, alpha=0.7)
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Generation Training Time', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Convergence speed (normalized best progress)
    ax = axes[1, 1]
    for r in results:
        if r.get('best_progress') and len(r['best_progress']) > 0:
            best_progress = np.array(r['best_progress'])
            if best_progress[-1] > 0:
                normalized = best_progress / best_progress[-1]
            else:
                normalized = best_progress
            ax.plot(normalized, label=r['name'], marker='o', markersize=3, alpha=0.7)
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Normalized Best Fitness', fontweight='bold')
    ax.set_title('Convergence Speed (Normalized)', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fitness_progression.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fitness_progression.png'}")
    plt.close()

def plot_hyperparameter_heatmaps(results, output_dir):
    """Create heatmaps showing parameter effects on performance."""
    # Extract unique parameter values
    params = {
        'population_size': sorted(set(r['config']['population_size'] for r in results)),
        'mutation_sigma': sorted(set(r['config']['mutation_sigma'] for r in results)),
        'matchups_per_agent': sorted(set(r['config']['matchups_per_agent'] for r in results)),
        'hands_per_matchup': sorted(set(r['config']['hands_per_matchup'] for r in results)),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Effects on Final Best Fitness', fontsize=16, fontweight='bold')
    
    # Population size vs Mutation sigma
    ax = axes[0, 0]
    data = np.full((len(params['mutation_sigma']), len(params['population_size'])), np.nan)
    for r in results:
        i = params['mutation_sigma'].index(r['config']['mutation_sigma'])
        j = params['population_size'].index(r['config']['population_size'])
        data[i, j] = r['final_best_fitness']
    
    im = ax.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(params['population_size'])))
    ax.set_yticks(range(len(params['mutation_sigma'])))
    ax.set_xticklabels(params['population_size'])
    ax.set_yticklabels(params['mutation_sigma'])
    ax.set_xlabel('Population Size', fontweight='bold')
    ax.set_ylabel('Mutation Sigma', fontweight='bold')
    ax.set_title('Population Size vs Mutation Sigma', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Final Best Fitness')
    
    # Add text annotations
    for i in range(len(params['mutation_sigma'])):
        for j in range(len(params['population_size'])):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.0f}',
                             ha="center", va="center", color="white", fontsize=9, fontweight='bold')
    
    # Matchups vs Hands
    ax = axes[0, 1]
    data = np.full((len(params['hands_per_matchup']), len(params['matchups_per_agent'])), np.nan)
    for r in results:
        i = params['hands_per_matchup'].index(r['config']['hands_per_matchup'])
        j = params['matchups_per_agent'].index(r['config']['matchups_per_agent'])
        data[i, j] = r['final_best_fitness']
    
    im = ax.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(params['matchups_per_agent'])))
    ax.set_yticks(range(len(params['hands_per_matchup'])))
    ax.set_xticklabels(params['matchups_per_agent'])
    ax.set_yticklabels(params['hands_per_matchup'])
    ax.set_xlabel('Matchups Per Agent', fontweight='bold')
    ax.set_ylabel('Hands Per Matchup', fontweight='bold')
    ax.set_title('Matchups vs Hands Per Matchup', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Final Best Fitness')
    
    for i in range(len(params['hands_per_matchup'])):
        for j in range(len(params['matchups_per_agent'])):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.0f}',
                             ha="center", va="center", color="white", fontsize=9, fontweight='bold')
    
    # Efficiency vs Performance
    ax = axes[1, 0]
    efficiencies = [r['efficiency'] for r in results]
    fitnesses = [r['final_best_fitness'] for r in results]
    names = [r['name'] for r in results]
    
    scatter = ax.scatter(efficiencies, fitnesses, c=range(len(results)), 
                        cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    for i, name in enumerate(names):
        ax.annotate(name, (efficiencies[i], fitnesses[i]), 
                   fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Efficiency (Fitness/Time)', fontweight='bold')
    ax.set_ylabel('Final Best Fitness', fontweight='bold')
    ax.set_title('Efficiency vs Performance Trade-off', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Overfitting analysis
    ax = axes[1, 1]
    train_fit = [r.get('final_mean_fitness', r.get('final_train_fitness', 0)) for r in results]
    best_fit = [r['final_best_fitness'] for r in results]
    
    scatter = ax.scatter(train_fit, best_fit, c=range(len(results)), 
                        cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    for i, name in enumerate(names):
        ax.annotate(name, (train_fit[i], best_fit[i]), 
                   fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    ax.plot([min(train_fit + best_fit), max(train_fit + best_fit)], 
           [min(train_fit + best_fit), max(train_fit + best_fit)], 
           'r--', alpha=0.5, label='Perfect consistency')
    ax.set_xlabel('Final Train Fitness', fontweight='bold')
    ax.set_ylabel('Final Best Fitness', fontweight='bold')
    ax.set_title('Training Consistency Analysis', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'hyperparameter_heatmaps.png'}")
    plt.close()

def plot_top_configurations(results, output_dir, top_n=5):
    """Detailed plots for top N configurations."""
    # Sort by final best fitness
    sorted_results = sorted(results, key=lambda x: x['final_best_fitness'], reverse=True)[:top_n]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_n} Configurations - Detailed Analysis', fontsize=16, fontweight='bold')
    
    # Best progress
    ax = axes[0, 0]
    for r in sorted_results:
        ax.plot(r['best_progress'], label=f"{r['name']} ({r['final_best_fitness']:.1f})", 
               marker='o', markersize=4, linewidth=2)
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Best Fitness', fontweight='bold')
    ax.set_title('Best Fitness Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Train fitness variance
    ax = axes[0, 1]
    for r in sorted_results:
        ax.plot(r['train_fitness'], label=r['name'], marker='o', markersize=4, linewidth=2, alpha=0.7)
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Train Fitness', fontweight='bold')
    ax.set_title('Training Fitness Stability', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Generation time stability
    ax = axes[1, 0]
    for r in sorted_results:
        ax.plot(r['gen_times'], label=r['name'], marker='o', markersize=4, linewidth=2, alpha=0.7)
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Generation Time Consistency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Configuration comparison table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Rank', 'Config', 'Fitness', 'Efficiency', 'Avg Time (s)']
    
    for i, r in enumerate(sorted_results, 1):
        table_data.append([
            str(i),
            r['name'],
            f"{r['final_best_fitness']:.1f}",
            f"{r['efficiency']:.4f}",
            f"{r['avg_gen_time']:.1f}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.1, 0.35, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Top Configurations Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_configurations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'top_configurations.png'}")
    plt.close()

def generate_summary_report(results, output_dir):
    """Generate a text summary report."""
    sorted_results = sorted(results, key=lambda x: x['final_best_fitness'], reverse=True)
    
    report = []
    report.append("=" * 80)
    report.append("HYPERPARAMETER SWEEP ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal Configurations Tested: {len(results)}")
    report.append(f"\nGenerations per Config: {results[0]['config']['generations']}")
    
    report.append("\n" + "=" * 80)
    report.append("TOP 5 CONFIGURATIONS BY FINAL BEST FITNESS")
    report.append("=" * 80)
    
    for i, r in enumerate(sorted_results[:5], 1):
        report.append(f"\n{i}. {r['name']}")
        report.append(f"   Configuration:")
        report.append(f"     - Population Size: {r['config']['population_size']}")
        report.append(f"     - Matchups Per Agent: {r['config']['matchups_per_agent']}")
        report.append(f"     - Hands Per Matchup: {r['config']['hands_per_matchup']}")
        report.append(f"     - Mutation Sigma: {r['config']['mutation_sigma']}")
        if 'hof_opponent_count' in r:
            report.append(f"     - HoF Opponents: {r['hof_opponent_count']}")
        report.append(f"   Results:")
        report.append(f"     - Final Best Fitness: {r['final_best_fitness']:.2f}")
        mean_fitness = r.get('final_mean_fitness', r.get('final_train_fitness', 0))
        report.append(f"     - Final Mean Fitness: {mean_fitness:.2f}")
        overfitting_gap = r.get('overfitting_gap', r['final_best_fitness'] - mean_fitness)
        report.append(f"     - Overfitting Gap: {overfitting_gap:.2f}")
        report.append(f"     - Convergence: {r['convergence']:.2f}")
        report.append(f"     - Efficiency: {r['efficiency']:.4f}")
        report.append(f"     - Avg Gen Time: {r['avg_gen_time']:.2f}s")
        report.append(f"     - Total Hands/Gen: {r['total_hands_per_gen']}")
    
    report.append("\n" + "=" * 80)
    report.append("KEY INSIGHTS")
    report.append("=" * 80)
    
    best_config = sorted_results[0]
    report.append(f"\nBest Overall: {best_config['name']}")
    report.append(f"  Fitness: {best_config['final_best_fitness']:.2f}")
    
    most_efficient = max(results, key=lambda x: x['efficiency'])
    report.append(f"\nMost Efficient: {most_efficient['name']}")
    report.append(f"  Efficiency: {most_efficient['efficiency']:.4f}")
    report.append(f"  Fitness: {most_efficient['final_best_fitness']:.2f}")
    
    fastest = min(results, key=lambda x: x['avg_gen_time'])
    report.append(f"\nFastest Training: {fastest['name']}")
    report.append(f"  Avg Gen Time: {fastest['avg_gen_time']:.2f}s")
    report.append(f"  Fitness: {fastest['final_best_fitness']:.2f}")
    
    least_overfitting = min(results, key=lambda x: x.get('overfitting_gap', x['final_best_fitness'] - x.get('final_mean_fitness', x.get('final_train_fitness', 0))))
    report.append(f"\nLeast Overfitting: {least_overfitting['name']}")
    overfitting_gap = least_overfitting.get('overfitting_gap', least_overfitting['final_best_fitness'] - least_overfitting.get('final_mean_fitness', least_overfitting.get('final_train_fitness', 0)))
    report.append(f"  Overfitting Gap: {overfitting_gap:.2f}")
    report.append(f"  Fitness: {least_overfitting['final_best_fitness']:.2f}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    
    # Print to console
    print("\n" + report_text)
    
    # Save to file
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\nSaved: {output_dir / 'analysis_report.txt'}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize hyperparameter sweep results with plots and analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_hyperparam_sweep.py
  python visualize_hyperparam_sweep.py hyperparam_results/sweep_20260127_133129
  python visualize_hyperparam_sweep.py hyperparam_results/sweep_20260127_133129/results.json
        """
    )
    parser.add_argument('sweep_path', nargs='?', type=str,
                       help='Path to sweep directory or results.json file (default: latest sweep)')
    
    args = parser.parse_args()
    
    # Determine project root and hyperparam directory
    project_root = Path(__file__).parent.parent.parent
    hyperparam_dir = project_root / 'hyperparam_results'
    
    # Resolve path to results.json
    if args.sweep_path:
        sweep_path = Path(args.sweep_path)
        if not sweep_path.is_absolute():
            sweep_path = project_root / sweep_path
        
        if sweep_path.is_dir():
            json_path = sweep_path / 'results.json'
            output_dir = sweep_path / 'visualizations'
        elif sweep_path.name == 'results.json':
            json_path = sweep_path
            output_dir = sweep_path.parent / 'visualizations'
        else:
            print(f"Error: Invalid path - {sweep_path}")
            print("Please provide a directory containing results.json or the results.json file itself.")
            return
    else:
        # Auto-detect latest sweep
        sweep_dirs = sorted([d for d in hyperparam_dir.glob('sweep*') if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not sweep_dirs:
            print("No hyperparam sweep results found!")
            return
        
        latest_sweep = sweep_dirs[0]
        json_path = latest_sweep / 'results.json'
        output_dir = latest_sweep / 'visualizations'
    
    if not json_path.exists():
        print(f"No results.json found at: {json_path}")
        return
    
    print(f"Loading results from: {json_path}")
    results, sweep_input = load_results(json_path)
    print(f"Loaded {len(results)} configurations")
    
    # Create output directory for plots
    output_dir.mkdir(exist_ok=True)
    print(f"\nGenerating visualizations in: {output_dir}\n")
    
    # Generate all visualizations
    print("Generating plots...")
    plot_final_metrics_comparison(results, output_dir)
    plot_fitness_progression(results, output_dir)
    plot_hyperparameter_heatmaps(results, output_dir)
    plot_top_configurations(results, output_dir, top_n=5)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results, output_dir)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - final_metrics_comparison.png")
    print("  - fitness_progression.png")
    print("  - hyperparameter_heatmaps.png")
    print("  - top_configurations.png")
    print("  - analysis_report.txt")

if __name__ == '__main__':
    main()
