#!/usr/bin/env python3
"""
Hall of Fame Member Analysis

Analyzes the actual Hall of Fame members stored in checkpoint files,
including their fitness evolution and neural network characteristics.

Usage: python scripts/analysis/analyze_hof_members.py
"""

import json
import numpy as np
from pathlib import Path
import statistics
from collections import defaultdict

class HallOfFameAnalyzer:
    def __init__(self, checkpoints_dir="checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.hof_runs = []
        self.analysis_data = defaultdict(list)
        
    def find_hof_runs(self):
        """Find all runs with Hall of Fame data."""
        print("🔍 Searching for Hall of Fame runs...")
        
        for checkpoint_dir in self.checkpoints_dir.iterdir():
            if not checkpoint_dir.is_dir() or "hof" not in checkpoint_dir.name:
                continue
                
            runs_dir = checkpoint_dir / "runs"
            if not runs_dir.exists():
                continue
                
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                    
                hof_file = run_dir / "hall_of_fame.npy"
                state_file = run_dir / "state.json"
                history_file = run_dir / "history.json"
                
                if hof_file.exists() and state_file.exists() and history_file.exists():
                    self.hof_runs.append({
                        'run_path': run_dir,
                        'config_name': checkpoint_dir.name,
                        'run_name': run_dir.name
                    })
        
        print(f"✓ Found {len(self.hof_runs)} Hall of Fame runs")
        return self.hof_runs
    
    def analyze_hof_run(self, run_info):
        """Analyze a single Hall of Fame run."""
        run_path = run_info['run_path']
        
        try:
            # Load HOF members
            hof_data = np.load(run_path / "hall_of_fame.npy", allow_pickle=True)
            
            # Load state
            with open(run_path / "state.json", 'r') as f:
                state = json.load(f)
                
            # Load history
            with open(run_path / "history.json", 'r') as f:
                history = json.load(f)
            
            return {
                'config_name': run_info['config_name'],
                'run_name': run_info['run_name'],
                'hof_members': hof_data,
                'final_hof_size': len(hof_data),
                'max_hof_size': state.get('hof_size', len(hof_data)),
                'final_best_fitness': state.get('best_fitness', 0),
                'best_genome_id': state.get('best_genome_id', 'unknown'),
                'history': history,
                'network_size': hof_data[0].shape[0] if len(hof_data) > 0 else 0
            }
            
        except Exception as e:
            print(f"⚠️  Error analyzing {run_info['config_name']}: {e}")
            return None
    
    def analyze_hof_evolution(self, run_data):
        """Analyze how the Hall of Fame evolved over generations."""
        history = run_data['history']
        
        # Track HOF growth
        hof_growth = []
        best_fitness_progression = []
        
        for entry in history:
            gen = entry['generation']
            hof_size = entry.get('hof_size', 0)
            best_ever = entry.get('best_ever_fitness', 0)
            
            hof_growth.append(hof_size)
            best_fitness_progression.append(best_ever)
        
        # Find when HOF reached capacity
        max_hof_size = max(hof_growth) if hof_growth else 0
        capacity_gen = next((i for i, size in enumerate(hof_growth) if size >= max_hof_size), len(hof_growth))
        
        # Find major fitness improvements
        improvements = []
        for i in range(1, len(best_fitness_progression)):
            current = best_fitness_progression[i]
            previous = best_fitness_progression[i-1]
            if current > previous * 1.1:  # 10% improvement
                improvements.append({
                    'generation': i,
                    'fitness': current,
                    'improvement': (current - previous) / previous * 100
                })
        
        return {
            'hof_growth': hof_growth,
            'fitness_progression': best_fitness_progression,
            'capacity_reached_gen': capacity_gen,
            'major_improvements': improvements,
            'final_best': best_fitness_progression[-1] if best_fitness_progression else 0
        }
    
    def analyze_network_characteristics(self, run_data):
        """Analyze neural network characteristics of HOF members."""
        hof_members = run_data['hof_members']
        if len(hof_members) == 0:
            return None
            
        # Calculate statistics on network weights
        all_weights = np.array(hof_members)  # Shape: (num_members, network_size)
        
        analysis = {
            'num_members': len(hof_members),
            'network_size': hof_members[0].shape[0],
            'weight_stats': {
                'mean': float(np.mean(all_weights)),
                'std': float(np.std(all_weights)),
                'min': float(np.min(all_weights)),
                'max': float(np.max(all_weights))
            },
            'diversity_stats': {
                'pairwise_distances': [],
                'mean_diversity': 0,
                'min_diversity': 0,
                'max_diversity': 0
            }
        }
        
        # Calculate pairwise diversity (L2 distances)
        distances = []
        for i in range(len(hof_members)):
            for j in range(i+1, len(hof_members)):
                dist = float(np.linalg.norm(hof_members[i] - hof_members[j]))
                distances.append(dist)
        
        if distances:
            analysis['diversity_stats']['pairwise_distances'] = distances
            analysis['diversity_stats']['mean_diversity'] = statistics.mean(distances)
            analysis['diversity_stats']['min_diversity'] = min(distances)
            analysis['diversity_stats']['max_diversity'] = max(distances)
        
        return analysis
    
    def generate_hof_report(self):
        """Generate comprehensive Hall of Fame analysis report."""
        print("\n" + "=" * 80)
        print("HALL OF FAME MEMBER ANALYSIS")
        print("=" * 80)
        
        all_run_data = []
        for run_info in self.hof_runs:
            run_data = self.analyze_hof_run(run_info)
            if run_data:
                all_run_data.append(run_data)
        
        if not all_run_data:
            print("❌ No valid Hall of Fame data found")
            return
        
        print(f"\n📊 Analyzed {len(all_run_data)} Hall of Fame runs")
        
        # Overall statistics
        total_hof_members = sum(r['final_hof_size'] for r in all_run_data)
        avg_hof_size = statistics.mean([r['final_hof_size'] for r in all_run_data])
        max_fitness = max(r['final_best_fitness'] for r in all_run_data)
        avg_fitness = statistics.mean([r['final_best_fitness'] for r in all_run_data])
        
        print(f"\n🏆 Overall Statistics:")
        print(f"  Total HOF members across all runs: {total_hof_members}")
        print(f"  Average HOF size per run: {avg_hof_size:.1f}")
        print(f"  Highest fitness achieved: {max_fitness:.2f}")
        print(f"  Average final best fitness: {avg_fitness:.2f}")
        
        # Network architecture analysis
        network_sizes = [r['network_size'] for r in all_run_data if r['network_size'] > 0]
        if network_sizes:
            unique_sizes = set(network_sizes)
            print(f"\n🧠 Neural Network Architecture:")
            print(f"  Network sizes found: {sorted(unique_sizes)}")
            print(f"  Most common size: {statistics.mode(network_sizes)} parameters")
        
        # Configuration-specific analysis
        print(f"\n📈 Configuration Performance:")
        config_performance = defaultdict(list)
        
        for run_data in all_run_data:
            config_name = run_data['config_name']
            fitness = run_data['final_best_fitness']
            config_performance[config_name].append(fitness)
        
        for config, fitnesses in sorted(config_performance.items()):
            mean_fitness = statistics.mean(fitnesses)
            max_fitness = max(fitnesses)
            num_runs = len(fitnesses)
            print(f"  {config}:")
            print(f"    Runs: {num_runs}, Mean: {mean_fitness:.2f}, Max: {max_fitness:.2f}")
        
        # HOF evolution analysis
        print(f"\n🔄 Hall of Fame Evolution:")
        
        for run_data in all_run_data[:3]:  # Show first 3 runs as examples
            evolution = self.analyze_hof_evolution(run_data)
            config = run_data['config_name']
            
            print(f"\n  📊 {config}:")
            print(f"    Final HOF size: {run_data['final_hof_size']}")
            print(f"    HOF capacity reached at generation: {evolution['capacity_reached_gen']}")
            print(f"    Final best fitness: {evolution['final_best']:.2f}")
            
            if evolution['major_improvements']:
                print(f"    Major fitness breakthroughs:")
                for imp in evolution['major_improvements'][:3]:  # Show first 3
                    print(f"      Gen {imp['generation']:2d}: {imp['fitness']:7.2f} (+{imp['improvement']:5.1f}%)")
        
        # Network diversity analysis  
        print(f"\n🎯 Neural Network Diversity Analysis:")
        
        for run_data in all_run_data[:2]:  # Show first 2 as examples
            net_analysis = self.analyze_network_characteristics(run_data)
            if net_analysis:
                config = run_data['config_name']
                print(f"\n  🧠 {config}:")
                print(f"    HOF members: {net_analysis['num_members']}")
                print(f"    Network size: {net_analysis['network_size']:,} parameters")
                
                weights = net_analysis['weight_stats']
                print(f"    Weight statistics:")
                print(f"      Range: [{weights['min']:.3f}, {weights['max']:.3f}]")
                print(f"      Mean ± Std: {weights['mean']:.3f} ± {weights['std']:.3f}")
                
                diversity = net_analysis['diversity_stats']
                if diversity['mean_diversity'] > 0:
                    print(f"    Member diversity (L2 distances):")
                    print(f"      Mean: {diversity['mean_diversity']:.1f}")
                    print(f"      Range: [{diversity['min_diversity']:.1f}, {diversity['max_diversity']:.1f}]")
        
        # Save detailed report
        self.save_detailed_report(all_run_data)
        
    def save_detailed_report(self, all_run_data):
        """Save detailed HOF analysis to file."""
        output_file = Path("hyperparam_results/analysis/hall_of_fame_members_analysis.txt")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("HALL OF FAME MEMBERS DETAILED ANALYSIS\\n")
            f.write("=" * 80 + "\\n\\n")
            
            for run_data in all_run_data:
                f.write(f"Configuration: {run_data['config_name']}\\n")
                f.write(f"Run: {run_data['run_name']}\\n")
                f.write(f"HOF Members: {run_data['final_hof_size']}\\n")
                f.write(f"Final Best Fitness: {run_data['final_best_fitness']:.2f}\\n")
                f.write(f"Network Size: {run_data['network_size']:,} parameters\\n")
                
                # Evolution data
                evolution = self.analyze_hof_evolution(run_data)
                f.write(f"HOF Capacity Reached: Generation {evolution['capacity_reached_gen']}\\n")
                
                if evolution['major_improvements']:
                    f.write("Major Fitness Breakthroughs:\\n")
                    for imp in evolution['major_improvements']:
                        f.write(f"  Gen {imp['generation']:2d}: {imp['fitness']:7.2f} (+{imp['improvement']:5.1f}%)\\n")
                
                # Network analysis
                net_analysis = self.analyze_network_characteristics(run_data)
                if net_analysis:
                    weights = net_analysis['weight_stats']
                    f.write(f"Weight Range: [{weights['min']:.3f}, {weights['max']:.3f}]\\n")
                    f.write(f"Weight Mean ± Std: {weights['mean']:.3f} ± {weights['std']:.3f}\\n")
                    
                    diversity = net_analysis['diversity_stats']
                    if diversity['mean_diversity'] > 0:
                        f.write(f"Member Diversity: {diversity['mean_diversity']:.1f} (L2 distance)\\n")
                
                f.write("\\n" + "-" * 40 + "\\n\\n")
        
        print(f"\\n📄 Detailed report saved to: {output_file}")

def main():
    analyzer = HallOfFameAnalyzer()
    hof_runs = analyzer.find_hof_runs()
    
    if not hof_runs:
        print("❌ No Hall of Fame runs found in checkpoints directory")
        return 1
    
    analyzer.generate_hof_report()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())