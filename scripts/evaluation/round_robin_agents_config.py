"""
Enhanced round-robin tournament: sorts by wins, includes config insights, and prints per-agent configuration.
Uses descriptive names based on genome specifications instead of run names.

Usage:
    # Run all checkpoints
    python3 scripts/evaluation/round_robin_agents_config.py
    
    # Run specific checkpoint directories
    python3 scripts/evaluation/round_robin_agents_config.py --checkpoints deep_p12_m6_h375_s0.1_hof3_g50 deep_p20_m6_h500_s0.15_hof3_g50
    
    # Run checkpoints matching a pattern
    python3 scripts/evaluation/round_robin_agents_config.py --pattern "*hof3*"
    
    # Run specific agent files
    python3 scripts/evaluation/round_robin_agents_config.py --agents checkpoints/deep_p12_m6_h375_s0.1_hof3_g50/runs/run_20260128_032747/best_genome.npy
    
    # Customize settings
    python3 scripts/evaluation/round_robin_agents_config.py --hands 5000 --checkpoints deep_p12_* deep_p20_*
"""
import os
import sys
import subprocess
import glob
import re
import json
import argparse
import collections
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

ARCH = "17 64 32 6"
DEFAULT_HANDS = 10000
DEFAULT_PLAYERS = 2
MATCH_SCRIPT = "scripts/evaluation/match_agents.py"

# Generate descriptive name from config
def get_descriptive_name(config, original_name):
    """Generate a short descriptive name from genome config."""
    if config is None:
        return original_name
    
    try:
        pop = config['evolution']['population_size']
        sigma = config['evolution']['mutation_sigma']
        gens = config['num_generations']
        
        # Extract matchups and hands from the correct paths
        matchups = config['fitness']['matchups_per_agent']
        hands = config['fitness']['hands_per_matchup']
        
        # Format: p40_m8_h375_s0.1_g200
        name = f"p{pop}_m{matchups}_h{hands}_s{sigma}_g{gens}"
        return name
    except (KeyError, TypeError) as e:
        # Fallback to original name if config is malformed
        print(f"Warning: Could not parse config for {original_name}: {e}")
        return original_name

# Find all agents and configs
def get_agents_and_configs(checkpoint_dirs=None, checkpoint_pattern=None, agent_paths=None):
    """
    Find agents to include in tournament.
    
    Args:
        checkpoint_dirs: List of checkpoint directory names (e.g., ['deep_p12_m6_h375_s0.1_hof3_g50'])
        checkpoint_pattern: Glob pattern for checkpoint directories (e.g., '*hof3*')
        agent_paths: List of specific agent file paths
    """
    agent_files = []
    
    if agent_paths:
        # Use specific agent files provided
        agent_files = agent_paths
    elif checkpoint_dirs:
        # Use specific checkpoint directories
        for ckpt_dir in checkpoint_dirs:
            # Handle both with and without 'checkpoints/' prefix
            if not ckpt_dir.startswith('checkpoints/'):
                ckpt_dir = f'checkpoints/{ckpt_dir}'
            
            # Find all best_genome.npy files in this checkpoint
            pattern = f"{ckpt_dir}/runs/*/best_genome.npy"
            found = glob.glob(pattern)
            if not found:
                # Try without runs/ subdirectory
                pattern = f"{ckpt_dir}/best_genome.npy"
                found = glob.glob(pattern)
            
            if found:
                agent_files.extend(found)
                print(f"Found {len(found)} agent(s) in {ckpt_dir}")
            else:
                print(f"Warning: No agents found in {ckpt_dir}")
    elif checkpoint_pattern:
        # Use pattern matching for checkpoint directories
        pattern = f"checkpoints/{checkpoint_pattern}/runs/*/best_genome.npy"
        agent_files = glob.glob(pattern)
        if not agent_files:
            # Try without runs/ subdirectory
            pattern = f"checkpoints/{checkpoint_pattern}/best_genome.npy"
            agent_files = glob.glob(pattern)
        print(f"Found {len(agent_files)} agent(s) matching pattern '{checkpoint_pattern}'")
    else:
        # Default: find all agents
        agent_files = glob.glob("checkpoints/*/runs/*/best_genome.npy")
        print(f"Found {len(agent_files)} agent(s) in all checkpoints")
    
    if not agent_files:
        print("ERROR: No agents found!")
        sys.exit(1)
    
    agents = []
    seen_names = {}  # Track duplicate names
    
    for f in agent_files:
        original_name = re.sub(r"checkpoints/(.*)/runs/(.*)/best_genome.npy", r"\1/\2", f)
        # Handle case without runs/ subdirectory
        if 'runs/' not in f:
            original_name = re.sub(r"checkpoints/(.*)/best_genome.npy", r"\1", f)
        
        config_path = os.path.join(os.path.dirname(f), "config.json")
        
        if os.path.exists(config_path):
            with open(config_path) as cf:
                config = json.load(cf)
        else:
            config = None
        
        # Generate descriptive name
        desc_name = get_descriptive_name(config, original_name)
        
        # Handle duplicate names by adding a suffix
        if desc_name in seen_names:
            seen_names[desc_name] += 1
            desc_name = f"{desc_name}_v{seen_names[desc_name]}"
        else:
            seen_names[desc_name] = 1
        
        agents.append({
            'name': desc_name,
            'file': f,
            'config': config,
            'original_name': original_name
        })
    
    return agents

def run_tournament(agents, hands, players):
    """Run the round-robin tournament between all agents."""
    results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'chips': 0, 'matchups': {}})
    
    total_matches = len(agents) * (len(agents) - 1)
    match_count = 0
    
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i == j:
                continue
            
            match_count += 1
            name1 = agent1['name']
            name2 = agent2['name']
            print(f"\n[{match_count}/{total_matches}] === {name1} vs {name2} ===")
            
            cmd = [
                "python3", MATCH_SCRIPT,
                "--agent1", agent1['file'],
                "--arch1", ARCH,
                "--agent2", agent2['file'],
                "--arch2", ARCH,
                "--hands", str(hands),
                "--players", str(players)
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = proc.stdout
            if proc.returncode != 0:
                print(f"Error running match: {name1} vs {name2}")
                print(proc.stderr)
            m = re.search(r"Final stacks: Agent 1: (\d+), Agent 2: (\d+)", output)
            if m:
                chips1 = int(m.group(1))
                chips2 = int(m.group(2))
                results[name1]['chips'] += chips1
                results[name2]['chips'] += chips2
                results[name1]['matchups'][name2] = chips1
                results[name2]['matchups'][name1] = chips2
                if chips1 > chips2:
                    results[name1]['wins'] += 1
                    results[name2]['losses'] += 1
                elif chips2 > chips1:
                    results[name2]['wins'] += 1
                    results[name1]['losses'] += 1
            else:
                print("Could not parse result for", name1, "vs", name2)
    
    return results

def analyze_and_report(agents, results, output_dir=None):
    # Create output directory with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"tournament_reports/tournament_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "report.json")
    
    print("\n=== Round Robin Results (sorted by wins) ===")
    print(f"{'Agent':35}  Wins  Losses  Chips  Pop  Sigma  Gen")
    sorted_agents = sorted(agents, key=lambda a: (-results[a['name']]['wins'], results[a['name']]['losses']))
    
    # For report
    report = {
        'agents': [],
        'matchups': {},
        'insights': {}
    }
    
    # Per-agent summary
    for agent in sorted_agents:
        name = agent['name']
        stats = results[name]
        cfg = agent['config']
        pop = cfg['evolution']['population_size'] if cfg else '-'
        sigma = cfg['evolution']['mutation_sigma'] if cfg else '-'
        ngen = cfg['num_generations'] if cfg else '-'
        print(f"{name:35}  {stats['wins']:4}  {stats['losses']:6}  {stats['chips']:5}  {pop:3}  {sigma:5}  {ngen:3}")
        
        # Who did this agent beat/lose to?
        beat = []
        lost = []
        for opp in sorted_agents:
            if opp['name'] == name:
                continue
            my_score = results[name]['matchups'].get(opp['name'])
            opp_score = results[opp['name']]['matchups'].get(name)
            if my_score is not None and opp_score is not None:
                if my_score > opp_score:
                    beat.append(opp['name'])
                elif my_score < opp_score:
                    lost.append(opp['name'])
        
        report['agents'].append({
            'name': name,
            'original_path': agent['original_name'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'chips': stats['chips'],
            'population_size': pop,
            'mutation_sigma': sigma,
            'generations': ngen,
            'beat': beat,
            'lost_to': lost
        })
        report['matchups'][name] = stats['matchups']
    
    print("\nConfig columns: Pop=population size, Sigma=mutation sigma, Gen=generations trained")
    print("\nPer-matchup chip results:")
    for agent in sorted_agents:
        name = agent['name']
        print(f"{name:35}", end=': ')
        for opp in sorted_agents:
            if opp['name'] == name:
                continue
            val = results[name]['matchups'].get(opp['name'], '-')
            print(f"{val:6}", end=' ')
        print()
    
    # Parameter insights: which config values are most successful?
    pop_counter = collections.Counter()
    sigma_counter = collections.Counter()
    gen_counter = collections.Counter()
    for agent in report['agents']:
        if agent['population_size'] != '-':
            pop_counter[agent['population_size']] += agent['wins']
        if agent['mutation_sigma'] != '-':
            sigma_counter[agent['mutation_sigma']] += agent['wins']
        if agent['generations'] != '-':
            gen_counter[agent['generations']] += agent['wins']
    
    report['insights']['top_population_sizes'] = pop_counter.most_common()
    report['insights']['top_mutation_sigmas'] = sigma_counter.most_common()
    report['insights']['top_generations'] = gen_counter.most_common()
    
    print("\nParameter insights (by total wins):")
    print("Population size:", report['insights']['top_population_sizes'])
    print("Mutation sigma:", report['insights']['top_mutation_sigmas'])
    print("Generations:", report['insights']['top_generations'])
    
    # Write report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {report_path}")
    
    # Print a summary of who each agent beat and lost to
    print("\nAgent win/loss breakdown:")
    for i, agent in enumerate(sorted_agents):
        agent_data = report['agents'][i]
        print(f"{agent['name']}: beat {len(agent_data['beat'])} agents, lost to {len(agent_data['lost_to'])} agents")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_visualizations(sorted_agents, results, report, output_dir)
    print(f"Visualizations saved to {output_dir}")

def create_visualizations(sorted_agents, results, report, output_dir):
    """Create visualization charts for tournament results."""
    
    # 1. Win/Loss Bar Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    agent_names = [a['name'].split('/')[-1] for a in sorted_agents]  # Short names
    wins = [results[a['name']]['wins'] for a in sorted_agents]
    losses = [results[a['name']]['losses'] for a in sorted_agents]
    
    x = np.arange(len(agent_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, wins, width, label='Wins', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, losses, width, label='Losses', color='red', alpha=0.7)
    
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Round Robin Tournament Results: Wins vs Losses', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wins_losses.png'), dpi=150)
    plt.close()
    
    # 2. Parameter Performance Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Population Size
    pop_data = report['insights']['top_population_sizes']
    if pop_data:
        pops, pop_wins = zip(*pop_data)
        axes[0].bar([str(p) for p in pops], pop_wins, color='blue', alpha=0.7)
        axes[0].set_xlabel('Population Size', fontsize=12)
        axes[0].set_ylabel('Total Wins', fontsize=12)
        axes[0].set_title('Wins by Population Size', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
    
    # Mutation Sigma
    sigma_data = report['insights']['top_mutation_sigmas']
    if sigma_data:
        sigmas, sigma_wins = zip(*sigma_data)
        axes[1].bar([str(s) for s in sigmas], sigma_wins, color='orange', alpha=0.7)
        axes[1].set_xlabel('Mutation Sigma', fontsize=12)
        axes[1].set_ylabel('Total Wins', fontsize=12)
        axes[1].set_title('Wins by Mutation Sigma', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
    
    # Generations
    gen_data = report['insights']['top_generations']
    if gen_data:
        gens, gen_wins = zip(*gen_data)
        axes[2].bar([str(g) for g in gens], gen_wins, color='purple', alpha=0.7)
        axes[2].set_xlabel('Generations Trained', fontsize=12)
        axes[2].set_ylabel('Total Wins', fontsize=12)
        axes[2].set_title('Wins by Training Generations', fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_performance.png'), dpi=150)
    plt.close()
    
    # 3. Head-to-Head Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    n_agents = len(sorted_agents)
    heatmap_data = np.zeros((n_agents, n_agents))
    
    for i, agent1 in enumerate(sorted_agents):
        for j, agent2 in enumerate(sorted_agents):
            if i != j:
                chips = results[agent1['name']]['matchups'].get(agent2['name'], 0)
                heatmap_data[i, j] = chips
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_agents))
    ax.set_yticks(np.arange(n_agents))
    ax.set_xticklabels(agent_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(agent_names, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Chips Won', rotation=270, labelpad=20)
    
    ax.set_title('Head-to-Head Results (Chips Won)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Opponent', fontsize=12)
    ax.set_ylabel('Agent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'head_to_head_heatmap.png'), dpi=150)
    plt.close()
    
    # 4. Chip Distribution
    fig, ax = plt.subplots(figsize=(14, 8))
    chips = [results[a['name']]['chips'] for a in sorted_agents]
    
    bars = ax.bar(agent_names, chips, color='teal', alpha=0.7)
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Total Chips', fontsize=12)
    ax.set_title('Total Chips Accumulated', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chip_distribution.png'), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Run round-robin tournament between poker agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all checkpoints
  python3 scripts/evaluation/round_robin_agents_config.py
  
  # Run specific checkpoints
  python3 scripts/evaluation/round_robin_agents_config.py --checkpoints deep_p12_m6_h375_s0.1_hof3_g50 deep_p20_m6_h500_s0.15_hof3_g50
  
  # Run checkpoints matching pattern
  python3 scripts/evaluation/round_robin_agents_config.py --pattern "*hof3_g50"
  
  # Run specific agent files
  python3 scripts/evaluation/round_robin_agents_config.py --agents checkpoints/deep_p12_*/runs/*/best_genome.npy
  
  # Customize settings
  python3 scripts/evaluation/round_robin_agents_config.py --hands 5000 --checkpoints deep_p12_* deep_p20_*
        """
    )
    
    # Selection arguments
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--checkpoints', nargs='+', metavar='DIR',
                          help='Specific checkpoint directory names (e.g., deep_p12_m6_h375_s0.1_hof3_g50)')
    selection.add_argument('--pattern', type=str, metavar='PATTERN',
                          help='Glob pattern for checkpoint directories (e.g., "*hof3*")')
    selection.add_argument('--agents', nargs='+', metavar='PATH',
                          help='Specific agent file paths')
    
    # Tournament settings
    parser.add_argument('--hands', type=int, default=DEFAULT_HANDS,
                       help=f'Hands per matchup (default: {DEFAULT_HANDS})')
    parser.add_argument('--players', type=int, default=DEFAULT_PLAYERS,
                       help=f'Players per table (default: {DEFAULT_PLAYERS})')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: tournament_reports/tournament_<timestamp>)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("POKER AI ROUND-ROBIN TOURNAMENT")
    print("=" * 80)
    print(f"Hands per matchup: {args.hands}")
    print(f"Players per table: {args.players}")
    print()
    
    # Get agents based on selection criteria
    agents = get_agents_and_configs(
        checkpoint_dirs=args.checkpoints,
        checkpoint_pattern=args.pattern,
        agent_paths=args.agents
    )
    
    print(f"\nTournament participants: {len(agents)} agents")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent['name']}")
    print()
    
    if len(agents) < 2:
        print("ERROR: Need at least 2 agents for a tournament!")
        sys.exit(1)
    
    # Run tournament
    results = run_tournament(agents, args.hands, args.players)
    
    # Analyze and report
    analyze_and_report(agents, results, output_dir=args.output)
    
    print("\n" + "=" * 80)
    print("Tournament complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()

