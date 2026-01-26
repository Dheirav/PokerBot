#!/usr/bin/env python3
"""
Tournament History Analyzer

Analyzes cumulative tournament results across multiple tournament runs to identify:
- Best performing agents consistently
- Hyperparameter patterns that correlate with success
- Development recommendations for future training
- Head-to-head matchup statistics between specific agents
- Visual analysis with charts and heatmaps

Usage:
    python scripts/analyze_tournament_history.py [--min-tournaments N]

Options:
    --min-tournaments N    Only include agents that participated in N+ tournaments (default: 1)
    --top-n N             Show top N agents in rankings (default: 10)
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import re
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


class AgentStats:
    """Tracks cumulative statistics for a single agent across tournaments."""
    
    def __init__(self, name: str):
        self.name = name
        self.total_wins = 0
        self.total_losses = 0
        self.total_tournaments = 0
        self.total_chip_earnings = 0
        self.chip_counts = []  # Track per-tournament chip counts
        self.tournament_dates = []
        self.opponents = defaultdict(lambda: {'wins': 0, 'losses': 0})  # Head-to-head stats
        
    @property
    def total_games(self) -> int:
        return self.total_wins + self.total_losses
    
    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games
    
    @property
    def avg_chips_per_tournament(self) -> float:
        if self.total_tournaments == 0:
            return 0.0
        return self.total_chip_earnings / self.total_tournaments
    
    @property
    def chip_consistency(self) -> float:
        """Lower is better - measures variance in chip performance."""
        if len(self.chip_counts) < 2:
            return 0.0
        mean = sum(self.chip_counts) / len(self.chip_counts)
        variance = sum((x - mean) ** 2 for x in self.chip_counts) / len(self.chip_counts)
        return variance ** 0.5
    
    def add_tournament_result(self, wins: int, losses: int, final_chips: int, date: str):
        """Add results from a single tournament."""
        self.total_wins += wins
        self.total_losses += losses
        self.total_tournaments += 1
        self.total_chip_earnings += final_chips
        self.chip_counts.append(final_chips)
        self.tournament_dates.append(date)
    
    def add_head_to_head(self, opponent: str, won: bool):
        """Record a head-to-head matchup result."""
        if won:
            self.opponents[opponent]['wins'] += 1
        else:
            self.opponents[opponent]['losses'] += 1


def parse_genome_spec(name: str) -> Dict[str, float]:
    """
    Parse genome specification from agent name.
    
    Expected format: p{pop}_m{matchups}_h{hands}_s{sigma}_g{gens}
    
    Returns:
        Dictionary with keys: population, matchups, hands, sigma, generations
    """
    pattern = r'p(\d+)_m(\d+)_h(\d+)_s([\d.]+)(?:_g(\d+))?'
    match = re.search(pattern, name)
    
    if not match:
        return {}
    
    return {
        'population': int(match.group(1)),
        'matchups': int(match.group(2)),
        'hands': int(match.group(3)),
        'sigma': float(match.group(4)),
        'generations': int(match.group(5)) if match.group(5) else None
    }


def find_tournament_reports(base_dir: str = 'tournament_reports') -> List[Tuple[str, Path]]:
    """
    Find all tournament JSON reports.
    
    Returns:
        List of (timestamp, path) tuples sorted by timestamp
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    reports = []
    for tournament_dir in base_path.iterdir():
        if not tournament_dir.is_dir():
            continue
        
        # Extract timestamp from directory name (tournament_YYYYMMDD_HHMMSS)
        match = re.search(r'tournament_(\d{8}_\d{6})', tournament_dir.name)
        if not match:
            continue
        
        timestamp = match.group(1)
        
        # Try both possible report filenames
        report_path = tournament_dir / 'report.json'
        if not report_path.exists():
            report_path = tournament_dir / 'round_robin_report.json'
        
        if report_path.exists():
            reports.append((timestamp, report_path))
    
    return sorted(reports)  # Sort by timestamp


def load_tournament_data(report_path: Path) -> Dict:
    """Load tournament data from JSON report."""
    with open(report_path, 'r') as f:
        return json.load(f)


def analyze_tournament_history(min_tournaments: int = 1) -> Tuple[Dict[str, AgentStats], List[Dict]]:
    """
    Analyze all tournament results and aggregate statistics.
    
    Args:
        min_tournaments: Minimum number of tournaments an agent must participate in
        
    Returns:
        Tuple of (agent_stats dictionary, list of match data dictionaries)
    """
    reports = find_tournament_reports()
    
    if not reports:
        print("No tournament reports found in tournament_reports/")
        return {}, []
    
    print(f"Found {len(reports)} tournament(s) to analyze\n")
    
    agent_stats = {}
    all_matches = []  # Store all individual match results
    
    for timestamp, report_path in reports:
        print(f"Processing tournament from {timestamp}...")
        data = load_tournament_data(report_path)
        
        # Extract results for each agent (handle both list and dict formats)
        agents_data = data.get('agents', [])
        if isinstance(agents_data, dict):
            # Old format: dictionary with agent names as keys
            for agent_name, stats in agents_data.items():
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = AgentStats(agent_name)
                
                agent_stats[agent_name].add_tournament_result(
                    wins=stats.get('wins', 0),
                    losses=stats.get('losses', 0),
                    final_chips=stats.get('final_chips', stats.get('chips', 0)),
                    date=timestamp
                )
        elif isinstance(agents_data, list):
            # New format: list of agent objects
            for agent in agents_data:
                agent_name = agent.get('name', 'Unknown')
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = AgentStats(agent_name)
                
                agent_stats[agent_name].add_tournament_result(
                    wins=agent.get('wins', 0),
                    losses=agent.get('losses', 0),
                    final_chips=agent.get('chips', 0),
                    date=timestamp
                )
                
                # Extract head-to-head from beat/lost_to lists
                for opponent in agent.get('beat', []):
                    if opponent in agent_stats or opponent != agent_name:
                        if opponent not in agent_stats:
                            agent_stats[opponent] = AgentStats(opponent)
                        agent_stats[agent_name].add_head_to_head(opponent, won=True)
                        
                        # Record in all_matches
                        all_matches.append({
                            'winner': agent_name,
                            'loser': opponent,
                            'tournament': timestamp
                        })
                
                for opponent in agent.get('lost_to', []):
                    if opponent in agent_stats or opponent != agent_name:
                        if opponent not in agent_stats:
                            agent_stats[opponent] = AgentStats(opponent)
                        agent_stats[agent_name].add_head_to_head(opponent, won=False)
        
        # Also handle explicit matches list if present (old format)
        if 'matches' in data:
            for match in data['matches']:
                winner = match['winner']
                loser = match['loser']
                
                # Track in all_matches for later analysis
                all_matches.append({
                    'winner': winner,
                    'loser': loser,
                    'tournament': timestamp
                })
                
                # Update head-to-head stats
                if winner in agent_stats:
                    agent_stats[winner].add_head_to_head(loser, won=True)
                if loser in agent_stats:
                    agent_stats[loser].add_head_to_head(winner, won=False)
    
    # Filter by minimum tournaments
    if min_tournaments > 1:
        agent_stats = {
            name: stats for name, stats in agent_stats.items()
            if stats.total_tournaments >= min_tournaments
        }
    
    return agent_stats, all_matches


def analyze_hyperparameter_correlations(agent_stats: Dict[str, AgentStats]) -> Dict:
    """
    Analyze which hyperparameters correlate with better performance.
    
    Returns:
        Dictionary with hyperparameter analysis results
    """
    # Group agents by each hyperparameter value
    by_population = defaultdict(list)
    by_matchups = defaultdict(list)
    by_hands = defaultdict(list)
    by_sigma = defaultdict(list)
    
    for name, stats in agent_stats.items():
        spec = parse_genome_spec(name)
        if not spec:
            continue
        
        by_population[spec['population']].append((name, stats))
        by_matchups[spec['matchups']].append((name, stats))
        by_hands[spec['hands']].append((name, stats))
        by_sigma[spec['sigma']].append((name, stats))
    
    def avg_win_rate(agents: List[Tuple[str, AgentStats]]) -> float:
        if not agents:
            return 0.0
        return sum(stats.win_rate for _, stats in agents) / len(agents)
    
    def avg_chips(agents: List[Tuple[str, AgentStats]]) -> float:
        if not agents:
            return 0.0
        return sum(stats.avg_chips_per_tournament for _, stats in agents) / len(agents)
    
    # Calculate average performance for each hyperparameter value
    correlations = {
        'population': {
            pop: {
                'avg_win_rate': avg_win_rate(agents),
                'avg_chips': avg_chips(agents),
                'count': len(agents)
            }
            for pop, agents in by_population.items()
        },
        'matchups': {
            m: {
                'avg_win_rate': avg_win_rate(agents),
                'avg_chips': avg_chips(agents),
                'count': len(agents)
            }
            for m, agents in by_matchups.items()
        },
        'hands': {
            h: {
                'avg_win_rate': avg_win_rate(agents),
                'avg_chips': avg_chips(agents),
                'count': len(agents)
            }
            for h, agents in by_hands.items()
        },
        'sigma': {
            s: {
                'avg_win_rate': avg_win_rate(agents),
                'avg_chips': avg_chips(agents),
                'count': len(agents)
            }
            for s, agents in by_sigma.items()
        }
    }
    
    return correlations


def generate_recommendations(agent_stats: Dict[str, AgentStats],
                            correlations: Dict) -> List[str]:
    """
    Generate development recommendations based on analysis.
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # 1. Identify top performers
    if agent_stats:
        sorted_agents = sorted(agent_stats.values(),
                             key=lambda s: (s.win_rate, s.avg_chips_per_tournament),
                             reverse=True)
        
        top_agent = sorted_agents[0]
        top_spec = parse_genome_spec(top_agent.name)
        
        if top_spec:
            recommendations.append(
                f"BEST PERFORMER: {top_agent.name} with {top_agent.win_rate:.1%} win rate\n"
                f"  → Configuration: pop={top_spec['population']}, matchups={top_spec['matchups']}, "
                f"hands={top_spec['hands']}, sigma={top_spec['sigma']}"
            )
    
    # 2. Analyze population size
    if 'population' in correlations and correlations['population']:
        pop_data = correlations['population']
        best_pop = max(pop_data.items(), key=lambda x: x[1]['avg_win_rate'])
        recommendations.append(
            f"POPULATION SIZE: Best performing = {best_pop[0]} "
            f"(avg win rate: {best_pop[1]['avg_win_rate']:.1%})"
        )
    
    # 3. Analyze matchups per agent
    if 'matchups' in correlations and correlations['matchups']:
        matchup_data = correlations['matchups']
        best_matchups = max(matchup_data.items(), key=lambda x: x[1]['avg_win_rate'])
        recommendations.append(
            f"MATCHUPS PER AGENT: Best performing = {best_matchups[0]} "
            f"(avg win rate: {best_matchups[1]['avg_win_rate']:.1%})"
        )
    
    # 4. Analyze hands per matchup
    if 'hands' in correlations and correlations['hands']:
        hands_data = correlations['hands']
        best_hands = max(hands_data.items(), key=lambda x: x[1]['avg_win_rate'])
        recommendations.append(
            f"HANDS PER MATCHUP: Best performing = {best_hands[0]} "
            f"(avg win rate: {best_hands[1]['avg_win_rate']:.1%})"
        )
    
    # 5. Analyze sigma (mutation strength)
    if 'sigma' in correlations and correlations['sigma']:
        sigma_data = correlations['sigma']
        best_sigma = max(sigma_data.items(), key=lambda x: x[1]['avg_win_rate'])
        recommendations.append(
            f"SIGMA (MUTATION): Best performing = {best_sigma[0]} "
            f"(avg win rate: {best_sigma[1]['avg_win_rate']:.1%})"
        )
    
    # 6. Consistency analysis
    if agent_stats:
        sorted_by_consistency = sorted(
            [s for s in agent_stats.values() if s.total_tournaments >= 2],
            key=lambda s: s.chip_consistency
        )
        
        if sorted_by_consistency:
            most_consistent = sorted_by_consistency[0]
            recommendations.append(
                f"MOST CONSISTENT: {most_consistent.name} "
                f"(chip std dev: {most_consistent.chip_consistency:.0f})"
            )
    
    # 7. Development suggestions
    recommendations.append("\nDEVELOPMENT SUGGESTIONS:")
    
    if agent_stats:
        # Find agents with high win rate and high consistency
        good_candidates = [
            s for s in agent_stats.values()
            if s.win_rate > 0.5 and s.total_tournaments >= 2
        ]
        
        if good_candidates:
            good_candidates.sort(key=lambda s: (s.win_rate, -s.chip_consistency), reverse=True)
            recommendations.append(
                f"  → Continue training: {', '.join(s.name for s in good_candidates[:3])}"
            )
        
        # Find underperformers
        poor_performers = [
            s for s in agent_stats.values()
            if s.win_rate < 0.4 and s.total_tournaments >= 2
        ]
        
        if poor_performers:
            recommendations.append(
                f"  → Consider retiring: {', '.join(s.name for s in poor_performers[:3])}"
            )
        
        # Suggest hyperparameter exploration
        if 'population' in correlations and len(correlations['population']) > 1:
            pop_values = sorted(correlations['population'].keys())
            recommendations.append(
                f"  → Population sizes tested: {pop_values}"
            )
            
            # Suggest gaps to explore
            if max(pop_values) < 100:
                recommendations.append(
                    f"  → Try larger population sizes (60-100) for better diversity"
                )
    
    return recommendations


def print_report(agent_stats: Dict[str, AgentStats],
                correlations: Dict,
                recommendations: List[str],
                top_n: int = 10):
    """Print comprehensive analysis report to console."""
    
    print("\n" + "="*80)
    print(" TOURNAMENT HISTORY ANALYSIS ".center(80, "="))
    print("="*80 + "\n")
    
    # Overall statistics
    total_tournaments = max((s.total_tournaments for s in agent_stats.values()), default=0)
    total_agents = len(agent_stats)
    total_games = sum(s.total_games for s in agent_stats.values())
    
    print(f"Total Tournaments Analyzed: {total_tournaments}")
    print(f"Unique Agents: {total_agents}")
    print(f"Total Games Played: {total_games}")
    print()
    
    # Top performers
    print("-" * 80)
    print(f" TOP {top_n} AGENTS BY WIN RATE ".center(80, "-"))
    print("-" * 80)
    print(f"{'Rank':<6} {'Agent Name':<40} {'Win Rate':<12} {'W-L':<12} {'Avg Chips':<15}")
    print("-" * 80)
    
    sorted_agents = sorted(agent_stats.values(),
                          key=lambda s: (s.win_rate, s.avg_chips_per_tournament),
                          reverse=True)
    
    for rank, stats in enumerate(sorted_agents[:top_n], 1):
        print(f"{rank:<6} {stats.name:<40} {stats.win_rate:>6.1%}     "
              f"{stats.total_wins:>4}-{stats.total_losses:<5} {stats.avg_chips_per_tournament:>12,.0f}")
    
    print()
    
    # Hyperparameter analysis
    print("-" * 80)
    print(" HYPERPARAMETER CORRELATION ANALYSIS ".center(80, "-"))
    print("-" * 80)
    
    for param_name, param_data in correlations.items():
        if not param_data:
            continue
        
        print(f"\n{param_name.upper()}:")
        sorted_values = sorted(param_data.items(),
                             key=lambda x: x[1]['avg_win_rate'],
                             reverse=True)
        
        for value, metrics in sorted_values:
            print(f"  {value:>8}: Win Rate = {metrics['avg_win_rate']:>6.1%}, "
                  f"Avg Chips = {metrics['avg_chips']:>10,.0f}, "
                  f"Agents = {metrics['count']}")
    
    print()
    
    # Recommendations
    print("-" * 80)
    print(" RECOMMENDATIONS ".center(80, "-"))
    print("-" * 80)
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*80 + "\n")


def create_head_to_head_matrix(agent_stats: Dict[str, AgentStats]) -> Tuple[List[str], np.ndarray]:
    """
    Create a head-to-head win rate matrix for visualization.
    
    Returns:
        Tuple of (agent names list, win rate matrix)
    """
    if not MATPLOTLIB_AVAILABLE:
        return [], np.array([])
    
    agent_names = sorted(agent_stats.keys())
    n = len(agent_names)
    matrix = np.zeros((n, n))
    
    for i, agent1 in enumerate(agent_names):
        for j, agent2 in enumerate(agent_names):
            if i == j:
                matrix[i, j] = 0.5  # Neutral for self
            else:
                stats = agent_stats[agent1].opponents.get(agent2, {'wins': 0, 'losses': 0})
                total = stats['wins'] + stats['losses']
                if total > 0:
                    matrix[i, j] = stats['wins'] / total
                else:
                    matrix[i, j] = 0.5  # No data = neutral
    
    return agent_names, matrix


def create_visualizations(agent_stats: Dict[str, AgentStats],
                         correlations: Dict,
                         output_dir: Path):
    """Create and save visualization charts."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualizations (matplotlib not available)")
        return
    
    print("Generating visualizations...")
    
    # 1. Win Rate Comparison Bar Chart
    sorted_agents = sorted(agent_stats.values(),
                          key=lambda s: s.win_rate,
                          reverse=True)[:15]  # Top 15
    
    names = [s.name for s in sorted_agents]
    win_rates = [s.win_rate * 100 for s in sorted_agents]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(names)), win_rates, color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Win Rate (%)', fontsize=12)
    ax.set_title('Top 15 Agents by Win Rate', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        ax.text(rate + 0.5, i, f'{rate:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Average Chips per Tournament
    sorted_by_chips = sorted(agent_stats.values(),
                            key=lambda s: s.avg_chips_per_tournament,
                            reverse=True)[:15]
    
    names_chips = [s.name for s in sorted_by_chips]
    avg_chips = [s.avg_chips_per_tournament for s in sorted_by_chips]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(names_chips)), avg_chips, color='green', alpha=0.7)
    ax.set_yticks(range(len(names_chips)))
    ax.set_yticklabels(names_chips, fontsize=9)
    ax.set_xlabel('Average Chips per Tournament', fontsize=12)
    ax.set_title('Top 15 Agents by Average Chips', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, chips) in enumerate(zip(bars, avg_chips)):
        ax.text(chips + max(avg_chips)*0.01, i, f'{chips:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_chips_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Head-to-Head Matchup Matrix
    agent_names, matrix = create_head_to_head_matrix(agent_stats)
    
    if len(agent_names) > 0:
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(agent_names)))
        ax.set_yticks(range(len(agent_names)))
        ax.set_xticklabels(agent_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(agent_names, fontsize=8)
        
        ax.set_xlabel('Opponent', fontsize=12)
        ax.set_ylabel('Agent', fontsize=12)
        ax.set_title('Head-to-Head Win Rate Matrix\n(Row agent vs Column opponent)',
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Win Rate', rotation=270, labelpad=20)
        
        # Add text annotations for key matchups
        for i in range(min(len(agent_names), 10)):  # First 10 agents
            for j in range(min(len(agent_names), 10)):
                if i != j:
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'head_to_head_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Hyperparameter Correlation Charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hyperparameter Impact on Performance', fontsize=16, fontweight='bold')
    
    param_names = ['population', 'matchups', 'hands', 'sigma']
    titles = ['Population Size', 'Matchups per Agent', 'Hands per Matchup', 'Sigma (Mutation Strength)']
    
    for idx, (param, title) in enumerate(zip(param_names, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if param in correlations and correlations[param]:
            data = correlations[param]
            values = sorted(data.keys())
            win_rates = [data[v]['avg_win_rate'] * 100 for v in values]
            counts = [data[v]['count'] for v in values]
            
            # Create bar chart with counts as labels
            bars = ax.bar(range(len(values)), win_rates, color='coral', alpha=0.7)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(v) for v in values])
            ax.set_xlabel(title, fontsize=11)
            ax.set_ylabel('Avg Win Rate (%)', fontsize=11)
            ax.set_title(f'{title} vs Performance', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'n={count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Consistency Analysis
    agents_with_multiple = [s for s in agent_stats.values() if s.total_tournaments >= 2]
    if agents_with_multiple:
        sorted_consistency = sorted(agents_with_multiple,
                                   key=lambda s: s.chip_consistency)[:15]
        
        names_cons = [s.name for s in sorted_consistency]
        consistency = [s.chip_consistency for s in sorted_consistency]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(names_cons)), consistency, color='purple', alpha=0.6)
        ax.set_yticks(range(len(names_cons)))
        ax.set_yticklabels(names_cons, fontsize=9)
        ax.set_xlabel('Chip Standard Deviation (Lower = More Consistent)', fontsize=12)
        ax.set_title('Top 15 Most Consistent Agents', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'consistency_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def save_json_report(agent_stats: Dict[str, AgentStats],
                    correlations: Dict,
                    recommendations: List[str],
                    output_dir: Path):
    """Save analysis results as JSON."""
    
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_agents': len(agent_stats),
            'total_tournaments': max((s.total_tournaments for s in agent_stats.values()), default=0),
            'total_games': sum(s.total_games for s in agent_stats.values())
        },
        'agents': {},
        'correlations': correlations,
        'recommendations': recommendations
    }
    
    # Add detailed agent stats
    for name, stats in agent_stats.items():
        spec = parse_genome_spec(name)
        report['agents'][name] = {
            'win_rate': stats.win_rate,
            'total_wins': stats.total_wins,
            'total_losses': stats.total_losses,
            'tournaments_participated': stats.total_tournaments,
            'avg_chips_per_tournament': stats.avg_chips_per_tournament,
            'chip_consistency': stats.chip_consistency,
            'configuration': spec,
            'tournament_dates': stats.tournament_dates,
            'head_to_head': dict(stats.opponents)
        }
    
    # Save JSON
    with open(output_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"JSON report saved to {output_dir / 'analysis_report.json'}")


def save_text_report(agent_stats: Dict[str, AgentStats],
                    correlations: Dict,
                    recommendations: List[str],
                    output_dir: Path):
    """Save detailed report to text file."""
    
    output_file = output_dir / 'analysis_report.txt'
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" TOURNAMENT HISTORY ANALYSIS ".center(80, "=") + "\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        total_tournaments = max((s.total_tournaments for s in agent_stats.values()), default=0)
        total_agents = len(agent_stats)
        total_games = sum(s.total_games for s in agent_stats.values())
        
        f.write(f"Total Tournaments Analyzed: {total_tournaments}\n")
        f.write(f"Unique Agents: {total_agents}\n")
        f.write(f"Total Games Played: {total_games}\n\n")
        
        # Detailed agent statistics
        f.write("-" * 80 + "\n")
        f.write(" DETAILED AGENT STATISTICS ".center(80, "-") + "\n")
        f.write("-" * 80 + "\n\n")
        
        sorted_agents = sorted(agent_stats.values(),
                              key=lambda s: (s.win_rate, s.avg_chips_per_tournament),
                              reverse=True)
        
        for rank, stats in enumerate(sorted_agents, 1):
            f.write(f"{rank}. {stats.name}\n")
            f.write(f"   Win Rate: {stats.win_rate:.1%} ({stats.total_wins}W - {stats.total_losses}L)\n")
            f.write(f"   Tournaments: {stats.total_tournaments}\n")
            f.write(f"   Avg Chips/Tournament: {stats.avg_chips_per_tournament:,.0f}\n")
            f.write(f"   Chip Consistency (std dev): {stats.chip_consistency:.0f}\n")
            
            spec = parse_genome_spec(stats.name)
            if spec:
                f.write(f"   Config: pop={spec['population']}, matchups={spec['matchups']}, "
                       f"hands={spec['hands']}, sigma={spec['sigma']}")
                if spec['generations']:
                    f.write(f", gens={spec['generations']}")
                f.write("\n")
            
            f.write(f"   Tournament Dates: {', '.join(stats.tournament_dates)}\n\n")
        
        # Hyperparameter analysis
        f.write("-" * 80 + "\n")
        f.write(" HYPERPARAMETER CORRELATION ANALYSIS ".center(80, "-") + "\n")
        f.write("-" * 80 + "\n\n")
        
        for param_name, param_data in correlations.items():
            if not param_data:
                continue
            
            f.write(f"{param_name.upper()}:\n")
            sorted_values = sorted(param_data.items(),
                                 key=lambda x: x[1]['avg_win_rate'],
                                 reverse=True)
            
            for value, metrics in sorted_values:
                f.write(f"  {value:>8}: Win Rate = {metrics['avg_win_rate']:>6.1%}, "
                       f"Avg Chips = {metrics['avg_chips']:>10,.0f}, "
                       f"Agents = {metrics['count']}\n")
            f.write("\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write(" RECOMMENDATIONS ".center(80, "-") + "\n")
        f.write("-" * 80 + "\n\n")
        
        for rec in recommendations:
            f.write(rec + "\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Text report saved to: {output_file}")


def analyze_specific_matchups(agent_stats: Dict[str, AgentStats], output_dir: Path):
    """
    Analyze and report on specific agent-vs-agent matchups.
    Creates a detailed head-to-head comparison report.
    """
    
    output_file = output_dir / 'head_to_head_analysis.txt'
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" HEAD-TO-HEAD MATCHUP ANALYSIS ".center(80, "=") + "\n")
        f.write("="*80 + "\n\n")
        
        f.write("This report shows detailed head-to-head statistics between all agents\n")
        f.write("that have faced each other in tournaments.\n\n")
        
        # Get all agents sorted by win rate
        sorted_agents = sorted(agent_stats.values(),
                             key=lambda s: s.win_rate,
                             reverse=True)
        
        for agent in sorted_agents:
            if not agent.opponents:
                continue
            
            f.write("-" * 80 + "\n")
            f.write(f"{agent.name}\n")
            f.write(f"Overall: {agent.win_rate:.1%} ({agent.total_wins}W - {agent.total_losses}L)\n")
            f.write("-" * 80 + "\n\n")
            
            # Sort opponents by number of games played
            opponent_data = []
            for opp_name, stats in agent.opponents.items():
                total = stats['wins'] + stats['losses']
                win_rate = stats['wins'] / total if total > 0 else 0
                opponent_data.append((opp_name, stats['wins'], stats['losses'], win_rate))
            
            opponent_data.sort(key=lambda x: x[1] + x[2], reverse=True)
            
            f.write("  Matchups:\n")
            for opp_name, wins, losses, wr in opponent_data:
                total = wins + losses
                f.write(f"    vs {opp_name}: {wr:.1%} ({wins}W - {losses}L, {total} games)\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(" KEY RIVALRIES (Most Games Played) ".center(80, "=") + "\n")
        f.write("="*80 + "\n\n")
        
        # Find most common matchups
        matchup_counts = defaultdict(int)
        matchup_details = {}
        
        for agent in agent_stats.values():
            for opp_name, stats in agent.opponents.items():
                pair = tuple(sorted([agent.name, opp_name]))
                total = stats['wins'] + stats['losses']
                matchup_counts[pair] += total
                
                if pair not in matchup_details:
                    matchup_details[pair] = defaultdict(lambda: {'wins': 0, 'losses': 0})
                
                matchup_details[pair][agent.name]['wins'] += stats['wins']
                matchup_details[pair][agent.name]['losses'] += stats['losses']
        
        # Sort by total games
        top_matchups = sorted(matchup_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for (agent1, agent2), total_games in top_matchups:
            details = matchup_details[(agent1, agent2)]
            
            a1_wins = details[agent1]['wins']
            a1_losses = details[agent1]['losses']
            a2_wins = details[agent2]['wins']
            a2_losses = details[agent2]['losses']
            
            # Note: a1_losses should equal a2_wins and vice versa
            a1_record = f"{a1_wins}W - {a1_losses}L"
            a2_record = f"{a2_wins}W - {a2_losses}L"
            
            f.write(f"{agent1}  vs  {agent2}\n")
            f.write(f"  {agent1}: {a1_record}\n")
            f.write(f"  {agent2}: {a2_record}\n")
            f.write(f"  Total games: {total_games}\n\n")
    
    print(f"Head-to-head analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tournament history to identify best performing agents and development paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all tournaments
  python scripts/analyze_tournament_history.py
  
  # Only analyze agents that participated in 2+ tournaments
  python scripts/analyze_tournament_history.py --min-tournaments 2
  
  # Show top 5 agents and save to custom file
  python scripts/analyze_tournament_history.py --top-n 5 --output my_analysis.txt
        """
    )
    
    parser.add_argument('--min-tournaments', type=int, default=1,
                       help='Minimum tournaments an agent must participate in (default: 1)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top agents to show (default: 10)')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('tournament_reports') / 'overall_reports' / f'analysis_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Analyze tournament history
    agent_stats, all_matches = analyze_tournament_history(min_tournaments=args.min_tournaments)
    
    if not agent_stats:
        print("No agent data found matching criteria.")
        return
    
    # Analyze hyperparameter correlations
    correlations = analyze_hyperparameter_correlations(agent_stats)
    
    # Generate recommendations
    recommendations = generate_recommendations(agent_stats, correlations)
    
    # Print report to console
    print_report(agent_stats, correlations, recommendations, top_n=args.top_n)
    
    # Save reports
    save_text_report(agent_stats, correlations, recommendations, output_dir)
    save_json_report(agent_stats, correlations, recommendations, output_dir)
    
    # Analyze specific matchups
    analyze_specific_matchups(agent_stats, output_dir)
    
    # Create visualizations
    create_visualizations(agent_stats, correlations, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! All reports saved to:")
    print(f"  {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
