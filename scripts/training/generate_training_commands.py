#!/usr/bin/env python3
"""
Generate Training Commands for Specific Configurations

Interactive script to input hyperparameter configurations and generate
training commands for both 50 and 200 generation runs.

Usage:
    # Interactive mode
    python scripts/training/generate_training_commands.py
    
    # Batch mode with predefined configs
    python scripts/training/generate_training_commands.py --batch
    
    # Generate from file
    python scripts/training/generate_training_commands.py --config-file configs.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class ConfigGenerator:
    def __init__(self):
        self.configs = []
        self.output_commands = []
        
    def add_config(self, pop: int, matchups: int, hands: int, sigma: float, hof_count: int = 3):
        """Add a configuration to the list."""
        config = {
            'population': pop,
            'matchups': matchups, 
            'hands': hands,
            'sigma': sigma,
            'hof_count': hof_count
        }
        self.configs.append(config)
        
    def generate_config_name(self, config: Dict) -> str:
        """Generate a standard configuration name."""
        hof_str = f"_hof{config['hof_count']}" if config['hof_count'] > 0 else ""
        return f"p{config['population']}_m{config['matchups']}_h{config['hands']}_s{config['sigma']}{hof_str}"
        
    def generate_training_command(self, config: Dict, generations: int) -> str:
        """Generate a training command for the given configuration."""
        name = self.generate_config_name(config)
        
        # Base command
        cmd = [
            "python scripts/training/train.py",
            f"--pop {config['population']}",
            f"--matchups {config['matchups']}",
            f"--hands {config['hands']}", 
            f"--sigma {config['sigma']}",
            f"--gens {generations}",
            f"--name {name}_g{generations}",
            "--workers 4"
        ]
        
        # Add HOF if specified
        if config['hof_count'] > 0:
            cmd.extend([
                "--tournament-winners",
                f"--hof-count {config['hof_count']}"
            ])
            
        return " \\\n    ".join(cmd)
        
    def interactive_input(self):
        """Interactive configuration input."""
        print("🎯 Interactive Configuration Input")
        print("=" * 40)
        
        while True:
            try:
                print(f"\nConfiguration #{len(self.configs) + 1}:")
                
                pop = int(input("Population size: "))
                matchups = int(input("Matchups per agent: "))
                hands = int(input("Hands per matchup: "))
                sigma = float(input("Mutation sigma: "))
                
                hof_input = input("Hall of Fame count (0 for none, default 3): ").strip()
                hof_count = 3 if hof_input == "" else int(hof_input)
                
                self.add_config(pop, matchups, hands, sigma, hof_count)
                
                name = self.generate_config_name(self.configs[-1])
                print(f"✓ Added config: {name}")
                
                another = input("\nAdd another configuration? (y/N): ").strip().lower()
                if another not in ['y', 'yes']:
                    break
                    
            except (ValueError, KeyboardInterrupt):
                print("\n❌ Invalid input or cancelled")
                break
                
    def batch_input(self):
        """Add some common configurations for batch mode."""
        print("🚀 Adding Common High-Performance Configurations")
        print("=" * 50)
        
        # Based on analysis results - top performing configs
        batch_configs = [
            (12, 6, 375, 0.1, 3),   # Champion config
            (12, 10, 375, 0.09, 3), # Population 12 optimal
            (20, 9, 375, 0.09, 3),  # Population 20 optimal
            (40, 7, 375, 0.09, 3),  # Population 40 optimal
            (12, 8, 500, 0.1, 3),   # Alternative 12 config
            (20, 8, 500, 0.09, 3),  # Alternative 20 config
            (12, 7, 750, 0.09, 3),  # High depth config
            (40, 8, 500, 0.08, 3),  # Large population config
        ]
        
        for i, (pop, matchups, hands, sigma, hof) in enumerate(batch_configs, 1):
            self.add_config(pop, matchups, hands, sigma, hof)
            name = self.generate_config_name(self.configs[-1])
            print(f"{i:2d}. {name}")
            
    def load_from_file(self, filepath: str):
        """Load configurations from a text file."""
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    try:
                        # Expected format: pop,matchups,hands,sigma,hof_count
                        parts = line.split(',')
                        if len(parts) < 4:
                            print(f"⚠️  Line {line_num}: Not enough values (need at least 4)")
                            continue
                            
                        pop = int(parts[0])
                        matchups = int(parts[1])
                        hands = int(parts[2])
                        sigma = float(parts[3])
                        hof_count = int(parts[4]) if len(parts) > 4 else 3
                        
                        self.add_config(pop, matchups, hands, sigma, hof_count)
                        
                    except ValueError:
                        print(f"⚠️  Line {line_num}: Invalid format '{line}'")
                        
            print(f"✓ Loaded {len(self.configs)} configurations from {filepath}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            
    def generate_all_commands(self):
        """Generate training commands for all configurations."""
        if not self.configs:
            print("❌ No configurations to generate")
            return
            
        print(f"\n🔥 Generating Training Commands for {len(self.configs)} Configurations")
        print("=" * 60)
        
        # Generate for both 50 and 200 generations
        for generations in [50, 200]:
            print(f"\n## {generations} Generation Runs")
            print("-" * 30)
            
            for i, config in enumerate(self.configs, 1):
                name = self.generate_config_name(config)
                cmd = self.generate_training_command(config, generations)
                
                print(f"\n# Config {i}: {name} ({generations} gens)")
                print(cmd)
                print()
                
                self.output_commands.append((f"{name}_g{generations}", cmd))
                
    def save_to_script(self, filename: str = "train_configs.sh"):
        """Save all commands to a bash script."""
        script_path = Path(filename)
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated Training Commands\n")
            f.write(f"# Total configurations: {len(self.configs)}\n")
            f.write(f"# Generated on: $(date)\n\n")
            
            f.write("set -e  # Exit on any error\n\n")
            
            for name, cmd in self.output_commands:
                f.write(f"echo \"🚀 Starting: {name}\"\n")
                f.write(f"{cmd}\n")
                f.write(f"echo \"✅ Completed: {name}\"\n")
                f.write("echo\n\n")
                
        # Make script executable
        script_path.chmod(0o755)
        
        print(f"💾 Saved {len(self.output_commands)} commands to {script_path}")
        print(f"📝 Run with: ./{filename}")
        
    def show_summary(self):
        """Show summary of configurations."""
        if not self.configs:
            return
            
        print(f"\n📊 Configuration Summary ({len(self.configs)} configs)")
        print("=" * 50)
        
        for i, config in enumerate(self.configs, 1):
            name = self.generate_config_name(config)
            print(f"{i:2d}. {name}")
            print(f"    Pop: {config['population']}, Matchups: {config['matchups']}, " 
                  f"Hands: {config['hands']}, Sigma: {config['sigma']}, HOF: {config['hof_count']}")
                  
        total_runs = len(self.configs) * 2  # 50 + 200 gens each
        est_time_50 = len(self.configs) * 4  # ~4 min per 50-gen run
        est_time_200 = len(self.configs) * 15  # ~15 min per 200-gen run
        total_time = est_time_50 + est_time_200
        
        print(f"\n⏱️  Estimated Training Time:")
        print(f"   50-gen runs:  {est_time_50:3d} minutes ({len(self.configs)} configs)")
        print(f"   200-gen runs: {est_time_200:3d} minutes ({len(self.configs)} configs)")
        print(f"   Total:        {total_time:3d} minutes ({total_time/60:.1f} hours)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate training commands for specific configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python scripts/training/generate_training_commands.py
    
    # Batch mode with common configs
    python scripts/training/generate_training_commands.py --batch
    
    # Load from file (CSV format: pop,matchups,hands,sigma,hof_count)
    python scripts/training/generate_training_commands.py --config-file my_configs.txt
    
    # Save to custom script name
    python scripts/training/generate_training_commands.py --batch --output my_training.sh
        """
    )
    
    parser.add_argument('--batch', action='store_true',
                       help='Use predefined high-performance configurations')
    parser.add_argument('--config-file', type=str,
                       help='Load configurations from CSV file')
    parser.add_argument('--output', type=str, default='train_configs.sh',
                       help='Output script filename (default: train_configs.sh)')
    parser.add_argument('--no-save', action='store_true',
                       help='Only display commands, do not save to script')
    
    args = parser.parse_args()
    
    generator = ConfigGenerator()
    
    # Input configurations
    if args.config_file:
        generator.load_from_file(args.config_file)
    elif args.batch:
        generator.batch_input()
    else:
        generator.interactive_input()
        
    if not generator.configs:
        print("❌ No configurations provided")
        return 1
        
    # Show summary
    generator.show_summary()
    
    # Generate commands
    generator.generate_all_commands()
    
    # Save to script
    if not args.no_save:
        generator.save_to_script(args.output)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        sys.exit(1)