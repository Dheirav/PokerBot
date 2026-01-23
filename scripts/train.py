#!/usr/bin/env python3
"""
Training script for evolutionary poker AI.

Usage:
    python scripts/train.py                  # Default settings
    python scripts/train.py --quick          # Quick test (5 generations)
    python scripts/train.py --production     # Full production run
    python scripts/train.py --resume PATH    # Resume from checkpoint

Examples:
    # Quick test run
    python3 scripts/train.py --quick --seed 42
    
    # Full training with custom settings
    python3 scripts/train.py --pop 30 --gens 200 --hands 1000
    
    # Resume from checkpoint
    python3 scripts/train.py --resume checkpoints/evolution_run
"""
import os
# Optimize numpy for multiprocessing - prevent thread conflicts
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import (
    EvolutionTrainer,
    TrainingConfig,
    NetworkConfig,
    EvolutionConfig,
    FitnessConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train poker AI via evolutionary self-play',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true',
                      help='Quick test run (5 generations)')
    mode.add_argument('--production', action='store_true',
                      help='Full production training')
    mode.add_argument('--resume', type=str, default=None,
                      help='Resume from checkpoint directory')
    mode.add_argument('--phase1', action='store_true',
                      help='Phase 1: Stabilization and Debug Phase')
    mode.add_argument('--phase2', action='store_true',
                      help='Phase 2: Strategy Refinement Phase')

    # Curriculum argument (not mutually exclusive)
    parser.add_argument('--curriculum', action='store_true',
                       help='Curriculum learning: gradually increase difficulty')

    # Evolution parameters
    parser.add_argument('--pop', type=int, default=20,
                        help='Population size')
    parser.add_argument('--gens', type=int, default=100,
                        help='Number of generations')
    parser.add_argument('--elite', type=float, default=0.1,
                        help='Elite fraction to preserve')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Mutation strength (sigma)')
    
    # Fitness evaluation
    parser.add_argument('--hands', type=int, default=500,
                        help='Hands per matchup')
    parser.add_argument('--matchups', type=int, default=4,
                        help='Matchups per agent')
    parser.add_argument('--players', type=int, default=6,
                        help='Players per table (2-6)')
    
    # Network architecture
    parser.add_argument('--hidden', type=int, nargs='+', default=[64, 32],
                        help='Hidden layer sizes')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='evolution_run',
                        help='Experiment name')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers (default 4 for speedup)')
    
    parser.add_argument('--seed-weights', type=str, default=None,
                        help='Path to .npy file with weights to seed initial population')
    return parser.parse_args()


def create_config(args) -> TrainingConfig:
    if getattr(args, 'curriculum', False):
        # Start with simple config, then increase complexity in main()
        return TrainingConfig(
            network=NetworkConfig(hidden_sizes=[32]),
            evolution=EvolutionConfig(
                population_size=12,
                elite_fraction=0.2,
                mutation_sigma=0.1,
                hof_size=4,
            ),
            fitness=FitnessConfig(
                hands_per_matchup=300,
                matchups_per_agent=4,
                num_players=2,
                starting_stack=2000,
                small_blind=5,
                big_blind=10,
                ante=0,
                num_workers=2,
            ),
            num_generations=40,
            seed=args.seed,
            output_dir=args.output,
            experiment_name=args.name + '_curriculum',
        )
    """Create training config from arguments."""
    if args.quick:
        config = TrainingConfig.for_quick_test()
        config.seed = args.seed
        return config
    if args.production:
        return TrainingConfig.for_production(seed=args.seed)
    if args.phase1:
        return TrainingConfig.for_phase_1(seed=args.seed)
    if args.phase2:
        return TrainingConfig.for_phase_2(seed=args.seed)
    return TrainingConfig(
        network=NetworkConfig(
            hidden_sizes=args.hidden,
        ),
        evolution=EvolutionConfig(
            population_size=args.pop,
            elite_fraction=args.elite,
            mutation_sigma=args.sigma,
        ),
        fitness=FitnessConfig(
            hands_per_matchup=args.hands,
            matchups_per_agent=args.matchups,
            num_players=args.players,
            num_workers=args.workers,
        ),
        num_generations=args.gens,
        seed=args.seed,
        output_dir=args.output,
        experiment_name=args.name,
    )


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Evolutionary Poker AI Training")
    print("=" * 60)
    
    # Create config
    config = create_config(args)
    
    # Create trainer
    trainer = EvolutionTrainer(config)

    # Resume or initialize
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    else:
        print("\nInitializing new population...")
        seed_weights = None
        if args.seed_weights:
            import numpy as np
            print(f"Loading seed weights from {args.seed_weights}...")
            seed_weights = np.load(args.seed_weights)
        trainer.initialize(seed_weights=seed_weights)
    
    # Print config summary
    print(f"\nConfiguration:")
    print(f"  Population size: {config.evolution.population_size}")
    print(f"  Generations: {config.num_generations}")
    print(f"  Elite fraction: {config.evolution.elite_fraction}")
    print(f"  Mutation sigma: {config.evolution.mutation_sigma}")
    print(f"  Hands per matchup: {config.fitness.hands_per_matchup}")
    print(f"  Matchups per agent: {config.fitness.matchups_per_agent}")
    print(f"  Players per table: {config.fitness.num_players}")
    print(f"  Network architecture: {trainer.factory._network.layer_sizes}")
    print(f"  Genome size: {trainer.factory.genome_size} parameters")
    print(f"  Seed: {config.seed}")
    print(f"  Output: {trainer.output_dir}")
    
    # Curriculum learning: phase-by-phase training
    if getattr(args, 'curriculum', False):
        curriculum_phases = [
            # (description, config overrides, generations)
            ("Phase 1: Heads-up, deep stacks", dict(
                fitness=dict(num_players=2, starting_stack=2000, small_blind=5, big_blind=10, ante=0),
                network=dict(hidden_sizes=[32]),
                evolution=dict(population_size=12, mutation_sigma=0.1, hof_size=4),
                num_generations=40,
            ), 40),
            ("Phase 2: 3 players, moderate stacks", dict(
                fitness=dict(num_players=3, starting_stack=1500, small_blind=10, big_blind=20, ante=0),
                network=dict(hidden_sizes=[64, 32]),
                evolution=dict(population_size=16, mutation_sigma=0.07, hof_size=6),
                num_generations=40,
            ), 40),
            ("Phase 3: 6-max, standard stacks", dict(
                fitness=dict(num_players=6, starting_stack=1000, small_blind=5, big_blind=10, ante=1),
                network=dict(hidden_sizes=[128, 64]),
                evolution=dict(population_size=24, mutation_sigma=0.05, hof_size=10),
                num_generations=60,
            ), 60),
        ]
        for desc, overrides, gens in curriculum_phases:
            print(f"\n{'='*60}\n{desc}\n{'='*60}")
            # Apply overrides
            for section, vals in overrides.items():
                if section == 'fitness':
                    for k, v in vals.items():
                        setattr(trainer.config.fitness, k, v)
                elif section == 'network':
                    for k, v in vals.items():
                        setattr(trainer.config.network, k, v)
                elif section == 'evolution':
                    for k, v in vals.items():
                        setattr(trainer.config.evolution, k, v)
                elif section == 'num_generations':
                    trainer.config.num_generations = v
            trainer.config.experiment_name += f"_{desc.split(':')[0].replace(' ', '')}"
            trainer.save_checkpoint()  # Save before phase
            best = trainer.train()
            trainer.save_checkpoint()  # Save after phase
        print("\nCurriculum training complete!")
        print(f"Best fitness: {trainer.best_fitness:.2f} BB/100")
        print(f"Best genome: {trainer.best_genome}")
        print(f"Checkpoint saved to: {trainer.output_dir}")
        baseline_score = trainer.evaluate_best(num_hands=2000)
        print(f"Best agent vs random: {baseline_score:.2f} BB/100")
    else:
        # Standard training
        try:
            best = trainer.train()
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            print(f"Best fitness: {trainer.best_fitness:.2f} BB/100")
            print(f"Best genome: {trainer.best_genome}")
            print(f"Checkpoint saved to: {trainer.output_dir}")
            print("\nFinal evaluation against random opponents...")
            baseline_score = trainer.evaluate_best(num_hands=2000)
            print(f"Best agent vs random: {baseline_score:.2f} BB/100")
        except KeyboardInterrupt:
            print("\n\nTraining interrupted. Saving checkpoint...")
            trainer.save_checkpoint()
            print("Checkpoint saved. You can resume with --resume flag.")
            sys.exit(1)


if __name__ == '__main__':
    main()
