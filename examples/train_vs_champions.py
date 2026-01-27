"""
Example: Load tournament winners into Hall of Fame for training.

This demonstrates how to train new agents against proven strong opponents.
"""
import numpy as np
from pathlib import Path
from training.evolution import EvolutionTrainer
from training.config import TrainingConfig, NetworkConfig, EvolutionConfig, FitnessConfig

# Load the tournament winners
hof_models = [
    'checkpoints/deep_p40_m6_h500_s0.15/run_20260126_042643/best_genome.npy',
    'checkpoints/deep_p40_m8_h375_s0.1/run_20260126_093215/best_genome.npy',
    'checkpoints/deep_p20_m6_h500_s0.15/run_20260126_000023/best_genome.npy',
    'checkpoints/deep_p12_m6_h500_s0.15/run_20260125_222103/best_genome.npy',
]

hof_weights = []
for path_str in hof_models:
    path = Path(path_str)
    if path.exists():
        hof_weights.append(np.load(path))
        print(f"Loaded: {path.name}")

print(f"\nTotal HoF models loaded: {len(hof_weights)}")

# Configure training
config = TrainingConfig(
    network=NetworkConfig(
        input_size=17,
        hidden_sizes=[128, 128],
        output_size=6,
    ),
    evolution=EvolutionConfig(
        population_size=12,
        mutation_sigma=0.15,
        num_elites=2,
        num_immigrants=1,
    ),
    fitness=FitnessConfig(
        matchups_per_agent=6,
        hands_per_matchup=500,
    ),
    num_generations=50,
    seed=42,
    experiment_name='vs_tournament_winners',
)

# Initialize with HoF opponents
trainer = EvolutionTrainer(config)
trainer.initialize(hof_weights=hof_weights)

# Train - your new agents will be evaluated against the loaded champions
print("\nTraining against tournament winners...")
best = trainer.train()

print(f"\nBest fitness: {best.fitness:.2f} BB/100")
print("If positive, your new agent beats the tournament winners on average!")
