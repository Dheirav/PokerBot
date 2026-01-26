"""
Evaluate the best agent against baseline agents and log results.
"""
import numpy as np
import os
import json
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from training.genome import GenomeFactory
from training.policy_network import PolicyNetwork
from training.config import TrainingConfig

def load_best_genome(weights_path, config):
    weights = np.load(weights_path)
    factory = GenomeFactory(config.network, config.evolution)
    return factory.create_from_weights(weights)

def evaluate_vs_baseline(best_genome, config, num_games=100):
    # Example: play best_genome vs. heuristic and random agents
    # You must implement the actual game logic
    results = {'heuristic': 0, 'random': 0}
    # ... run games and update results ...
    return results

def main():
    # Paths (edit as needed)
    best_weights = 'checkpoints/evolution_run/best_genome.npy'
    config_path = 'checkpoints/evolution_run/config.json'
    with open(config_path) as f:
        config_dict = json.load(f)
    config = TrainingConfig.from_dict(config_dict['training'])
    best_genome = load_best_genome(best_weights, config)
    results = evaluate_vs_baseline(best_genome, config)
    print('Baseline evaluation results:', results)
    with open('checkpoints/evolution_run/baseline_eval.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
