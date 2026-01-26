"""
Plot fitness and diversity curves from history.json after training.
"""
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_history(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    generations = [g['generation'] for g in history]
    mean_fitness = [g['mean_fitness'] for g in history]
    max_fitness = [g['max_fitness'] for g in history]
    min_fitness = [g['min_fitness'] for g in history]
    median_fitness = [g.get('median_fitness', 0) for g in history]
    diversity = [g.get('diversity', 0) for g in history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_fitness, label='Mean Fitness')
    plt.plot(generations, max_fitness, label='Max Fitness')
    plt.plot(generations, min_fitness, label='Min Fitness')
    plt.plot(generations, median_fitness, label='Median Fitness', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fitness_curve.png')
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(generations, diversity, label='Mean Pairwise Diversity', color='purple')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Diversity over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('diversity_curve.png')
    plt.close()
    print('Saved fitness_curve.png and diversity_curve.png')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_history.py /path/to/history.json')
        sys.exit(1)
    plot_history(sys.argv[1])
