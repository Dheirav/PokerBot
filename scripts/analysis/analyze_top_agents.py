#!/usr/bin/env python3
"""
Analyze the top N agents from the latest checkpoint.
Prints their parameters and playing style statistics.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.policy_network import PolicyNetwork
from training.config import TrainingConfig
from training.self_play import analyze_agent

def load_checkpoint_dir():
    # Find latest checkpoint dir
    ckpt_root = 'checkpoints'
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError('No checkpoints directory found.')
    
    # Get all checkpoint directories
    runs = [d for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))]
    if not runs:
        raise FileNotFoundError('No checkpoint runs found.')
    
    # Sort by modification time (most recent first)
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(ckpt_root, d)), reverse=True)
    
    # Try to find a checkpoint with population.npy or best_genome.npy
    for run in runs:
        run_path = os.path.join(ckpt_root, run)
        # Check in both the run directory and runs/ subdirectory
        for subdir in ['', 'runs']:
            check_dir = os.path.join(run_path, subdir) if subdir else run_path
            if os.path.isdir(check_dir):
                # Look for subdirectories with run_* pattern
                if subdir == 'runs':
                    run_subdirs = [d for d in os.listdir(check_dir) 
                                  if os.path.isdir(os.path.join(check_dir, d)) and d.startswith('run_')]
                    if run_subdirs:
                        # Use most recent run
                        run_subdirs.sort(reverse=True)
                        check_dir = os.path.join(check_dir, run_subdirs[0])
                
                if os.path.exists(os.path.join(check_dir, 'population.npy')) or \
                   os.path.exists(os.path.join(check_dir, 'best_genome.npy')):
                    return check_dir
    
    # If no valid checkpoint found, return the most recent one anyway
    return os.path.join(ckpt_root, runs[0])

def main(top_n=3):
    ckpt_dir = load_checkpoint_dir()
    print(f"Analyzing checkpoint: {ckpt_dir}\n")
    
    # Try to load population weights
    pop_path = os.path.join(ckpt_dir, 'population.npy')
    best_path = os.path.join(ckpt_dir, 'best_genome.npy')
    
    pop_weights = None
    if os.path.exists(pop_path):
        pop_weights = np.load(pop_path)
        print(f"Loaded population with {len(pop_weights)} agents")
    elif os.path.exists(best_path):
        best_weights = np.load(best_path)
        pop_weights = [best_weights]
        print(f"Loaded best_genome.npy (single agent)")
    else:
        print(f"Warning: No population.npy or best_genome.npy found in {ckpt_dir}")
        print("Skipping this checkpoint.\n")
        return
    # Load config
    config_path = os.path.join(ckpt_dir, 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config_dict = json.load(f)
        net_config = TrainingConfig().network
        if 'network' in config_dict:
            for k, v in config_dict['network'].items():
                setattr(net_config, k, v)
    else:
        net_config = TrainingConfig().network
    # Analyze top N
    for i in range(min(top_n, len(pop_weights))):
        print(f"=== Agent #{i} ===")
        net = PolicyNetwork(net_config)
        net.set_weights_from_genome(pop_weights[i])
        print(f"Weights (first 10): {pop_weights[i][:10]}")
        print(f"Total params: {len(pop_weights[i])}")
        # Analyze playing style
        stats = analyze_agent(weights_path=None, num_hands=1000, verbose=True, network=net)
        print()

if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(top_n=n)
