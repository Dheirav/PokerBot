#!/usr/bin/env python3
"""
Analyze the top N agents from the latest checkpoint.
Prints their parameters and playing style statistics.
"""

import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import PolicyNetwork, analyze_agent, TrainingConfig

def load_checkpoint_dir():
    # Find latest checkpoint dir
    ckpt_root = 'checkpoints'
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError('No checkpoints directory found.')
    runs = [d for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))]
    if not runs:
        raise FileNotFoundError('No checkpoint runs found.')
    # Use most recent
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(ckpt_root, d)), reverse=True)
    return os.path.join(ckpt_root, runs[0])

def main(top_n=3):
    ckpt_dir = load_checkpoint_dir()
    print(f"Analyzing checkpoint: {ckpt_dir}\n")
    # Load population weights
    pop_path = os.path.join(ckpt_dir, 'population.npy')
    if not os.path.exists(pop_path):
        raise FileNotFoundError(f'No population.npy in {ckpt_dir}')
    pop_weights = np.load(pop_path)
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
