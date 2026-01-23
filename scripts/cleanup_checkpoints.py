"""
Utility to compress or clean up old checkpoints, keeping only the most recent or best ones.
"""
import os
import shutil
from pathlib import Path
import argparse

def cleanup_checkpoints(run_dir, keep_last=3, keep_best=True):
    run_dir = Path(run_dir)
    # Find all subdirs (assume each is a checkpoint)
    checkpoints = sorted([d for d in run_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    to_keep = checkpoints[:keep_last]
    if keep_best:
        # Optionally keep the one with the best fitness
        best = None
        best_fitness = float('-inf')
        for d in checkpoints:
            state_path = d / 'state.json'
            if state_path.exists():
                import json
                with open(state_path) as f:
                    state = json.load(f)
                if state.get('best_fitness', float('-inf')) > best_fitness:
                    best_fitness = state['best_fitness']
                    best = d
        if best and best not in to_keep:
            to_keep.append(best)
    # Delete the rest
    for d in checkpoints:
        if d not in to_keep:
            print(f'Removing old checkpoint: {d}')
            shutil.rmtree(d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', help='Directory containing checkpoint subfolders')
    parser.add_argument('--keep_last', type=int, default=3, help='Number of recent checkpoints to keep')
    parser.add_argument('--no_keep_best', action='store_true', help='Do not keep best checkpoint')
    args = parser.parse_args()
    cleanup_checkpoints(args.run_dir, keep_last=args.keep_last, keep_best=not args.no_keep_best)

if __name__ == '__main__':
    main()
