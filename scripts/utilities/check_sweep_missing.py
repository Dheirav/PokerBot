#!/usr/bin/env python3
"""
Check which hyperparameter configurations are missing from a sweep.

This script reads a sweep's results.json file and compares the actual
configurations run against the intended sweep grid (stored in the results.json),
then outputs missing configurations that need to be completed.

Usage:
    python scripts/utilities/check_sweep_missing.py hyperparam_results/sweep_hof_20260129_062341
    python scripts/utilities/check_sweep_missing.py hyperparam_results/sweep_20260127_133129/results.json
    python scripts/utilities/check_sweep_missing.py hyperparam_results/sweep_20260127_133129 --out-dir hyperparam_results/missing_sweeps

The script automatically reads the sweep_input from results.json to determine
which configurations were supposed to be run.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product


def load_results(results_path):
    """Load results from JSON file. Handles both old and new formats."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Handle both old format (list) and new format (dict with sweep_input)
    if isinstance(data, dict) and 'results' in data:
        return data['results'], data.get('sweep_input')
    else:
        # Old format - no sweep_input embedded
        return data, None


def build_expected_names(sweep_input, name_format=None):
    """Build expected configuration names from sweep_input metadata."""
    # Handle both old and new key names
    pops = sweep_input.get('population_sizes') or sweep_input.get('pop') or sweep_input.get('pops', [])
    matchups = sweep_input.get('matchups', [])
    hands = sweep_input.get('hands', [])
    sigmas = sweep_input.get('mutation_sigmas') or sweep_input.get('sigma') or sweep_input.get('sigmas', [])
    
    if not (pops and matchups and hands and sigmas):
        raise ValueError(f"Could not extract complete hyperparameter grid from sweep_input: {sweep_input}")
    
    # Determine name format from hof_count if available
    hof_count = sweep_input.get('hof_count')
    if name_format is None:
        if hof_count:
            name_format = f'p{{pop}}_m{{m}}_h{{h}}_s{{s}}_hof{hof_count}'
        else:
            name_format = 'p{pop}_m{m}_h{h}_s{s}'
    
    names = []
    for pop, m, h, s in product(pops, matchups, hands, sigmas):
        # Format sigma without padding (0.1 not 0.10) to match actual names
        # Python's default str() and repr() will give 0.1 for 0.1
        s_form = f"{s:g}" if s % 1 != 0 else f"{int(s)}"
        names.append(name_format.format(pop=pop, m=m, h=h, s=s_form))
    
    return sorted(names)


def save_missing_list(out_dir, sweep_name, missing, sweep_input):
    """Save missing configuration list and run instructions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sweep-specific subdirectory
    sub = out_dir / sweep_name
    sub.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    missing_json = sub / f'missing_{ts}.json'
    missing_txt = sub / f'missing_{ts}.txt'
    run_input_file = sub / f'run_input_{ts}.json'
    latest_missing = sub / 'latest_missing.json'
    latest_txt = sub / 'latest_missing.txt'
    
    # Write missing configurations
    with open(missing_json, 'w') as f:
        json.dump(missing, f, indent=2)
    
    with open(missing_txt, 'w') as f:
        f.write('\n'.join(missing))
        if missing:
            f.write('\n')
    
    # Write run instructions (how to reproduce missing configs)
    run_input = {
        'script': 'scripts/training/hyperparam_sweep_with_hof.py',
        'original_sweep_input': sweep_input,
        'missing_configs': missing,
        'timestamp': datetime.now().isoformat(),
        'note': 'Re-run with --pop, --matchups, --hands, --sigma flags containing only missing values'
    }
    
    with open(run_input_file, 'w') as f:
        json.dump(run_input, f, indent=2)
    
    # Update "latest" pointers for quick access
    with open(latest_missing, 'w') as f:
        f.write(str(missing_json))
    
    with open(latest_txt, 'w') as f:
        f.write(str(missing_txt))
    
    return missing_json, missing_txt, run_input_file


def main():
    parser = argparse.ArgumentParser(
        description='Check which hyperparameter configurations are missing from a sweep',
        epilog='Examples:\n'
               '  # New interface (reads sweep_input from results.json):\n'
               '  python scripts/utilities/check_sweep_missing.py hyperparam_results/sweep_hof_20260129_062341\n'
               '  python scripts/utilities/check_sweep_missing.py hyperparam_results/sweep_20260127_133129/results.json\n'
               '\n'
               '  # Old interface (backward compatible with explicit parameters):\n'
               '  python scripts/utilities/check_sweep_missing.py sweep_XXX \\\n'
               '    --pops 12,20,40 \\\n'
               '    --matchups 6,8,10 \\\n'
               '    --hands 375,500,750 \\\n'
               '    --sigmas 0.08,0.10,0.12',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'sweep_path',
        nargs='?',
        help='Path to sweep folder or results.json file'
    )
    parser.add_argument(
        '--out-dir',
        default='hyperparam_results/missing_sweeps',
        help='Directory to save missing configuration lists (default: hyperparam_results/missing_sweeps)'
    )
    
    # Backward compatibility: Old CLI parameters
    parser.add_argument(
        '--pops',
        help='[DEPRECATED] Comma-separated population sizes (e.g., 12,20,40). Only used if sweep_input not in results.json'
    )
    parser.add_argument(
        '--matchups',
        help='[DEPRECATED] Comma-separated matchups per agent (e.g., 6,8,10). Only used if sweep_input not in results.json'
    )
    parser.add_argument(
        '--hands',
        help='[DEPRECATED] Comma-separated hands per matchup (e.g., 375,500,750). Only used if sweep_input not in results.json'
    )
    parser.add_argument(
        '--sigmas',
        help='[DEPRECATED] Comma-separated mutation sigma values (e.g., 0.08,0.10,0.12). Only used if sweep_input not in results.json'
    )
    
    args = parser.parse_args()
    
    results = None
    sweep_input = None
    results_path = None
    sweep_path = None
    
    # Step 1: Determine results.json path
    if args.sweep_path:
        sweep_path = Path(args.sweep_path)
        
        if sweep_path.suffix == '.json':
            results_path = sweep_path
            sweep_name = sweep_path.parent.name
        else:
            results_path = sweep_path / 'results.json'
            sweep_name = sweep_path.name
        
        if not results_path.exists():
            print(f"❌ results.json not found at: {results_path}")
            return
    elif not (args.pops and args.matchups and args.hands and args.sigmas):
        # No sweep_path provided and no CLI parameters - try to find latest sweep
        hyperparam_dir = Path('hyperparam_results')
        if not hyperparam_dir.exists():
            print(f"❌ hyperparam_results directory not found. Provide an explicit sweep path or CLI parameters.")
            parser.print_help()
            return
        
        sweep_dirs = sorted(
            [d for d in hyperparam_dir.glob('sweep_*') if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not sweep_dirs:
            print(f"❌ No sweep directories found in {hyperparam_dir}")
            return
        
        sweep_path = sweep_dirs[0]
        sweep_name = sweep_path.name
        results_path = sweep_path / 'results.json'
        
        if not results_path.exists():
            print(f"❌ results.json not found at: {results_path}")
            return
    
    # Step 2: Load results from file if it exists
    if results_path and results_path.exists():
        print(f"\n📂 Loading results from: {results_path}")
        try:
            results, sweep_input = load_results(results_path)
            print(f"✓ Loaded {len(results)} configurations from results.json")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error reading results.json: {e}")
            return
    
    # Step 3: Handle sweep_input availability
    if sweep_input is None:
        # No sweep_input in file - try to construct from CLI parameters (backward compatibility)
        if not (args.pops and args.matchups and args.hands and args.sigmas):
            print(f"\n⚠️  WARNING: sweep_input not found in results.json")
            print("Provide CLI parameters to check the sweep, or re-run with a recent version of the sweep runner:")
            print(f"\n  python scripts/utilities/check_sweep_missing.py {sweep_path.name} \\")
            print(f"    --pops <comma-separated> \\")
            print(f"    --matchups <comma-separated> \\")
            print(f"    --hands <comma-separated> \\")
            print(f"    --sigmas <comma-separated>")
            return
        
        # Parse CLI parameters (backward compatibility mode)
        try:
            sweep_input = {
                'population_sizes': [int(x.strip()) for x in args.pops.split(',')],
                'matchups': [int(x.strip()) for x in args.matchups.split(',')],
                'hands': [int(x.strip()) for x in args.hands.split(',')],
                'mutation_sigmas': [float(x.strip()) for x in args.sigmas.split(',')],
                'generations': None,
                'timestamp': None,
            }
            print(f"\n✓ Using backward-compatible CLI parameters (sweep_input not in results.json)")
        except (ValueError, AttributeError) as e:
            print(f"❌ Error parsing parameters: {e}")
            return
    else:
        # sweep_input found in file - warn if CLI parameters also provided
        if args.pops or args.matchups or args.hands or args.sigmas:
            print(f"ℹ️  Ignoring CLI parameters (sweep_input found in results.json)")
    
    # Step 4: Normalize sweep_input keys
    if 'population_sizes' not in sweep_input:
        if 'pop' in sweep_input:
            sweep_input['population_sizes'] = sweep_input['pop']
        elif 'pops' in sweep_input:
            sweep_input['population_sizes'] = sweep_input['pops']
    
    if 'mutation_sigmas' not in sweep_input:
        if 'sigma' in sweep_input:
            sweep_input['mutation_sigmas'] = sweep_input['sigma']
        elif 'sigmas' in sweep_input:
            sweep_input['mutation_sigmas'] = sweep_input['sigmas']
    
    # Step 5: Build expected and actual configuration lists
    print(f"✓ Sweep input metadata: {sweep_input}")
    
    # Auto-detect hof_count from actual configurations if not in sweep_input
    if 'hof_count' not in sweep_input and results:
        for config in results:
            if 'name' in config and '_hof' in config['name']:
                # Extract hof count from name (e.g., "p12_m7_h375_s0.07_hof3" -> 3)
                import re
                match = re.search(r'_hof(\d+)', config['name'])
                if match:
                    sweep_input['hof_count'] = int(match.group(1))
                    print(f"ℹ️  Auto-detected hof_count={sweep_input['hof_count']} from configuration names")
                    break
    
    expected_names = build_expected_names(sweep_input)
    print(f"✓ Expected {len(expected_names)} total configurations")
    
    if results:
        names_run = {entry.get('name', '') for entry in results if 'name' in entry}
        print(f"✓ Found {len(names_run)} executed configurations")
    else:
        names_run = set()
    
    # Find missing
    missing = sorted([name for name in expected_names if name not in names_run])
    completed = len(expected_names) - len(missing)
    
    # Step 6: Report results
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETION REPORT")
    print(f"{'='*70}")
    if sweep_path:
        print(f"\nDirectory:        {sweep_path}")
    print(f"Total Expected:   {len(expected_names)}")
    print(f"Completed:        {completed}")
    print(f"Missing:          {len(missing)}")
    if len(expected_names) > 0:
        print(f"Progress:         {100*completed//len(expected_names)}% ({completed}/{len(expected_names)})")
    
    if missing:
        print(f"\n{'='*70}")
        print(f"MISSING CONFIGURATIONS ({len(missing)} total)")
        print(f"{'='*70}")
        for name in missing[:20]:
            print(f"  - {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more")
    else:
        print(f"\n✅ All configurations completed!")
    
    if missing and sweep_path:
        # Save missing list
        print(f"\n{'='*70}")
        print(f"SAVING MISSING CONFIGURATION LIST")
        print(f"{'='*70}")
        
        out_json, out_txt, run_input_file = save_missing_list(
            args.out_dir,
            sweep_name,
            missing,
            sweep_input
        )
        
        print(f"\n📁 Saved to:")
        print(f"   - {out_json}")
        print(f"   - {out_txt}")
        print(f"   - {run_input_file}")
        
        print(f"\n💡 To complete missing configs, extract unique values and re-run:")
        # Extract unique values from missing configs for re-run suggestion
        import re
        missing_pops = sorted(set(int(m.split('_')[0][1:]) for m in missing))
        missing_matchups = sorted(set(int(m.split('_')[1][1:]) for m in missing))
        missing_hands = sorted(set(int(m.split('_')[2][1:]) for m in missing))
        missing_sigmas = sorted(set(float(m.split('_')[3][1:]) for m in missing))
        
        cmd = f"python scripts/training/hyperparam_sweep_with_hof.py \\\n"
        cmd += f"  --pop {' '.join(map(str, missing_pops))} \\\n"
        cmd += f"  --matchups {' '.join(map(str, missing_matchups))} \\\n"
        cmd += f"  --hands {' '.join(map(str, missing_hands))} \\\n"
        cmd += f"  --sigma {' '.join(map(str, missing_sigmas))} \\\n"
        cmd += f"  --tournament-winners \\\n"
        cmd += f"  --gens {sweep_input.get('generations', 50)}"
        
        print(f"\n{cmd}\n")
    
    print(f"✅ Done!\n")


if __name__ == '__main__':
    main()
