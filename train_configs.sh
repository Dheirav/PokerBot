#!/bin/bash
# Run specific 29 configurations efficiently (no subprocesses)
# Total: 29 unique configs × 2 generation counts = 58 runs
# With Hall of Fame pre-seeding from tournament champions

set -e  # Exit on any error

echo "🚀 Running 29 specific configs in single process (no subprocess overhead)"
echo ""

python3 run_specific_configs.py --gens 50 200 --tournament-winners

echo ""
echo "✅ Completed all configurations"
 