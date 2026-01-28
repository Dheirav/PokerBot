# Hall of Fame

This directory contains the best performing agent genomes from tournament play and training.

## Storage Format

Each genome file should be stored with a descriptive name following this pattern:
```
{agent_name}_gen{generation}.npy
```

For example:
- `p12_m8_h500_s0.08_g200_best.npy`
- `p12_m6_h750_s0.1_g200_hof.npy`

## Selection Criteria

Agents should be included in the Hall of Fame if they meet one or more of these criteria:

1. **Tournament Champions**: Won or placed highly in round-robin tournaments
2. **High Win Rate**: Consistently achieve >60% win rate against diverse opponents
3. **Generation Milestones**: Best performers at key generation checkpoints (50, 100, 200, etc.)
4. **Hyperparameter Representatives**: Best example of specific hyperparameter configurations

## Directory Structure

```
hall_of_fame/
├── README.md                    # This file
├── champions/                   # Tournament winners
├── milestones/                  # Generation milestone best performers
└── archived/                    # Previously top performers, now superseded
```

## Usage

These genomes can be loaded and used as:
- Training opponents for new agents
- Baseline comparisons for new hyperparameter configurations
- Starting points for continued evolution
- Reference implementations for behavior analysis

## Maintenance

Regularly review and update this collection based on tournament results. Use the tournament history analyzer to identify candidates:
```bash
python scripts/analysis/analyze_tournament_history.py --top-n 10
```
