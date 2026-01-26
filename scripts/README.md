# Scripts Directory

**Complete guide to all training, evaluation, analysis, and utility scripts**

---

## 📋 Quick Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| [train.py](#trainpy) | Main training | Train new poker AI agents |
| [eval_baseline.py](#eval_baselinepy) | Agent evaluation | Test trained agent vs baselines |
| [match_agents.py](#match_agentspy) | Head-to-head matches | Compare two specific agents |
| [round_robin_agents_config.py](#round_robin_agents_configpy) | Tournament | Find best agent among all trained |
| [hyperparam_sweep.py](#hyperparam_sweeppy) | Quick optimization | Find good hyperparameters fast |
| [deep_hyperparam_sweep.py](#deep_hyperparam_sweeppy) | Thorough optimization | Comprehensive hyperparameter search |
| [plot_history.py](#plot_historypy) | Training visualization | See fitness curves and progress |
| [analyze_top_agents.py](#analyze_top_agentspy) | Agent analysis | Deep dive into agent performance |
| [visualize_agent_behavior.py](#visualize_agent_behaviorpy) | Behavior analysis | Understand agent decision patterns |
| [cleanup_checkpoints.py](#cleanup_checkpointspy) | Disk management | Free up storage space |

---

## 🎯 Training Scripts

### train.py

**Purpose**: Main evolutionary training script for creating poker AI agents

**When to Use**:
- Starting a new training run from scratch
- Resuming interrupted training
- Training with specific hyperparameters

**Usage**:
```bash
# Quick test (5 generations, ~30 seconds)
python scripts/train.py --quick

# Standard training (50 generations, ~4-5 minutes)
python scripts/train.py

# Full production run (200 generations, ~15-20 minutes)
python scripts/train.py --production

# Custom configuration
python scripts/train.py \
    --pop 40 \              # Population size
    --gens 100 \            # Generations
    --hands 3000 \          # Hands per matchup
    --matchups 12 \         # Matchups per agent
    --workers 4 \           # Parallel workers
    --seed 42               # Random seed

# Resume from checkpoint
python scripts/train.py --resume checkpoints/evolution_run
```

**Output**:
- `checkpoints/evolution_run/` - Training checkpoints
- `checkpoints/evolution_run/best_genome.npy` - Best agent
- `checkpoints/evolution_run/history.json` - Training history
- `checkpoints/evolution_run/config.json` - Configuration used

**Time**: ~4-6 seconds per generation (with Numba)

---

### hyperparam_sweep.py

**Purpose**: Quick hyperparameter search to find good training configurations

**When to Use**:
- Before starting full training runs
- Testing different population sizes, mutation rates
- Finding best hands/matchups settings

**Usage**:
```bash
# Quick sweep (6 configs, ~2-3 minutes)
python scripts/hyperparam_sweep.py --quick

# Normal sweep (12 configs, ~5-10 minutes)
python scripts/hyperparam_sweep.py

# Thorough sweep (24+ configs, ~15-30 minutes)
python scripts/hyperparam_sweep.py --thorough

# Custom sweep
python scripts/hyperparam_sweep.py --generations 20 --trials 15
```

**Output**:
- `hyperparam_results/sweep_YYYYMMDD_HHMMSS/results.json` - All results
- Console output with ranked configurations

**Follow-up**:
```bash
# Analyze convergence
python scripts/analyze_convergence.py

# Generate visualizations
python scripts/visualize_hyperparam_sweep.py
```

---

### deep_hyperparam_sweep.py

**Purpose**: Comprehensive hyperparameter optimization with longer evaluation

**When to Use**:
- After initial quick sweep narrows down options
- Need highly-optimized training configuration
- Preparing for large-scale training runs

**Usage**:
```bash
# Deep sweep with 50 generations per config
python scripts/deep_hyperparam_sweep.py

# Custom deep sweep
python scripts/deep_hyperparam_sweep.py --generations 100 --trials 20

# Specific parameter ranges
python scripts/deep_hyperparam_sweep.py \
    --pop-sizes 12,20,40 \
    --sigmas 0.1,0.15,0.2 \
    --hands 500,1000,2000
```

**Output**:
- `hyperparam_results/deep_sweep_YYYYMMDD_HHMMSS/` - Detailed results
- `deep_sweep_report.txt` - Summary report

**Time**: 20-60 minutes depending on configuration

**Follow-up**:
```bash
# Generate detailed report
python scripts/report_deep_sweep.py
```

---

## 📊 Evaluation Scripts

### eval_baseline.py

**Purpose**: Evaluate trained agent against baseline opponents (random, heuristic)

**When to Use**:
- After training completes
- Testing agent strength
- Comparing different training runs

**Usage**:
```bash
# Evaluate best agent
python scripts/eval_baseline.py checkpoints/evolution_run/best_genome.npy

# Multiple games for more reliable results
python scripts/eval_baseline.py checkpoints/evolution_run/best_genome.npy --games 1000
```

**Output**:
- Win rates vs RandomAgent and HeuristicAgent
- BB/100 statistics
- Console summary

---

### match_agents.py

**Purpose**: Head-to-head match between two specific agents

**When to Use**:
- Comparing two training runs
- Testing which hyperparameters produced better agent
- Debugging agent performance

**Usage**:
```bash
# Basic match (10,000 hands)
python scripts/match_agents.py \
    --agent1 checkpoints/run1/best_genome.npy \
    --arch1 "17 64 32 6" \
    --agent2 checkpoints/run2/best_genome.npy \
    --arch2 "17 64 32 6"

# More hands for reliable results
python scripts/match_agents.py \
    --agent1 checkpoints/run1/best_genome.npy \
    --arch1 "17 64 32 6" \
    --agent2 checkpoints/run2/best_genome.npy \
    --arch2 "17 64 32 6" \
    --hands 50000

# Enable detailed logging (for debugging)
python scripts/match_agents.py \
    --agent1 agent1.npy \
    --arch1 "17 64 32 6" \
    --agent2 agent2.npy \
    --arch2 "17 64 32 6" \
    --log
```

**Output**:
- Final chip counts
- Winner determination
- Match logs in `match_logs/` (if --log enabled)

**Time**: ~30-60 seconds for 10,000 hands

---

### round_robin_agents_config.py

**Purpose**: Tournament between all trained agents to find the best

**When to Use**:
- After multiple training runs completed
- Comparing different hyperparameter configurations
- Finding overall best agent

**Usage**:
```bash
# Run full tournament (all agents vs all agents)
python scripts/round_robin_agents_config.py
```

**Requirements**:
- Multiple agents in `checkpoints/*/runs/*/best_genome.npy`
- Each agent folder has `config.json`

**Output**:
- `tournament_reports/tournament_YYYYMMDD_HHMMSS/` - Timestamped results
  - `report.json` - Detailed matchup data
  - `wins_losses.png` - Win/loss bar chart
  - `parameter_performance.png` - Which configs perform best
  - `head_to_head_heatmap.png` - All matchup results
  - `chip_distribution.png` - Total chips per agent

**Time**: 5-15 minutes depending on number of agents

---

## 📈 Analysis Scripts

### plot_history.py

**Purpose**: Visualize training progress with fitness curves

**When to Use**:
- After training run completes
- Checking if training converged
- Comparing multiple training runs

**Usage**:
```bash
# Plot single training run
python scripts/plot_history.py checkpoints/evolution_run/

# Compare multiple runs
python scripts/plot_history.py \
    checkpoints/run1/ \
    checkpoints/run2/ \
    checkpoints/run3/
```

**Output**:
- Fitness curves (mean, best, diversity)
- PNG files saved to checkpoint directory

---

### analyze_top_agents.py

**Purpose**: Deep performance analysis of elite agents

**When to Use**:
- Understanding why top agents perform well
- Comparing elite vs average agents
- Strategy analysis

**Usage**:
```bash
# Analyze top 5 agents from a run
python scripts/analyze_top_agents.py checkpoints/evolution_run/

# Analyze more agents
python scripts/analyze_top_agents.py checkpoints/evolution_run/ --top 10
```

**Output**:
- Performance statistics per agent
- Win rate breakdowns
- Strategy patterns
- Console report

---

### analyze_convergence.py

**Purpose**: Detect if hyperparameter sweep configs have converged

**When to Use**:
- After running hyperparameter sweeps
- Before deciding which config to use for full training
- Determining if more generations needed

**Usage**:
```bash
# Analyze most recent sweep
python scripts/analyze_convergence.py

# Analyze specific sweep
python scripts/analyze_convergence.py --sweep-dir hyperparam_results/sweep_20260126_120000
```

**Output**:
- `convergence_analysis.txt` - Detailed convergence patterns
- Warnings if configs still improving
- Recommendations for full training

---

### visualize_agent_behavior.py

**Purpose**: Understand agent decision-making patterns

**When to Use**:
- Debugging strange agent behavior
- Understanding learned strategies
- Comparing aggressive vs conservative agents

**Usage**:
```bash
# Visualize single agent
python scripts/visualize_agent_behavior.py checkpoints/evolution_run/best_genome.npy

# Compare multiple agents
python scripts/visualize_agent_behavior.py \
    checkpoints/run1/best_genome.npy \
    checkpoints/run2/best_genome.npy
```

**Output**:
- Action distribution heatmaps
- Position-based strategy charts
- Hand strength vs action plots
- PNG visualizations

---

### visualize_hyperparam_sweep.py

**Purpose**: Generate comprehensive visualizations from hyperparameter sweeps

**When to Use**:
- After running hyperparam_sweep.py or deep_hyperparam_sweep.py
- Need visual comparison of configurations
- Presenting results

**Usage**:
```bash
# Visualize most recent sweep
python scripts/visualize_hyperparam_sweep.py

# Visualize specific sweep
python scripts/visualize_hyperparam_sweep.py --sweep-dir hyperparam_results/sweep_20260126_120000
```

**Requirements**:
- `matplotlib`, `seaborn` installed
- Sweep results in `hyperparam_results/`

**Output**:
- `visualizations/` folder with multiple plots:
  - Comparison plots (box plots, violin plots)
  - Parameter heatmaps
  - Learning curves
  - Best configuration highlights

---

### report_deep_sweep.py

**Purpose**: Generate detailed text report from deep hyperparameter sweep

**When to Use**:
- After deep_hyperparam_sweep.py completes
- Need summary of which configs performed best

**Usage**:
```bash
# Generate report from most recent deep sweep
python scripts/report_deep_sweep.py
```

**Output**:
- `deep_sweep_report.txt` - Ranked configurations by fitness
- Console summary

---

## 🧪 Testing & Debugging Scripts

### test_ai_hands.py

**Purpose**: Test agent decisions on specific poker scenarios

**When to Use**:
- Debugging agent behavior
- Testing specific situations (bluffing, pot odds, etc.)
- Validating agent logic

**Usage**:
```bash
# Test specific scenarios
python scripts/test_ai_hands.py

# Test with specific agent
python scripts/test_ai_hands.py --agent checkpoints/evolution_run/best_genome.npy
```

**Output**:
- Agent decisions for predefined scenarios
- Console output with reasoning

---

### test_ai_features.py

**Purpose**: Verify feature extraction is working correctly

**When to Use**:
- Debugging training issues
- Ensuring input features are correct
- Testing feature engineering changes

**Usage**:
```bash
# Test feature extraction
python scripts/test_ai_features.py
```

**Output**:
- Feature vector examples
- Validation checks
- Console report

---

### test_cli.py

**Purpose**: Interactive poker game for testing engine

**When to Use**:
- Testing game engine manually
- Debugging game logic
- Playing against agents interactively

**Usage**:
```bash
# Start interactive game
python scripts/test_cli.py
```

**Controls**:
- `f` - Fold
- `c` - Check/Call
- `r <amount>` - Raise
- `a` - All-in
- `q` - Quit

---

### benchmark_jit.py

**Purpose**: Measure Numba JIT compilation speedup

**When to Use**:
- Verifying Numba is installed correctly
- Measuring optimization impact
- Performance tuning

**Usage**:
```bash
# Run benchmark suite
python scripts/benchmark_jit.py
```

**Output**:
- Speedup measurements for each JIT function
- With vs without Numba comparison
- Console report

**Expected**: 2-3× speedup with Numba enabled

---

## 🛠️ Utility Scripts

### cleanup_checkpoints.py

**Purpose**: Free up disk space by removing old training runs

**When to Use**:
- Running low on disk space
- After training many experimental runs
- Before archiving project

**Usage**:
```bash
# Keep only 5 most recent checkpoints
python scripts/cleanup_checkpoints.py --keep 5

# Keep checkpoints from last 7 days
python scripts/cleanup_checkpoints.py --older-than 7

# Dry run (see what would be deleted)
python scripts/cleanup_checkpoints.py --dry-run --keep 3
```

**⚠️ Warning**: Deleted checkpoints cannot be recovered!

---

## 🔄 Typical Workflow

### 1️⃣ Initial Training
```bash
# Quick test to verify setup
python scripts/train.py --quick

# Standard training run
python scripts/train.py --pop 20 --gens 100
```

### 2️⃣ Hyperparameter Optimization
```bash
# Quick sweep to narrow options
python scripts/hyperparam_sweep.py

# Visualize results
python scripts/visualize_hyperparam_sweep.py

# Deep sweep on promising configs
python scripts/deep_hyperparam_sweep.py

# Generate report
python scripts/report_deep_sweep.py
```

### 3️⃣ Production Training
```bash
# Train with best hyperparameters
python scripts/train.py --production --pop 40 --sigma 0.15
```

### 4️⃣ Evaluation & Analysis
```bash
# Plot training progress
python scripts/plot_history.py checkpoints/evolution_run/

# Evaluate vs baselines
python scripts/eval_baseline.py checkpoints/evolution_run/best_genome.npy

# Analyze behavior
python scripts/visualize_agent_behavior.py checkpoints/evolution_run/best_genome.npy
```

### 5️⃣ Tournament
```bash
# Run tournament between all trained agents
python scripts/round_robin_agents_config.py

# View results in tournament_reports/tournament_YYYYMMDD_HHMMSS/
```

---

## 📦 Required vs Optional Scripts

### ✅ Required (Core Functionality)
- **train.py** - Essential for training
- **match_agents.py** - Used by tournament scripts
- **eval_baseline.py** - Standard evaluation

### 🎯 Highly Recommended
- **round_robin_agents_config.py** - Find best agent
- **plot_history.py** - Visualize progress
- **hyperparam_sweep.py** - Optimize training

### 📊 Analysis Tools
- **analyze_top_agents.py** - Deep analysis
- **visualize_agent_behavior.py** - Strategy insights
- **analyze_convergence.py** - Convergence detection

### 🔧 Utilities
- **cleanup_checkpoints.py** - Disk management
- **benchmark_jit.py** - Performance verification
- **test_*.py** - Testing and debugging

### 🚀 Advanced
- **deep_hyperparam_sweep.py** - Thorough optimization
- **report_deep_sweep.py** - Report generation
- **visualize_hyperparam_sweep.py** - Visual analysis

---

## ⚡ Performance Tips

**Faster Training**:
- Install Numba: `pip install numba` (2-3× speedup)
- Increase workers: `--workers 8`
- Reduce hands for quick tests: `--hands 1000`

**Better Results**:
- More generations: `--gens 200+`
- Larger population: `--pop 40+`
- More hands: `--hands 5000+`
- Use hyperparameter sweeps first

**Debugging**:
- Use `--quick` flag for fast iterations
- Enable logging with `--log` flag
- Test with `test_ai_hands.py`
- Check features with `test_ai_features.py`

---

## 🆘 Troubleshooting

**Script won't run**:
- Check Python version: `python --version` (need 3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Check current directory: `pwd` (should be PokerBot root)

**Out of memory**:
- Reduce `--workers`
- Reduce `--pop` (population size)
- Use `cleanup_checkpoints.py` to free space

**Slow performance**:
- Install Numba: `pip install numba`
- Verify with `benchmark_jit.py`
- Check CPU usage (should use all cores)

**Results not improving**:
- Try hyperparameter sweep
- Increase training time (more generations)
- Check convergence with `analyze_convergence.py`

---

**For more information, see main [README.md](../README.md)**
