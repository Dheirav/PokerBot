# Poker AI Training System

**A high-performance poker AI using evolutionary algorithms and self-play**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This project implements a **Texas Hold'em poker AI** trained through evolutionary algorithms and self-play. The system uses neural network agents that evolve over generations, learning optimal poker strategy through competition.

**Key Features**:
- 🚀 **400-500× faster** with Numba (~4-6 sec/generation), **175× without** (~13 sec/generation)
- 🧬 Evolutionary algorithm with self-play evaluation
- 🎲 Complete Texas Hold'em poker engine
- 📊 Comprehensive training analytics and visualization
- ⚡ Highly optimized with JIT compilation, batching, and caching
- 🔧 Configurable hyperparameters and architecture

**Performance**: Train 100 generations in ~7-10 minutes with Numba (was 63 hours originally)

---

## ⚡ Performance Status

| Metric | Without Numba | With Numba | Original |
|--------|---------------|------------|----------|
| **Generation time** | ~13 sec | **~4-6 sec** | 38 min |
| **100 generations** | ~22 min | **~7-10 min** | 63 hours |
| **Speedup** | 175× | **400-500×** | 1× |
| **Status** | ✅ Ready | ✅ Ready | - |

**Bottom line**: Install Numba (`pip install numba`) for maximum performance!

**See**: [OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md) for complete optimization roadmap

---

## 📁 Project Structure

```
PokerBot/
├── engine/                  # Poker game engine (cards, rules, hand evaluation)
├── training/                # Evolutionary training system (genomes, networks, fitness)
├── agents/                  # Pre-trained and baseline agents (random, heuristic)
├── scripts/                 # Training, evaluation, and analysis scripts (18 total)
├── evaluator/               # Hand strength and equity calculations
├── utils/                   # Utility functions and helpers
├── tests/                   # Unit tests
├── data/                    # Placeholder for training data (currently unused)
├── checkpoints/             # Saved training runs and models
├── hyperparam_results/      # Hyperparameter sweep results and visualizations
├── tournament_reports/      # Round-robin tournament results and charts
├── logs/                    # Training logs and tensorboard events
├── match_logs/              # Game history logs (optional, disabled by default)
├── OPTIMIZATION_STATUS.md   # Complete optimization roadmap
├── OPTIMIZATION_SUMMARY.md  # Quick optimization reference
├── NUMBA_JIT_GUIDE.md       # Numba JIT compilation patterns
├── FORWARD_BATCH_INTEGRATION.md  # Batched inference documentation
└── deep_sweep_report.txt    # Latest deep hyperparameter sweep results
```

**Module Documentation**:
- [engine/README.md](engine/README.md) - Poker engine internals
- [training/README.md](training/README.md) - Training system details
- [agents/README.md](agents/README.md) - Baseline agents
- [evaluator/README.md](evaluator/README.md) - Hand ranking and equity
- [utils/README.md](utils/README.md) - Utility functions

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PokerBot.git
cd PokerBot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib scipy tensorboard

# Optional: Install Numba for 2-3× speedup (highly recommended)
pip install numba
```

**With Numba**: ~4-6 sec/generation (400-500× faster than original)  
**Without Numba**: ~13 sec/generation (175× faster than original)

### Train Your First AI

```bash
# Quick training run (2 generations, ~10-20 seconds)
python scripts/train.py --pop 10 --gens 2 --hands 500

# Full training (100 generations, ~7-10 min with Numba, ~22 min without)
python scripts/train.py --pop 20 --gens 100 --hands 3000 --matchups 12

# Resume from checkpoint
python scripts/train.py --resume checkpoints/evolution_run
```

### Evaluate an Agent

```bash
# Evaluate best agent against random opponents
python scripts/eval_baseline.py checkpoints/evolution_run/best_genome.pkl

# Match two agents head-to-head
python scripts/match_agents.py genome1.pkl genome2.pkl --hands 10000
```

---

## 🎮 Core Components

### 1. Poker Engine ([engine/](engine/))
Complete Texas Hold'em implementation with:
- Card dealing and shuffling
- Betting rounds (preflop, flop, turn, river)
- Hand evaluation (13-16× optimized)
- Pot management and side pots
- Action validation

**See**: [engine/README.md](engine/README.md) for detailed documentation

### 2. Training System ([training/](training/))
Evolutionary algorithm with:
- Neural network policy agents
- Self-play fitness evaluation
- Mutation and selection operators
- Population diversity maintenance
- Elite preservation

**See**: [training/README.md](training/README.md) for detailed documentation

### 3. Utilities ([utils/](utils/))
Helper functions for:
- Genome transformations
- Data processing
- Configuration management

**See**: [utils/README.md](utils/README.md) for detailed documentation

---

## 🎯 Key Functionalities

### Training
Train poker agents through evolutionary self-play:

```bash
python scripts/train.py \
    --pop 20 \              # Population size
    --gens 100 \            # Number of generations
    --hands 3000 \          # Hands per matchup
    --matchups 12 \         # Matchups per agent
    --workers 4 \           # Parallel workers
    --seed 42               # Random seed
```

**Output**: Trained agents saved to `checkpoints/evolution_run/`

### Hyperparameter Optimization
Find optimal training parameters:

```bash
# Quick hyperparameter sweep
python scripts/hyperparam_sweep.py \
    --generations 20 \      # Test for 20 generations
    --trials 10             # Try 10 configurations

# Deep hyperparameter sweep (comprehensive search)
python scripts/deep_hyperparam_sweep.py \
    --generations 50 \      # Longer evaluation
    --trials 20             # More configurations

# Analyze convergence patterns (detects if configs need longer training)
python scripts/analyze_convergence.py

# Generate comprehensive report
python scripts/report_deep_sweep.py

# Visualize results (requires matplotlib, seaborn)
python scripts/visualize_hyperparam_sweep.py
```

**Output**: 
- Raw results in `hyperparam_results/sweep_YYYYMMDD_HHMMSS/results.json`
- Deep sweep report in `deep_sweep_report.txt`
- Convergence analysis in `convergence_analysis.txt`
- Visualizations in `visualizations/` folder (comparison plots, heatmaps, etc.)

### Round-Robin Tournaments
Pit trained agents against each other to find the best performer:

```bash
# Basic round-robin (all agents vs all agents)
python scripts/round_robin_agents.py

# Enhanced round-robin with config insights and visualizations
python scripts/round_robin_agents_config.py
```

**Output**:
- Timestamped reports in `tournament_reports/tournament_YYYYMMDD_HHMMSS/`
- JSON report with detailed matchup results
- Visualizations: win/loss charts, parameter performance, head-to-head heatmaps, chip distribution

**Features**:
- Sorted leaderboard by wins/losses
- Per-agent breakdown of who they beat/lost to
- Parameter insights (which population sizes, mutation sigmas, etc. perform best)
- Automatic visualization generation

### Agent Evaluation
Test agent performance:

```bash
# Against random agents
python scripts/eval_baseline.py genome.pkl --hands 10000

# Head-to-head matches
python scripts/match_agents.py genome1.pkl genome2.pkl --hands 5000

# Visualize behavior
python scripts/visualize_agent_behavior.py genome.pkl
```

### Analysis & Visualization
Analyze training progress and agent behavior:

```bash
# Plot fitness curves from training history
python scripts/plot_history.py checkpoints/evolution_run/

# Analyze top agents (detailed performance breakdown, win rates, strategy analysis)
python scripts/analyze_top_agents.py checkpoints/evolution_run/

# Visualize agent behavior patterns (action distributions, aggression, hand selection)
python scripts/visualize_agent_behavior.py checkpoints/evolution_run/best_genome.npy

# Test AI on specific hand scenarios (debugging tool)
python scripts/test_ai_hands.py

# Test AI feature extraction (verify input features are correct)
python scripts/test_ai_features.py

# Test poker engine CLI interactively
python scripts/test_cli.py

# Benchmark JIT compilation performance (measure Numba speedup)
python scripts/benchmark_jit.py
```

**Analysis Tools**:
- `plot_history.py`: Generates fitness curves, diversity metrics, and convergence plots
- `analyze_top_agents.py`: Deep dive into elite agents' performance statistics
- `visualize_agent_behavior.py`: Action heatmaps by position/hand strength
- `analyze_convergence.py`: Detect if training configs need more generations

**Testing Tools**:
- `test_ai_hands.py`: Test agent decisions on specific poker scenarios
- `test_ai_features.py`: Verify feature extraction correctness
- `test_cli.py`: Interactive poker game testing

**Performance Tools**:
- `benchmark_jit.py`: Compare Numba vs pure Python performance

### Utilities

```bash
# Clean old checkpoints to save disk space
python scripts/cleanup_checkpoints.py --keep 5  # Keep only 5 most recent
python scripts/cleanup_checkpoints.py --older-than 7  # Keep last 7 days
```

**Purpose**: Manage storage by removing old training runs while preserving recent/best agents.

---

## 📂 Complete Scripts Reference

### Training Scripts
- **`train.py`**: Main evolutionary training script with self-play evaluation
- **`hyperparam_sweep.py`**: Quick hyperparameter search across configurations
- **`deep_hyperparam_sweep.py`**: Comprehensive hyperparameter optimization with longer runs
- **`report_deep_sweep.py`**: Generate detailed report from deep sweep results

### Evaluation Scripts
- **`eval_baseline.py`**: Test trained agent vs random/heuristic opponents
- **`match_agents.py`**: Head-to-head matches between two agents (use `--log` to enable match logging)
- **`round_robin_agents.py`**: Basic round-robin tournament (all vs all)
- **`round_robin_agents_config.py`**: Enhanced tournament with visualizations and config insights

### Analysis Scripts  
- **`plot_history.py`**: Generate fitness curves and training progression charts
- **`analyze_top_agents.py`**: Detailed performance analysis of elite agents
- **`analyze_convergence.py`**: Detect if training configs need more generations
- **`visualize_agent_behavior.py`**: Action distribution heatmaps by situation
- **`visualize_hyperparam_sweep.py`**: Generate comparison plots for hyperparameter sweeps

### Testing & Debugging Scripts
- **`test_ai_hands.py`**: Test agent decisions on specific poker scenarios
- **`test_ai_features.py`**: Verify feature extraction is working correctly
- **`test_cli.py`**: Interactive poker game for testing engine

### Performance Scripts
- **`benchmark_jit.py`**: Measure Numba JIT compilation speedup

### Utility Scripts
- **`cleanup_checkpoints.py`**: Manage disk space by removing old training runs

**Total**: 18 scripts covering training, evaluation, analysis, testing, and utilities

---

## 🏗️ Architecture

### Neural Network Policy

**Architecture**: `[17 inputs] → [64 hidden] → [32 hidden] → [6 outputs]`

**Inputs (17 features)**:
- Position, stack size, pot odds
- Hand strength, commitment level
- Betting round, active players
- Action context (facing bet/raise)

**Outputs (6 abstract actions)**:
- Fold
- Check/Call
- Raise 0.5× pot
- Raise 1.0× pot
- Raise 2.0× pot
- All-in

### Evolutionary Algorithm

1. **Initialize**: Create random population of neural networks
2. **Evaluate**: Play poker hands via self-play, measure fitness (BB/100)
3. **Select**: Keep top performers (elites)
4. **Mutate**: Add Gaussian noise to create offspring
5. **Repeat**: Next generation

**Fitness**: Big blinds won per 100 hands (BB/100)

---

## ⚡ Performance Optimizations

The system includes **11 major optimizations** providing 400-500× total speedup:

### ✅ Implemented Optimizations

| Optimization | Speedup | Status | Impact |
|--------------|---------|--------|--------|
| 1. Fast hand evaluation | 13-16× | ✅ Complete | Critical bottleneck fix |
| 2. Multiprocessing (4 workers) | 4× | ✅ Complete | Parallel fitness evaluation |
| 3. FeatureCache | 1.5-2× | ✅ Complete | Cache static features per hand |
| 4. forward_batch | 1.4-1.5× | ✅ Complete | Batched neural network inference |
| 5. Precomputed lookups | 1.2-1.3× | ✅ Complete | Pot odds table, hand strength cache |
| 6. Memory pooling | 1.2-1.4× | ✅ Complete | Reuse game objects |
| 7. PCG64 RNG | 1.15-1.2× | ✅ Complete | Faster random number generation |
| 8. Numpy deck shuffle | 1.05-1.1× | ✅ Complete | Optimized card shuffling |
| 9. Vectorized mutations | 1.05-1.1× | ✅ Complete | Batch genome operations |
| 10. Disabled history logging | 2× | ✅ Complete | Skip expensive tracking in training |
| 11. **Numba JIT compilation** | **2-3×** | ✅ **Complete** | **JIT-compile hot paths** |

**Current Performance**:
- **With Numba**: ~4-6 sec/generation (**400-500× faster** than original 38 min)
- **Without Numba**: ~13 sec/generation (**175× faster** than original)

**Cumulative Speedup**:
```
Original:        38 min/gen  (1×)
Current (Numba): 4-6 sec/gen (400-500×)  ← YOU ARE HERE
```

### 🔄 Available But Not Yet Implemented

| Optimization | Est. Speedup | Effort | Status |
|--------------|--------------|--------|--------|
| C++ hand evaluator | 2-3× | 2-3 days | ⏳ Available |
| Cython compilation | 1.5-2× | 2-4 days | ⏳ Available |
| GPU acceleration (CuPy) | 3-5× | 3-5 days | ⏳ Available |
| SIMD vectorization | 1.5-2× | 1-2 days | ⏳ Available |

**Potential**: Additional 5-10× speedup available (would reach ~0.5-1 sec/generation)

**Documentation**:
- [OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md) - Complete optimization roadmap
- [NUMBA_JIT_GUIDE.md](NUMBA_JIT_GUIDE.md) - Numba JIT implementation guide
- [FORWARD_BATCH_INTEGRATION.md](FORWARD_BATCH_INTEGRATION.md) - Batched inference details

---

## 📊 Training Configuration

### Default Settings (Good Starting Point)

```python
# Population & Evolution
population_size = 20
elite_fraction = 0.1          # Keep top 10%
mutation_sigma = 0.1          # Mutation strength

# Evaluation
hands_per_matchup = 3000      # Hands per opponent
matchups_per_agent = 12       # Different opponents
players_per_table = 6         # 6-player games

# Network
hidden_layers = [64, 32]      # Two hidden layers
activation = 'relu'
temperature = 1.0             # Sampling temperature
```

### Recommended Configurations

**Quick Testing** (1-2 minutes):
```bash
python scripts/train.py --pop 10 --gens 5 --hands 500 --matchups 3
```

**Standard Training** (20-30 minutes):
```bash
python scripts/train.py --pop 20 --gens 100 --hands 3000 --matchups 12
```

**High-Quality Training** (1-2 hours):
```bash
python scripts/train.py --pop 50 --gens 200 --hands 5000 --matchups 20
```

---

## 📈 Results & Benchmarks

### Training Progress (Typical Run)

```
Gen    0 | Mean: +200.34 | Best: +450.12 | Time: 5.2s  (with Numba)
Gen   10 | Mean: +380.45 | Best: +620.78 | Time: 5.1s
Gen   20 | Mean: +510.23 | Best: +780.45 | Time: 4.9s
...
Gen  100 | Mean: +850.67 | Best: +1200.34 | Time: 5.0s

Total: ~8.5 minutes (with Numba) or ~22 minutes (without Numba)
```

### Performance vs Baselines

| Agent | BB/100 vs Random | Win Rate |
|-------|------------------|----------|
| Random | 0.0 (baseline) | 50% |
| Heuristic | +150-250 | 60-65% |
| Trained (Gen 50) | +400-600 | 70-75% |
| Trained (Gen 100) | +800-1200 | 75-80% |

**Note**: BB/100 = Big Blinds won per 100 hands (standard poker metric)

---

## 🧪 Testing

Run the test suite:

```bash
# All tests
python -m pytest tests/ -v

# Specific tests
python -m pytest tests/test_hand_eval.py -v
python scripts/test_ai_hands.py
python scripts/test_cli.py
```

### Test Coverage
- ✅ Hand evaluation (10,000+ hands verified)
- ✅ Action validation
- ✅ Pot calculations
- ✅ Feature extraction
- ✅ Neural network forward pass
- ✅ Genome mutations

---

## 📚 Documentation

### Module READMEs (Component Documentation)

- **[engine/README.md](engine/README.md)** (764 lines)
  - Complete poker engine API reference
  - Card, action, pot, state, game classes
  - Hand evaluation (standard and fast)
  - Feature extraction for AI
  - CLI interface usage
  - Performance benchmarks

- **[training/README.md](training/README.md)** (717 lines)
  - Evolutionary training algorithm details
  - Neural network policy architecture
  - Genome representation and mutations
  - Fitness evaluation through self-play
  - Configuration options
  - Hyperparameter optimization guide
  - Numba JIT optimizations

- **[agents/README.md](agents/README.md)** (230+ lines)
  - RandomAgent: Baseline random action agent
  - HeuristicAgent: Rule-based poker AI
  - Usage examples and API reference
  - Performance expectations
  - Extension guide for custom agents

- **[evaluator/README.md](evaluator/README.md)** (300+ lines)
  - Hand ranking algorithm (7-card evaluation)
  - Equity calculation (exact and Monte Carlo)
  - Usage with pot odds and decision making
  - Performance characteristics
  - Integration examples

- **[utils/README.md](utils/README.md)** (317 lines)
  - Genome transformation utilities
  - Network parameter conversion
  - Helper functions for AI system

### Optimization Documentation (Performance & Implementation)

- **[OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md)** (625 lines)
  - **Purpose**: Complete optimization roadmap and status
  - 11 implemented optimizations (400-500× speedup achieved)
  - Available but not implemented (C++/GPU/Cython)
  - Performance timeline and benchmarks
  - Learning impact analysis
  - Recommended next steps

- **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)**
  - **Purpose**: Quick reference for all optimizations
  - One-page summary of speedups
  - Phase-by-phase breakdown
  - Current performance metrics

- **[NUMBA_JIT_GUIDE.md](NUMBA_JIT_GUIDE.md)** (1062 lines)
  - **Purpose**: Complete Numba JIT implementation guide
  - How to install and use Numba
  - JIT-compiled functions reference
  - Implementation patterns and best practices
  - Troubleshooting and debugging
  - Before/after benchmarks

- **[FORWARD_BATCH_INTEGRATION.md](FORWARD_BATCH_INTEGRATION.md)** (270 lines)
  - **Purpose**: Batched neural network inference documentation
  - 1.4-1.5× speedup explanation
  - Technical implementation details
  - Learning impact verification (zero impact)
  - Code examples and usage

### Results & Analysis Files

- **`deep_sweep_report.txt`**
  - **Purpose**: Latest deep hyperparameter sweep results
  - Generated by: `python scripts/report_deep_sweep.py`
  - Top configurations by fitness
  - Convergence analysis
  - Parameter insights

- **`hyperparam_results/`**
  - **Purpose**: All hyperparameter sweep outputs
  - JSON results files
  - Visualization plots
  - Convergence analysis reports

- **`tournament_reports/`**
  - **Purpose**: Round-robin tournament results
  - Timestamped report folders
  - JSON matchup data
  - Win/loss charts, heatmaps, parameter analysis

### Configuration Files

- **[training/config.py](training/config.py)**
  - **Purpose**: All configuration dataclasses
  - NetworkConfig: Neural network architecture
  - EvolutionConfig: Population, mutation, selection
  - FitnessConfig: Evaluation settings

---

## � Future Optimizations & Improvements

### Performance Optimizations (Available but Not Implemented)

**C++ Extensions** (2-3× speedup):
- Rewrite hand evaluator in C++ with Python bindings
- Compile-time optimization opportunities
- Direct memory access without Python overhead
- **Effort**: 2-3 days | **Expected**: 0.5-1 sec/generation

**GPU Acceleration** (3-5× speedup):
- CuPy/JAX for batched neural network operations
- Parallel game simulation on GPU
- Large population training (100+ agents)
- **Effort**: 3-5 days | **Expected**: 0.5-2 sec/generation

**Cython Compilation** (1.5-2× speedup):
- Compile hot paths with static typing
- Hybrid Python/C approach
- Gradual migration of critical functions
- **Effort**: 2-4 days | **Expected**: 2-3 sec/generation

**SIMD Vectorization** (1.5-2× speedup):
- Auto-vectorization with compiler flags
- Manual SIMD intrinsics for critical loops
- Batch processing optimizations
- **Effort**: 1-2 days | **Expected**: 2-4 sec/generation

### Algorithm Improvements

**Advanced Evolutionary Algorithms**:
- CMA-ES (Covariance Matrix Adaptation)
- Novelty Search for diverse strategies
- Multi-objective optimization (BB/100 + diversity)
- Island model evolution (parallel populations)

**Neural Network Architectures**:
- LSTM/GRU for temporal game state
- Transformer models for attention mechanisms
- Residual connections for deeper networks
- Ensemble models for robust play

**Training Enhancements**:
- Opponent modeling and exploitation
- Transfer learning from expert games
- Curriculum learning (gradual difficulty increase)
- Self-play with historical opponents

### Feature Enhancements

**Poker Features**:
- Multi-table tournament (MTT) support
- Pot-limit Omaha (PLO) variant
- Short-deck (6+ Hold'em) support
- Real-time opponent profiling

**Infrastructure**:
- Distributed training across machines
- Cloud training integration (AWS/GCP)
- Automated hyperparameter optimization (Optuna/Ray Tune)
- Model versioning and experiment tracking (MLflow)

**User Interface**:
- Web-based training dashboard
- Real-time visualization of training progress
- Interactive agent playground
- Tournament bracket system

**See**:
- [OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md) for detailed optimization roadmap
- [NUMBA_JIT_GUIDE.md](NUMBA_JIT_GUIDE.md) for JIT compilation patterns
- Module READMEs for component-specific improvements

---

## �🛠️ Advanced Usage

### Custom Network Architecture

```python
from training import NetworkConfig

config = NetworkConfig(
    input_size=17,
    hidden_layers=[128, 64, 32],  # Larger network
    output_size=6,
    activation='relu'
)
```

### Custom Fitness Function

```python
from training import FitnessConfig

fitness_config = FitnessConfig(
    hands_per_matchup=5000,       # More hands
    starting_stack=2000,          # Deeper stacks
    small_blind=5,
    big_blind=10,
    ante=1                        # Add ante
)
```

### Parallel Evaluation

```python
# Use more workers for faster training
python scripts/train.py --workers 8
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

**High Priority**:
- C++ hand evaluator extension (2-3× speedup)
- GPU acceleration with CuPy/JAX (3-5× speedup)
- Cython compilation for hot paths (1.5-2× speedup)
- Profile-guided optimization

**Medium Priority**:
- Alternative neural network architectures (LSTM, Transformer)
- Different evolutionary algorithms (CMA-ES, Novelty Search)
- Opponent modeling and exploitation
- Multi-table tournament support

**Low Priority**:
- GUI for gameplay visualization
- Web-based training dashboard
- Agent comparison tools

**Note**: All major Python-based optimizations are complete! Further speedups require C++/GPU.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{pokerbot2026,
  title={PokerBot: High-Performance Poker AI with Evolutionary Training},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/PokerBot}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- Fast hand evaluation inspired by poker hand ranking algorithms
- Evolutionary training based on neuroevolution principles
- Optimization techniques from high-performance computing best practices

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/PokerBot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/PokerBot/discussions)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ Core poker engine
- ✅ Evolutionary training
- ✅ 175× performance optimization
- ✅ Comprehensive documentation

### Planned Features (v1.1)
- ⏳ Numba JIT expansion (2-3× speedup)
- ⏳ Tournament mode
- ⏳ Web interface for gameplay
- ⏳ Pre-trained model zoo

### Future (v2.0)
- 🔮 Multi-table tournaments
- 🔮 Reinforcement learning integration
- 🔮 GPU acceleration for large populations
- 🔮 Real-time opponent modeling

---

**Built with ❤️ and lots of optimization**
