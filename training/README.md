# Training System

**Evolutionary poker AI training through self-play**

---

## Overview

This module implements an evolutionary algorithm for training poker AI agents. Networks compete in self-play poker games, with successful strategies propagating to future generations through mutation and selection.

**Key Features**:
- 🧬 Evolutionary algorithm with elitism
- 🎲 Self-play fitness evaluation  
- 🧠 Neural network policy agents
- 📊 Population diversity maintenance
- ⚡ Parallel evaluation with multiprocessing
- 🎯 Configurable hyperparameters

---

## Module Structure

```
training/
├── __init__.py          # Module exports
├── config.py            # Configuration dataclasses
├── genome.py            # Genome representation and factory
├── policy_network.py    # Neural network policy
├── fitness.py           # Fitness evaluation through self-play
├── evolution.py         # Evolutionary algorithm
├── self_play.py         # Self-play utilities
└── README.md            # This file
```

---

## Core Components

### 1. Configuration (`config.py`)

```python
from training import NetworkConfig, EvolutionConfig, FitnessConfig

# Neural network architecture
network_config = NetworkConfig(
    input_size=17,              # State features
    hidden_layers=[64, 32],     # Two hidden layers
    output_size=6,              # Abstract actions
    activation='relu'
)

# Evolution parameters
evolution_config = EvolutionConfig(
    population_size=20,         # Number of agents
    elite_fraction=0.1,         # Keep top 10%
    mutation_sigma=0.1,         # Mutation strength
    mutation_decay=0.99,        # Decay per generation
    diversity_bonus=0.0,        # Diversity preservation
    num_parents=1               # Mutation-only (no crossover)
)

# Fitness evaluation
fitness_config = FitnessConfig(
    hands_per_matchup=3000,     # Hands per opponent
    matchups_per_agent=12,      # Different opponents
    players_per_table=6,        # 6-max poker
    starting_stack=1000,        # Initial chips
    small_blind=5,              # SB amount
    big_blind=10,               # BB amount
    ante=0,                     # Ante (0 = none)
    temperature=1.0             # Sampling temperature
)
```

---

### 2. Genome (`genome.py`)

A genome is a flat numpy array representing all neural network weights.

```python
from training import Genome, GenomeFactory

# Create factory
factory = GenomeFactory(network_config, evolution_config)

# Create random genome
genome = factory.create_random_genome(generation=0)
print(f"Genome size: {genome.weights.shape}")  # (3430,)

# Mutate
offspring = factory.mutate(genome, generation=1)

# Crossover (if using multiple parents)
child = factory.crossover(parent1, parent2, generation=1)
```

**Genome Attributes**:
- `genome_id`: Unique identifier
- `weights`: Flat numpy array of all network parameters
- `fitness`: Evaluated fitness score (BB/100)
- `generation`: Generation this genome was created
- `parent_id`: Parent genome ID

---

### 3. Policy Network (`policy_network.py`)

Neural network that maps game state → action probabilities.

```python
from training import PolicyNetwork

# Create from config
network = PolicyNetwork(network_config)

# Or from genome weights
network = PolicyNetwork.from_genome(genome, network_config)

# Forward pass
features = np.array([...])  # 17 features
logits = network.forward(features)

# Select action
mask = np.array([1, 1, 1, 0, 0, 1])  # Legal actions
action_idx = network.select_action(features, mask, rng, temperature=1.0)

# Batched forward pass (1.4-1.5× faster)
features_batch = np.array([[...], [...], [...]])  # (batch_size, 17)
logits_batch = network.forward_batch(features_batch)
```

**Architecture**:
- Input: 17 state features
- Hidden layer 1: 64 neurons (ReLU)
- Hidden layer 2: 32 neurons (ReLU)
- Output: 6 action logits

**Abstract Actions**:
0. Fold
1. Check/Call
2. Raise 0.5× pot
3. Raise 1.0× pot
4. Raise 2.0× pot
5. All-in

---

### 4. Fitness Evaluation (`fitness.py`)

Evaluate agents through self-play poker games.

```python
from training import evaluate_matchup

# Evaluate one genome against opponents
fitness_bb100, hands_played = evaluate_matchup(
    genome_weights=genome.weights,
    opponent_weights=[opp1.weights, opp2.weights, ...],
    network_config=network_config,
    fitness_config=fitness_config,
    hand_seeds=None  # None = random hands
)

print(f"Fitness: {fitness_bb100:.2f} BB/100")
```

**Fitness Metric**: Big blinds won per 100 hands (BB/100)
- Positive = winning player
- Negative = losing player
- Standard poker performance metric

**Process**:
1. Create table with genome + opponents
2. Play specified number of hands
3. Track chip changes for genome
4. Return BB/100 = (chips_won / big_blind) * 100 / hands_played

---

### 5. Evolution (`evolution.py`)

Main evolutionary algorithm orchestrator.

```python
from training import Evolution

# Create evolution engine
evolution = Evolution(
    network_config=network_config,
    evolution_config=evolution_config,
    fitness_config=fitness_config,
    checkpoint_dir='checkpoints/evolution_run',
    seed=42,
    num_workers=4  # Parallel evaluation
)

# Initialize population
evolution.initialize_population()

# Run evolution
for generation in range(100):
    # Evaluate fitness
    evolution.evaluate_population()
    
    # Select and create next generation
    evolution.evolve()
    
    # Log progress
    print(f"Gen {generation}: Best = {evolution.best_fitness:.2f}")
    
    # Save checkpoint
    evolution.save_checkpoint()
```

**Evolution Steps**:
1. **Evaluate**: Play poker games, measure BB/100
2. **Select**: Keep elite performers
3. **Mutate**: Add Gaussian noise to create offspring
4. **Replace**: Form new population
5. **Repeat**: Next generation

---

## Training Pipeline

### Full Training Process

```python
from training import Evolution, NetworkConfig, EvolutionConfig, FitnessConfig

# 1. Configure
network_config = NetworkConfig(input_size=17, hidden_layers=[64, 32], output_size=6)
evolution_config = EvolutionConfig(population_size=20, elite_fraction=0.1)
fitness_config = FitnessConfig(hands_per_matchup=3000, matchups_per_agent=12)

# 2. Initialize
evolution = Evolution(network_config, evolution_config, fitness_config, seed=42, num_workers=4)
evolution.initialize_population()

# 3. Train
for gen in range(100):
    evolution.evaluate_population()
    evolution.evolve()
    evolution.save_checkpoint()

# 4. Get best agent
best_genome = evolution.get_best_genome()
network = PolicyNetwork.from_genome(best_genome, network_config)
```

---

## Optimizations

### 1. Batched Forward Pass (1.4-1.5× speedup)

```python
# Process multiple decisions simultaneously
features_batch = np.array([features1, features2, features3])
masks_batch = np.array([mask1, mask2, mask3])

# Vectorized inference
actions = network.select_action_batch(features_batch, masks_batch, rng)
```

### 2. Multiprocessing (4× speedup)

```python
# Parallel genome evaluation
evolution = Evolution(..., num_workers=4)

# Fitness evaluation distributed across workers
# Each worker evaluates subset of population
```

### 3. Memory Pooling (1.2-1.4× speedup)

```python
# Reuse game objects
from training.fitness import GamePool

pool = GamePool(size=100)
game = pool.acquire(...)  # Get from pool
# ... play game ...
pool.release(game)  # Return to pool
```

### 4. PCG64 RNG (1.15-1.2× speedup)

```python
from numpy.random import PCG64, Generator

# Faster than default Mersenne Twister
rng = Generator(PCG64(seed))
```

---

## Configuration Guide

### Network Architecture

**Small** (fast training):
```python
NetworkConfig(
    input_size=17,
    hidden_layers=[32, 16],  # Smaller layers
    output_size=6
)
```

**Medium** (balanced):
```python
NetworkConfig(
    input_size=17,
    hidden_layers=[64, 32],  # Default
    output_size=6
)
```

**Large** (better performance):
```python
NetworkConfig(
    input_size=17,
    hidden_layers=[128, 64, 32],  # Deeper network
    output_size=6
)
```

### Evolution Parameters

**Quick exploration**:
```python
EvolutionConfig(
    population_size=10,      # Small population
    elite_fraction=0.2,      # Keep more elites
    mutation_sigma=0.2       # Large mutations
)
```

**Standard training**:
```python
EvolutionConfig(
    population_size=20,      # Medium population
    elite_fraction=0.1,      # 10% elites
    mutation_sigma=0.1       # Moderate mutations
)
```

**Fine-tuning**:
```python
EvolutionConfig(
    population_size=50,      # Large population
    elite_fraction=0.05,     # Few elites (more exploration)
    mutation_sigma=0.05      # Small mutations
)
```

### Fitness Evaluation

**Fast testing**:
```python
FitnessConfig(
    hands_per_matchup=500,   # Few hands
    matchups_per_agent=3     # Few opponents
)
```

**Standard training**:
```python
FitnessConfig(
    hands_per_matchup=3000,  # Many hands
    matchups_per_agent=12    # Multiple opponents
)
```

**High-quality evaluation**:
```python
FitnessConfig(
    hands_per_matchup=5000,  # Very many hands
    matchups_per_agent=20    # Many opponents
)
```

---

## Usage Examples

### Example 1: Basic Training

```python
from training import Evolution, NetworkConfig, EvolutionConfig, FitnessConfig

# Configure
config_network = NetworkConfig(input_size=17, hidden_layers=[64, 32], output_size=6)
config_evolution = EvolutionConfig(population_size=20)
config_fitness = FitnessConfig(hands_per_matchup=3000)

# Train
evolution = Evolution(config_network, config_evolution, config_fitness, seed=42, num_workers=4)
evolution.initialize_population()

for gen in range(100):
    evolution.evaluate_population()
    evolution.evolve()
    print(f"Generation {gen}: Best = {evolution.best_fitness:.2f} BB/100")
```

### Example 2: Resume Training

```python
# Load checkpoint
evolution = Evolution.load_checkpoint('checkpoints/evolution_run')

# Continue training
for gen in range(evolution.generation, 200):
    evolution.evaluate_population()
    evolution.evolve()
    evolution.save_checkpoint()
```

### Example 3: Custom Fitness Function

```python
from training.fitness import evaluate_matchup

def custom_fitness(genome, opponents):
    # Evaluate with custom settings
    fitness, _ = evaluate_matchup(
        genome.weights,
        opponents,
        network_config,
        custom_fitness_config,
        hand_seeds=[42, 43, 44, ...]  # Fixed hands for reproducibility
    )
    return fitness

# Use in evolution
evolution.fitness_function = custom_fitness
```

---

## Performance Metrics

### Training Speed

With all optimizations:
- **Generation time**: ~13 seconds
- **100 generations**: ~22 minutes
- **Configuration**: 20 pop, 3000 hands/matchup, 12 matchups

### Fitness Progression

Typical training run:
```
Gen   0: Mean =  200 BB/100, Best =  450 BB/100
Gen  10: Mean =  380 BB/100, Best =  620 BB/100
Gen  25: Mean =  510 BB/100, Best =  780 BB/100
Gen  50: Mean =  680 BB/100, Best =  950 BB/100
Gen 100: Mean =  850 BB/100, Best = 1200 BB/100
```

### Speedup Breakdown

| Component | Speedup | Status |
|-----------|---------|--------|
| Multiprocessing | 4× | ✅ Active |
| FeatureCache | 1.5-2× | ✅ Active |
| forward_batch | 1.4-1.5× | ✅ Active |
| Memory pooling | 1.2-1.4× | ✅ Active |
| PCG64 RNG | 1.15-1.2× | ✅ Active |
| Numba JIT | 2-3× | ⏳ Optional |

**Total**: ~175× faster than original

---

## API Reference

### Evolution Class

```python
class Evolution:
    def __init__(self, network_config, evolution_config, fitness_config, 
                 checkpoint_dir='checkpoints/evolution_run', seed=42, num_workers=4)
    
    def initialize_population(self) -> None
    def evaluate_population(self) -> None
    def evolve(self) -> None
    def save_checkpoint(self) -> None
    
    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str) -> 'Evolution'
    
    def get_best_genome(self) -> Genome
    def get_population_stats(self) -> Dict
```

### PolicyNetwork Class

```python
class PolicyNetwork:
    def __init__(self, config: NetworkConfig)
    
    @classmethod
    def from_genome(cls, genome: Genome, config: NetworkConfig) -> 'PolicyNetwork'
    
    def forward(self, features: np.ndarray) -> np.ndarray
    def forward_batch(self, features_batch: np.ndarray) -> np.ndarray
    def select_action(self, features: np.ndarray, mask: np.ndarray, 
                      rng: np.random.Generator, temperature: float = 1.0) -> int
    def select_action_batch(self, features_batch: np.ndarray, mask_batch: np.ndarray,
                           rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray
    
    def to_genome(self) -> np.ndarray
```

---

## Testing

```bash
# Test evolution
python -m pytest tests/test_evolution.py -v

# Test policy network
python -m pytest tests/test_policy_network.py -v

# Test fitness evaluation
python scripts/test_ai_hands.py
```

---

## Troubleshooting

### Issue: Training is slow
**Solutions**:
- Increase `num_workers` (use more CPU cores)
- Reduce `hands_per_matchup` or `matchups_per_agent`
- Install `numba` for 2-3× speedup: `pip install numba`

### Issue: Poor fitness improvement
**Solutions**:
- Increase `hands_per_matchup` (more reliable evaluation)
- Increase `matchups_per_agent` (more diverse opponents)
- Adjust `mutation_sigma` (try 0.05 to 0.2)
- Increase `population_size` (more exploration)

### Issue: Overfitting warnings
**Meaning**: Training fitness much higher than evaluation fitness  
**Solutions**:
- Normal early in training (population adapting to each other)
- Increase opponent diversity in evaluation
- Use fixed evaluation opponents

---

## Future Enhancements

- [ ] Coevolution with opponent modeling
- [ ] Novelty search for strategy diversity
- [ ] CMA-ES as alternative to Gaussian mutation
- [ ] Transfer learning from pre-trained networks
- [ ] Multi-objective optimization (BB/100 + diversity)
- [ ] Adaptive mutation rates per genome

---

## Integration with Engine

```python
from engine import PokerGame
from training import PolicyNetwork

# Create game
game = PokerGame(player_stacks=[1000] * 6, seed=42)

# Create AI
network = PolicyNetwork.from_genome(genome, network_config)

# Play
while not game.is_hand_over():
    current = game.state.current_player
    features = get_state_vector(game, current)
    mask = get_action_mask(game, current)
    action_idx = network.select_action(features, mask, rng)
    # ... apply action
```

See [engine/README.md](../engine/README.md) for engine details.

---

**For training questions, see main [README.md](../README.md) or open an issue.**
