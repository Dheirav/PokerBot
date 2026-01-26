# Utilities

**Helper functions and transformations for the PokerBot system**

---

## Overview

This module provides utility functions for genome manipulation, data transformations, and helper operations used across the poker AI training system.

---

## Module Structure

```
utils/
├── genome_transform.py    # Genome manipulation utilities
└── README.md              # This file
```

---

## Components

### Genome Transform (`genome_transform.py`)

Utilities for converting between genome representations and extracting network parameters.

```python
from utils.genome_transform import (
    genome_to_network_params,
    network_params_to_genome,
    get_genome_size
)

# Calculate required genome size
network_config = NetworkConfig(input_size=17, hidden_layers=[64, 32], output_size=6)
genome_size = get_genome_size(network_config)
print(f"Genome needs {genome_size} parameters")  # 3430

# Convert flat genome to network parameters
genome_weights = np.random.randn(genome_size)
weights, biases = genome_to_network_params(genome_weights, network_config)

# weights = [W1, W2, W3]  # Layer weight matrices
# biases = [b1, b2, b3]   # Layer bias vectors

# Convert network parameters back to genome
genome_flat = network_params_to_genome(weights, biases)
```

**Functions**:

#### `get_genome_size(config: NetworkConfig) -> int`
Calculate the total number of parameters needed for a network.

**Parameters**:
- `config`: Network configuration

**Returns**: Total parameter count

**Example**:
```python
config = NetworkConfig(input_size=17, hidden_layers=[64, 32], output_size=6)
size = get_genome_size(config)
# 17*64 + 64 + 64*32 + 32 + 32*6 + 6 = 3430
```

---

#### `genome_to_network_params(genome: np.ndarray, config: NetworkConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]`
Convert flat genome array to structured network parameters.

**Parameters**:
- `genome`: Flat numpy array of all weights
- `config`: Network configuration

**Returns**: 
- `weights`: List of weight matrices [W1, W2, ..., WL]
- `biases`: List of bias vectors [b1, b2, ..., bL]

**Example**:
```python
genome = np.random.randn(3430)
weights, biases = genome_to_network_params(genome, config)

# weights[0].shape = (17, 64)   # Input to hidden1
# biases[0].shape = (64,)       # Hidden1 biases
# weights[1].shape = (64, 32)   # Hidden1 to hidden2
# biases[1].shape = (32,)       # Hidden2 biases
# weights[2].shape = (32, 6)    # Hidden2 to output
# biases[2].shape = (6,)        # Output biases
```

---

#### `network_params_to_genome(weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray`
Convert structured network parameters to flat genome array.

**Parameters**:
- `weights`: List of weight matrices
- `biases`: List of bias vectors

**Returns**: Flat numpy array containing all parameters

**Example**:
```python
# Create network parameters
W1 = np.random.randn(17, 64)
b1 = np.random.randn(64)
W2 = np.random.randn(64, 32)
b2 = np.random.randn(32)
W3 = np.random.randn(32, 6)
b3 = np.random.randn(6)

weights = [W1, W2, W3]
biases = [b1, b2, b3]

# Flatten to genome
genome = network_params_to_genome(weights, biases)
print(genome.shape)  # (3430,)
```

---

## Usage Examples

### Example 1: Initialize Random Genome

```python
from utils.genome_transform import get_genome_size
import numpy as np

# Determine size
config = NetworkConfig(input_size=17, hidden_layers=[64, 32], output_size=6)
size = get_genome_size(config)

# Create random genome
genome = np.random.randn(size) * 0.1  # Small initialization
```

### Example 2: Load Genome into Network

```python
from utils.genome_transform import genome_to_network_params
from training import PolicyNetwork

# Load genome
genome = np.load('best_genome.npy')

# Convert to network parameters
weights, biases = genome_to_network_params(genome, config)

# Create network
network = PolicyNetwork(config)
network.weights = weights
network.biases = biases

# Or use built-in method
network = PolicyNetwork.from_genome_array(genome, config)
```

### Example 3: Extract Genome from Network

```python
from utils.genome_transform import network_params_to_genome

# Train network somehow
network = PolicyNetwork(config)
# ... training ...

# Extract genome
genome = network_params_to_genome(network.weights, network.biases)

# Save for later
np.save('trained_genome.npy', genome)
```

### Example 4: Genome Surgery (Transfer Learning)

```python
from utils.genome_transform import genome_to_network_params, network_params_to_genome

# Load pre-trained genome
pretrained = np.load('pretrained_genome.npy')
weights, biases = genome_to_network_params(pretrained, old_config)

# Modify architecture (add layer)
new_config = NetworkConfig(input_size=17, hidden_layers=[64, 64, 32], output_size=6)

# Transfer first two layers, initialize new layer randomly
new_weights = [
    weights[0],                           # Keep layer 1
    np.random.randn(64, 64) * 0.1,       # New layer 2
    weights[1],                           # Keep old layer 2 as layer 3
    weights[2]                            # Keep output layer
]
new_biases = [
    biases[0],
    np.random.randn(64) * 0.1,
    biases[1],
    biases[2]
]

# Create new genome
new_genome = network_params_to_genome(new_weights, new_biases)
```

---

## Genome Format

The genome is a flat numpy array containing all network parameters in order:

```
[W1_flat, b1, W2_flat, b2, ..., WL_flat, bL]
```

Where:
- `W1_flat` = First layer weights flattened (row-major)
- `b1` = First layer biases
- And so on for each layer

### Example Layout

For network `[17, 64, 32, 6]`:

```
Position    | Parameters            | Count
------------|-----------------------|-------
0-1087      | W1 (17×64 flattened)  | 1088
1088-1151   | b1 (64 biases)        | 64
1152-3199   | W2 (64×32 flattened)  | 2048
3200-3231   | b2 (32 biases)        | 32
3232-3423   | W3 (32×6 flattened)   | 192
3424-3429   | b3 (6 biases)         | 6
------------|-----------------------|-------
Total: 3430 parameters
```

---

## API Reference

### `get_genome_size(config: NetworkConfig) -> int`
**Description**: Calculate total parameter count  
**Time Complexity**: O(L) where L = number of layers  
**Space Complexity**: O(1)

### `genome_to_network_params(genome: np.ndarray, config: NetworkConfig) -> Tuple[List, List]`
**Description**: Unflatten genome into weight matrices and bias vectors  
**Time Complexity**: O(P) where P = parameter count  
**Space Complexity**: O(P) for storing structured parameters  
**Raises**: `ValueError` if genome size doesn't match config

### `network_params_to_genome(weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray`
**Description**: Flatten network parameters into single array  
**Time Complexity**: O(P) where P = parameter count  
**Space Complexity**: O(P) for flat array  

---

## Testing

```python
import numpy as np
from utils.genome_transform import *

# Test round-trip conversion
config = NetworkConfig(input_size=17, hidden_layers=[64, 32], output_size=6)
original_genome = np.random.randn(get_genome_size(config))

# Convert to params and back
weights, biases = genome_to_network_params(original_genome, config)
reconstructed_genome = network_params_to_genome(weights, biases)

# Should be identical
assert np.allclose(original_genome, reconstructed_genome)
```

---

## Performance Notes

- All operations are O(P) where P is parameter count (~3400)
- Very fast: <1ms for typical genomes
- Zero-copy where possible with numpy views
- Memory efficient: stores only one copy of parameters

---

## Related Scripts

Scripts that use this module:
- **`scripts/train.py`**: Main training script (genome creation and evolution)
- **`scripts/analyze_top_agents.py`**: Agent analysis (genome parameter inspection)
- **`scripts/visualize_agent_behavior.py`**: Behavior visualization (weight distributions)
- **`scripts/match_agents.py`**: Agent matchups (genome loading and comparison)
- **`scripts/round_robin_agents_config.py`**: Tournaments (multi-agent genome management)

---

## Future Utilities

**High Priority**:
- [ ] Genome compression/decompression (save storage space for checkpoints)
- [ ] Genome distance metrics (L2, cosine similarity for diversity analysis)
- [ ] Parameter statistics per layer (mean, std, min, max for debugging)

**Medium Priority**:
- [ ] Genome interpolation for advanced crossover strategies
- [ ] Visualization helpers (plot weight distributions, gradient flows)
- [ ] Gradient extraction for hybrid evolutionary + gradient methods
- [ ] Weight pruning and quantization (model compression)

**Low Priority**:
- [ ] Serialization utilities (JSON, MessagePack for interchange)
- [ ] Genome versioning (track parameter evolution over time)
- [ ] Automatic architecture inference from genome size

---

## Integration

Used extensively by:
- **`training/genome.py`**: Genome creation and mutation operations
- **`training/policy_network.py`**: Network initialization from genome
- **`training/evolution.py`**: Population management and breeding
- **`scripts/analyze_top_agents.py`**: Elite agent analysis tools
- **`scripts/train.py`**: Main training loop

---

## Troubleshooting

**Issue**: Shape mismatch when loading genome  
**Solution**: Verify `network_config` matches the genome's original architecture

**Issue**: Memory usage too high with large populations  
**Solution**: Use numpy views instead of copies; consider genome compression

**Issue**: Slow genome operations  
**Solution**: Already optimized at O(P); if still slow, check for unnecessary copies

---

**For more information, see main [README.md](../README.md)**
