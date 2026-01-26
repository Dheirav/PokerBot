# Baseline Agents

**Reference agents for evaluation and comparison**

---

## Overview

This module provides baseline poker agents that serve as benchmarks for evaluating trained AI performance. These agents implement simple, rule-based strategies without learning.

---

## Module Structure

```
agents/
├── random_agent.py      # Random action selection
├── heuristic_agent.py   # Rule-based strategy
└── README.md            # This file
```

---

## Agents

### 1. Random Agent (`random_agent.py`)

Makes random legal actions with no strategy.

```python
from agents.random_agent import RandomAgent

agent = RandomAgent(seed=42)

# Select random legal action
action = agent.select_action(state, legal_actions)
```

**Characteristics**:
- No skill or strategy
- Baseline for "worst possible" performance
- Expected BB/100: ~0 (break-even)
- Win rate: 50% vs other random agents

**Use Cases**:
- Sanity check for training system
- Minimum performance baseline
- Testing game engine correctness

---

### 2. Heuristic Agent (`heuristic_agent.py`)

Rule-based agent with simple poker strategy.

```python
from agents.heuristic_agent import HeuristicAgent

agent = HeuristicAgent()

# Select action based on hand strength and pot odds
action = agent.select_action(game_state, player_idx)
```

**Strategy**:
- **Strong hands (pairs, high cards)**: Aggressive betting
- **Weak hands**: Fold to large bets
- **Drawing hands**: Call if pot odds favorable
- **Position awareness**: Play tighter in early position

**Characteristics**:
- Simple, interpretable strategy
- Moderate skill level
- Expected BB/100: +150-250 vs random
- Win rate: 60-65% vs random

**Use Cases**:
- Meaningful performance baseline
- Testing if trained agents learn basic strategy
- Human-like opponent for practice

---

## Usage Examples

### Evaluating Trained Agent vs Baseline

```python
from training import PolicyNetwork
from agents import RandomAgent, HeuristicAgent
from engine import PokerGame
import numpy as np

# Load trained agent
trained_agent = PolicyNetwork.from_file('best_genome.npy')

# Create baseline agents
random_agent = RandomAgent(seed=42)
heuristic_agent = HeuristicAgent()

# Play matches
agents = [trained_agent, heuristic_agent, random_agent]
game = PokerGame(
    stacks=[1000, 1000, 1000],
    small_blind=5,
    big_blind=10
)

# ... run game loop ...
```

### Performance Benchmarking

```bash
# Compare trained agent against baselines
python scripts/eval_baseline.py checkpoints/evolution_run/best_genome.npy

# Expected output:
# vs Random:    +800-1200 BB/100  (trained should dominate)
# vs Heuristic: +200-400 BB/100   (trained should win)
```

---

## Performance Expectations

| Agent | BB/100 vs Random | BB/100 vs Heuristic | Skill Level |
|-------|------------------|---------------------|-------------|
| Random | 0 | -200 to -250 | None |
| Heuristic | +150 to +250 | 0 | Basic |
| Trained (Gen 50) | +400 to +600 | +100 to +200 | Intermediate |
| Trained (Gen 100) | +800 to +1200 | +200 to +400 | Advanced |

**Note**: BB/100 = Big Blinds won per 100 hands

---

## Extending Baselines

### Creating Custom Baseline Agent

```python
class MyCustomAgent:
    """Custom baseline agent with specific strategy."""
    
    def select_action(self, state, legal_actions):
        """
        Select action based on custom logic.
        
        Args:
            state: Game state information
            legal_actions: List of legal actions
            
        Returns:
            Action: Selected action
        """
        # Your custom logic here
        pass
```

### Adding New Heuristics

To improve the heuristic agent:
1. Add hand strength calculation refinements
2. Implement better position awareness
3. Add opponent modeling (bet sizing tells)
4. Improve bluffing frequency

---

## Implementation Details

### Random Agent
- Uses numpy RNG for reproducibility
- Uniform distribution over legal actions
- No state tracking or memory

### Heuristic Agent
- Hand strength via simple ranking
- Pot odds calculation
- Position-based adjustments
- No opponent modeling
- No learning or adaptation

---

## Testing

```bash
# Run agent tests
python -m pytest tests/test_agents.py -v

# Test specific agent
python scripts/test_ai_hands.py --agent heuristic
```

---

## Related Scripts

Scripts that use baseline agents:
- **`scripts/eval_baseline.py`**: Evaluate trained AI vs RandomAgent/HeuristicAgent
- **`scripts/match_agents.py`**: Head-to-head matches (use baseline as opponent)
- **`scripts/test_ai_hands.py`**: Test scenarios with different agent types
- **`scripts/train.py`**: Initial population can include baseline agents

---

## Performance Benchmarks

**RandomAgent**:
- BB/100: 0.0 (baseline)
- Win rate vs self: 50%
- Actions/sec: ~100,000+
- Memory: Minimal

**HeuristicAgent**:
- BB/100: +150 to +250 vs RandomAgent
- Win rate vs RandomAgent: 60-65%
- Actions/sec: ~50,000 (feature computation overhead)
- Memory: Low (no neural network)

**Trained AI (100 generations)**:
- BB/100: +800 to +1200 vs RandomAgent
- Win rate vs RandomAgent: 75-80%
- Win rate vs HeuristicAgent: 65-70%

---

## Future Improvements

**High Priority**:
- **GTO (Game Theory Optimal) agent**: Nash equilibrium baseline using solver
- **Pre-trained RL agent**: Strong baseline from reinforcement learning
- **Adaptive heuristic**: Dynamic strategy adjustment based on opponents

**Medium Priority**:
- **Position-specific heuristics**: Different strategies for early/late position
- **Stack-aware strategies**: Adjust play based on stack sizes
- **Exploitative agents**: Detect and exploit opponent patterns
- **Multi-table tournament (MTT) agents**: ICM-aware decision making

**Low Priority**:
- **Cash game specialists**: Deep stack optimized strategies
- **Short-handed agents**: 2-3 player specific tactics
- **Heads-up specialists**: 1v1 optimized play

---

## Troubleshooting

**Issue**: HeuristicAgent losing to RandomAgent  
**Solution**: Check pot odds calculations and hand strength thresholds; may need tuning

**Issue**: Agents too slow  
**Solution**: RandomAgent is instant; HeuristicAgent may be slow if hand evaluation is not using fast version

**Issue**: Trained AI only slightly better than heuristic  
**Solution**: May need more training generations or better hyperparameters

---

## References

- **Poker hand strength evaluation**: 7-card evaluator with lookup tables
- **Pot odds and expected value**: Break-even probability calculations
- **Position-based strategy**: Button, cutoff, blinds adjustments
- **Opponent modeling**: Tracking betting patterns for exploitation

---

**For more information, see main [README.md](../README.md) and [training/README.md](../training/README.md)**
