# Evaluator

**Hand strength and equity calculations for poker decision-making**

---

## Overview

This module provides utilities for calculating hand strength, equity, and hand rankings. These are used by agents to make informed decisions about betting, calling, and folding.

---

## Module Structure

```
evaluator/
├── hand_rank.py         # Hand ranking and strength calculation
├── equity.py            # Win probability and equity calculations
└── README.md            # This file
```

---

## Components

### 1. Hand Ranking (`hand_rank.py`)

Calculate the absolute strength of poker hands.

```python
from evaluator.hand_rank import rank_hand, get_hand_strength

# Rank a 5-card hand
hand = [Card('A', 'spades'), Card('K', 'spades'), 
        Card('Q', 'spades'), Card('J', 'spades'), 
        Card('T', 'spades')]
rank = rank_hand(hand)  # Returns hand type and rank value

# Get normalized strength (0-1)
strength = get_hand_strength(hand)  # 1.0 (royal flush = strongest)
```

**Hand Rankings** (from strongest to weakest):
1. **Royal Flush**: A♠ K♠ Q♠ J♠ T♠
2. **Straight Flush**: 9♥ 8♥ 7♥ 6♥ 5♥
3. **Four of a Kind**: K♣ K♦ K♥ K♠ 9♦
4. **Full House**: Q♠ Q♥ Q♦ 8♣ 8♠
5. **Flush**: A♦ J♦ 9♦ 6♦ 3♦
6. **Straight**: T♣ 9♦ 8♥ 7♠ 6♣
7. **Three of a Kind**: 7♥ 7♦ 7♣ A♠ K♦
8. **Two Pair**: J♠ J♣ 5♥ 5♦ A♣
9. **One Pair**: 9♦ 9♠ A♥ K♣ Q♦
10. **High Card**: A♣ K♠ Q♥ 8♦ 3♣

**Functions**:

#### `rank_hand(cards: List[Card]) -> Tuple[int, List[int]]`
Returns hand type (0-9) and tie-breaking ranks.

```python
hand_type, ranks = rank_hand(cards)
# hand_type: 0 = high card, 9 = royal flush
# ranks: tie-breaking values (e.g., [14, 13] for A-K high)
```

#### `get_hand_strength(cards: List[Card]) -> float`
Returns normalized hand strength (0.0 = worst, 1.0 = best).

```python
strength = get_hand_strength(cards)
# Uses hand type and ranks to compute 0-1 value
```

---

### 2. Equity Calculation (`equity.py`)

Calculate win probability and expected value.

```python
from evaluator.equity import calculate_equity, estimate_equity_monte_carlo

# Calculate exact equity (slow, for small card sets)
hole_cards = [Card('A', 'spades'), Card('K', 'spades')]
community = [Card('Q', 'spades'), Card('J', 'diamonds'), Card('T', 'clubs')]
equity = calculate_equity(hole_cards, community, num_opponents=2)
# Returns: 0.75 (75% chance to win)

# Estimate equity via Monte Carlo simulation (fast)
equity_estimate = estimate_equity_monte_carlo(
    hole_cards, 
    community, 
    num_opponents=2,
    num_simulations=1000
)
# Returns: ~0.75 (approximation)
```

**Functions**:

#### `calculate_equity(hole_cards, community, num_opponents) -> float`
Exact equity calculation (exhaustive enumeration).

**Parameters**:
- `hole_cards`: Your 2 hole cards
- `community`: Community cards (0-5 cards)
- `num_opponents`: Number of opponents

**Returns**: Win probability (0.0 to 1.0)

**Note**: Slow for many unknown cards. Use Monte Carlo for speed.

#### `estimate_equity_monte_carlo(hole_cards, community, num_opponents, num_simulations) -> float`
Fast equity estimation via random sampling.

**Parameters**:
- `hole_cards`: Your 2 hole cards
- `community`: Community cards (0-5 cards)
- `num_opponents`: Number of opponents
- `num_simulations`: Number of random trials (default: 1000)

**Returns**: Estimated win probability

**Accuracy**: ±1-2% with 1000 simulations, ±0.5% with 10000 simulations

---

## Usage Examples

### Pre-flop Hand Strength

```python
from evaluator import get_hand_strength

# Pocket aces
hole_cards = [Card('A', 'spades'), Card('A', 'hearts')]
strength = get_hand_strength(hole_cards + [])  # Empty community
print(f"AA strength: {strength:.2f}")  # ~0.95 (very strong)

# Pocket deuces
hole_cards = [Card('2', 'clubs'), Card('2', 'diamonds')]
strength = get_hand_strength(hole_cards + [])
print(f"22 strength: {strength:.2f}")  # ~0.45 (weak pair)
```

### Post-flop Equity

```python
from evaluator import estimate_equity_monte_carlo

# You have AK on A-7-2 rainbow flop
hole_cards = [Card('A', 'spades'), Card('K', 'spades')]
community = [
    Card('A', 'hearts'),
    Card('7', 'diamonds'),
    Card('2', 'clubs')
]

# Against 1 opponent
equity = estimate_equity_monte_carlo(
    hole_cards, 
    community, 
    num_opponents=1,
    num_simulations=5000
)
print(f"Equity: {equity*100:.1f}%")  # ~85% (strong top pair)
```

### Making Decisions with Pot Odds

```python
from evaluator import estimate_equity_monte_carlo

def should_call(hole_cards, community, pot_size, call_amount, num_opponents):
    """
    Decide if calling is profitable based on pot odds vs equity.
    """
    # Calculate required equity (pot odds)
    pot_odds = call_amount / (pot_size + call_amount)
    
    # Calculate actual equity
    equity = estimate_equity_monte_carlo(
        hole_cards, 
        community, 
        num_opponents=num_opponents,
        num_simulations=2000
    )
    
    # Call if equity > pot odds
    return equity > pot_odds

# Example: Flush draw on flop
# Pot: $100, Bet: $50, You need $50 to call
# Required equity: 50/(100+50) = 33.3%
# Flush draw equity: ~35% (9 outs twice)
should_call_decision = should_call(
    hole_cards=[Card('A', 'spades'), Card('K', 'spades')],
    community=[Card('J', 'spades'), Card('7', 'spades'), Card('2', 'hearts')],
    pot_size=100,
    call_amount=50,
    num_opponents=1
)
print(f"Should call: {should_call_decision}")  # True (35% > 33.3%)
```

---

## Performance

### Hand Ranking
- **Speed**: ~50,000 rankings/sec (pure Python)
- **Speed with fast evaluator**: ~800,000 rankings/sec (13-16× faster)
- Used by: Hand strength features for neural networks

### Equity Calculation
- **Exact**: ~100-1000 hands/sec (depends on unknowns)
- **Monte Carlo (1000 sims)**: ~500-1000 hands/sec
- **Monte Carlo (10000 sims)**: ~50-100 hands/sec

**Recommendation**: Use Monte Carlo with 1000-2000 simulations for training (fast, accurate enough)

---

## Integration with Training

The evaluator is used by the training system to:

1. **Feature extraction**: `get_hand_strength()` provides normalized hand strength as neural network input
2. **Heuristic baseline**: Heuristic agent uses hand strength and equity for decisions
3. **Analysis**: Evaluate trained agent decisions vs optimal equity-based play

```python
# In engine/features.py
from evaluator import get_hand_strength

def get_state_vector(game, player_idx):
    """Extract features for neural network."""
    hand_strength = get_hand_strength(
        player.hole_cards + game.state.community_cards
    )
    # ... other features ...
    return features
```

---

## Testing

```bash
# Test hand ranking
python -m pytest tests/test_evaluator.py -v -k "rank"

# Test equity calculation
python -m pytest tests/test_evaluator.py -v -k "equity"

# Benchmark performance
python scripts/benchmark_evaluator.py
```

---

## Optimization Notes

### Current Optimizations
- ✅ Fast hand evaluator (13-16× speedup)
- ✅ Precomputed hand rankings
- ✅ Efficient card representation
- ✅ Vectorized operations where possible

### Potential Future Optimizations

**Python-based** (Easier to implement):
- [ ] Monte Carlo equity with NumPy vectorization (2-3× faster)
  - Batch multiple simulations together
  - Vectorize card dealing and evaluation
  - **Effort**: 1-2 days

- [ ] Numba JIT for equity calculation (1.5-2× faster)
  - JIT-compile simulation loop
  - Inline hand evaluator calls
  - **Effort**: 1 day

- [ ] Precomputed equity tables (instant lookup)
  - Store common preflop/flop scenarios
  - ~200MB lookup table
  - **Effort**: 2-3 days

**C++/GPU-based** (Maximum performance):
- [ ] C++ extension for equity (5-10× faster)
  - Compile-time optimizations
  - SIMD vectorization
  - **Effort**: 3-4 days

- [ ] GPU-accelerated equity (10-50× faster)
  - CUDA/OpenCL for large batches
  - Parallel simulation across thousands of threads
  - **Effort**: 5-7 days

- [ ] Approximate equity with neural networks (100× faster)
  - Train NN to predict equity
  - Trade accuracy for speed
  - **Effort**: 1-2 weeks

---

## Related Scripts

Scripts that use this module:
- **`scripts/eval_baseline.py`**: Uses hand ranking for agent evaluation
- **`scripts/match_agents.py`**: Hand evaluation in game simulation
- **`engine/game.py`**: Showdown winner determination
- **`engine/features.py`**: Hand strength feature computation
- **`scripts/test_ai_hands.py`**: Scenario testing with hand evaluation

---

## Algorithm Details

### Hand Ranking Algorithm
1. Check for flush (all same suit)
2. Check for straight (consecutive ranks)
3. Count rank frequencies (pairs, trips, quads)
4. Classify hand based on patterns
5. Generate tie-breaking ranks

**Complexity**: O(1) - constant time (5 cards always)

### Monte Carlo Equity Algorithm
1. Deal random opponent hole cards
2. Deal random remaining community cards
3. Evaluate all hands at showdown
4. Repeat N times
5. Return win_count / N

**Complexity**: O(N × M) where N = simulations, M = hand evaluations

---

## References

- Poker hand ranking rules (official)
- Monte Carlo simulation techniques
- Expected value and pot odds theory
- Fast hand evaluation algorithms (lookup tables)
