# Poker Engine

**Complete Texas Hold'em poker game implementation**

---

## Overview

This module implements a full-featured Texas Hold'em poker engine with support for 2-10 players, multiple betting rounds, side pots, showdowns, and comprehensive hand evaluation.

**Key Features**:
- ♠️ Complete Texas Hold'em rules
- 🎲 Fast hand evaluation (13-16× speedup)
- 💰 Multi-way pot management with side pots
- 🔄 All betting actions (fold, check, call, raise, all-in)
- 🎯 Position tracking (blinds, button, action order)
- 📊 Hand history tracking
- ⚡ Optimized for AI training (10M+ hands/hour)

---

## Module Structure

```
engine/
├── __init__.py           # Module exports
├── cards.py              # Card and Deck classes
├── actions.py            # Player actions (Fold, Call, Raise, etc.)
├── pot.py                # Pot management and side pots
├── state.py              # Game state representation
├── game.py               # Main game controller
├── showdown.py           # Showdown logic and winner determination
├── hand_eval.py          # Standard hand evaluator
├── hand_eval_fast.py     # Optimized hand evaluator (13-16× faster)
├── features.py           # State feature extraction for AI
├── history.py            # Hand history tracking
└── cli.py                # Command-line interface
```

---

## Core Components

### 1. Cards (`cards.py`)

Represents cards and deck operations.

```python
from engine import Card, Deck

# Create cards
card = Card('A', 'spades')      # Ace of spades
print(card)                      # A♠
print(card.rank_value())         # 14

# Card from string
card = Card.from_string('Kh')   # King of hearts

# Deck operations
deck = Deck(seed=42)             # Reproducible shuffle
deck.shuffle()                   # Shuffle deck
card = deck.deal()               # Deal one card
hand = deck.deal_n(2)            # Deal multiple cards
```

**Card Ranks**: `2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A`  
**Card Suits**: `hearts (♥), diamonds (♦), clubs (♣), spades (♠)`

---

### 2. Actions (`actions.py`)

Player actions during a hand.

```python
from engine import Action, ActionType

# Action types
action = Action(ActionType.FOLD)
action = Action(ActionType.CHECK)
action = Action(ActionType.CALL, amount=50)
action = Action(ActionType.RAISE, amount=100)
action = Action(ActionType.ALL_IN, amount=500)

# From string
action = Action.from_string('raise 100')
action = Action.from_string('call')
action = Action.from_string('fold')
```

**Action Types**:
- `FOLD` - Surrender hand
- `CHECK` - Pass action (no bet to call)
- `CALL` - Match current bet
- `RAISE` - Increase bet size
- `ALL_IN` - Bet all remaining chips

---

### 3. Game State (`state.py`)

Represents the current state of a poker hand.

```python
from engine import GameState

# Access state information
state = game.state

# Player information
player_idx = state.current_player     # Whose turn
stack = state.stacks[player_idx]      # Player's chips
active = state.active[player_idx]     # Still in hand?
bet = state.current_bets[player_idx]  # Current bet amount

# Street information
street = state.street                 # 'preflop', 'flop', 'turn', 'river'
community = state.community_cards     # Board cards
pot_size = state.pot.total()          # Total pot

# Hole cards
hole_cards = state.hole_cards[player_idx]  # Player's hand

# Legal actions
can_check = state.can_check(player_idx)
can_raise = state.can_raise(player_idx)
min_raise = state.min_raise_amount()
```

---

### 4. Pot Management (`pot.py`)

Handles main pot and side pots.

```python
from engine import Pot

# Create pot
pot = Pot(num_players=6)

# Add bets
pot.add_bet(player_idx=0, amount=100)
pot.add_bet(player_idx=1, amount=100)
pot.add_bet(player_idx=2, amount=50)  # All-in

# Close betting round
pot.close_betting_round()

# Get pot info
total = pot.total()                   # Total chips
main = pot.get_main_pot()             # Main pot size
side_pots = pot.get_side_pots()       # List of side pots

# Award pot
pot.award_to_player(winner_idx=0)     # Winner takes pot
```

**Side Pot Example**:
- Player A: $100 (all-in)
- Player B: $200
- Player C: $200

Creates:
- Main pot: $300 (A, B, C eligible)
- Side pot: $200 (B, C eligible)

---

### 5. Game Controller (`game.py`)

Main game orchestrator.

```python
from engine import PokerGame

# Create game
game = PokerGame(
    player_stacks=[1000, 1000, 1000, 1000, 1000, 1000],  # 6 players
    small_blind=5,
    big_blind=10,
    ante=0,
    seed=42
)

# Start new hand
game.start_new_hand()

# Play actions
while not game.is_hand_over():
    player_idx = game.state.current_player
    
    # Get legal actions
    legal_actions = game.get_legal_actions(player_idx)
    
    # Player decides action
    action = decide_action(game.state, legal_actions)
    
    # Apply action
    game.apply_action(action)

# Hand over - showdown
winners, amounts = game.determine_winners()
game.award_pots(winners, amounts)
```

---

### 6. Hand Evaluation (`hand_eval_fast.py`)

Ultra-fast hand evaluation using lookup tables.

```python
from engine import evaluate_hand

# Evaluate 7-card hand
hole_cards = [Card('A', 'spades'), Card('K', 'spades')]
community = [
    Card('Q', 'spades'),
    Card('J', 'spades'),
    Card('T', 'spades'),
    Card('5', 'hearts'),
    Card('2', 'clubs')
]

all_cards = hole_cards + community
rank = evaluate_hand(all_cards)

# Lower rank = better hand
# 1 = Royal flush
# 7462 = 7-high
```

**Hand Rankings** (lower is better):
1. Royal Flush: 1
2. Straight Flush: 2-10
3. Four of a Kind: 11-166
4. Full House: 167-322
5. Flush: 323-1599
6. Straight: 1600-1609
7. Three of a Kind: 1610-2467
8. Two Pair: 2468-3325
9. One Pair: 3326-6185
10. High Card: 6186-7462

**Performance**: 13-16× faster than standard evaluator

---

### 7. Feature Extraction (`features.py`)

Extract state features for AI agents.

```python
from engine import get_state_vector, FeatureCache

# Create feature cache (1.5-2× speedup)
cache = FeatureCache()

# Extract features
features = get_state_vector(game, player_idx, cache)

# Features (17-dimensional vector):
# [0]     pot odds
# [1]     stack to pot ratio
# [2-7]   position encoding (one-hot)
# [8-11]  street encoding (one-hot)
# [12]    num active players
# [13]    hand strength (vs random)
# [14]    hand potential
# [15]    aggression factor
# [16]    commitment level
```

---

## Game Flow

### Complete Hand Example

```python
from engine import PokerGame

# Initialize
game = PokerGame(
    player_stacks=[1000] * 6,
    small_blind=5,
    big_blind=10,
    seed=42
)

# Start hand
game.start_new_hand()

# PREFLOP
# Player 0 (UTG): calls 10
game.apply_action(Action(ActionType.CALL, 10))

# Player 1: raises to 30
game.apply_action(Action(ActionType.RAISE, 30))

# Player 2: folds
game.apply_action(Action(ActionType.FOLD))

# ... more actions ...

# Betting round completes, advance to flop
# game automatically deals flop

# FLOP
# Players act...

# TURN
# Players act...

# RIVER
# Players act...

# SHOWDOWN
if not game.is_hand_over():
    winners, amounts = game.determine_winners()
    game.award_pots(winners, amounts)
```

---

## Betting Rules

### Minimum Raise

The minimum raise is the size of the previous raise:

```python
# Example: 
# BB posts 10
# Player A raises to 30 (raise of 20)
# Player B must raise to at least 50 (raise of 20)

min_raise = game.state.min_raise_amount()
max_raise = game.state.stacks[player_idx]  # Up to all chips
```

### All-In Scenarios

When a player goes all-in for less than a full bet:

```python
# Example:
# Bet is 100
# Player has 60 chips
# Player goes all-in for 60

action = Action(ActionType.ALL_IN, 60)
game.apply_action(action)

# Other players can still:
# - Call 60 (match the all-in)
# - Raise to 100+ (if they want to raise)
```

### Pot Odds Calculation

```python
amount_to_call = game.state.current_bet - game.state.current_bets[player_idx]
pot_after_call = game.state.pot.total() + amount_to_call

pot_odds = amount_to_call / pot_after_call if pot_after_call > 0 else 0
```

---

## Hand History

Track all actions during a hand:

```python
from engine import HandHistory

# Create history tracker
history = HandHistory()

# Record actions
history.record_action(player_idx=0, action=action, state=game.state)

# Record deal
history.record_deal(street='flop', cards=[Card(...), Card(...), Card(...)])

# Record showdown
history.record_showdown(winners=[0, 2], hands=[hand1, hand2])

# Export
history_dict = history.to_dict()
history_str = history.to_string()  # Human-readable
```

---

## CLI Interface

Play poker from the command line:

```bash
# Run interactive game
python -m engine.cli

# With specific settings
python -m engine.cli --players 6 --small-blind 5 --big-blind 10
```

**Controls**:
- `f` - Fold
- `c` - Check/Call
- `r <amount>` - Raise to amount
- `a` - All-in
- `q` - Quit

---

## Optimizations

### 1. Fast Hand Evaluation (13-16× speedup)

```python
from engine import evaluate_hand  # Uses hand_eval_fast.py

# Pre-computed lookup tables
# Evaluates 7-card hand in <1μs
rank = evaluate_hand(seven_cards)
```

### 2. Feature Caching (1.5-2× speedup)

```python
from engine import FeatureCache

cache = FeatureCache()

# Cache expensive computations
features = get_state_vector(game, player_idx, cache)

# Clear cache when state changes significantly
cache.clear()
```

### 3. Numpy RNG (1.05-1.1× speedup)

```python
from engine import Deck

# Uses numpy's PCG64 RNG (faster than Python's random)
deck = Deck(seed=42)
```

### 4. Memory Pooling (1.2-1.4× speedup)

```python
from engine import GamePool

# Reuse game objects
pool = GamePool(size=100)
game = pool.acquire(player_stacks=[1000]*6, seed=42)
# ... play game ...
pool.release(game)
```

---

## Usage Examples

### Example 1: Simple Game

```python
from engine import PokerGame, Action, ActionType

# Create game
game = PokerGame(player_stacks=[1000, 1000, 1000], seed=42)
game.start_new_hand()

# Play hand
while not game.is_hand_over():
    player = game.state.current_player
    
    # Simple strategy: always call
    if game.state.can_check(player):
        action = Action(ActionType.CHECK)
    else:
        action = Action(ActionType.CALL, game.state.current_bet)
    
    game.apply_action(action)

# Determine winner
winners, amounts = game.determine_winners()
print(f"Winners: {winners}, Amounts: {amounts}")
```

### Example 2: AI Integration

```python
from engine import PokerGame, get_state_vector, FeatureCache
from training import PolicyNetwork

# Setup
game = PokerGame(player_stacks=[1000]*6, seed=42)
network = PolicyNetwork.from_genome(genome, config)
cache = FeatureCache()

# Play
game.start_new_hand()
while not game.is_hand_over():
    player = game.state.current_player
    
    # Get state features
    features = get_state_vector(game, player, cache)
    
    # Get legal actions mask
    mask = get_action_mask(game, player)
    
    # AI selects action
    action_idx = network.select_action(features, mask, rng)
    
    # Convert to game action
    action = abstract_to_concrete_action(game, player, action_idx)
    
    # Apply
    game.apply_action(action)
```

### Example 3: Batch Simulation

```python
from engine import PokerGame
import numpy as np

def simulate_hands(num_hands=1000, num_players=6):
    """Simulate many hands for statistics."""
    game = PokerGame(player_stacks=[1000]*num_players, seed=42)
    
    results = []
    for hand_num in range(num_hands):
        game.start_new_hand()
        
        # Play hand with random actions
        while not game.is_hand_over():
            player = game.state.current_player
            legal_actions = game.get_legal_actions(player)
            action = np.random.choice(legal_actions)
            game.apply_action(action)
        
        # Record results
        winners, amounts = game.determine_winners()
        results.append({'winners': winners, 'amounts': amounts})
    
    return results

# Run simulation
results = simulate_hands(num_hands=10000)
```

---

## API Reference

### PokerGame

```python
class PokerGame:
    def __init__(self, player_stacks: List[int], small_blind: int = 5, 
                 big_blind: int = 10, ante: int = 0, seed: int = None)
    
    def start_new_hand(self) -> None
    def apply_action(self, action: Action) -> None
    def is_hand_over(self) -> bool
    def determine_winners(self) -> Tuple[List[int], List[int]]
    def award_pots(self, winners: List[int], amounts: List[int]) -> None
    
    def get_legal_actions(self, player_idx: int) -> List[Action]
    def get_action_mask(self, player_idx: int) -> np.ndarray
```

### Card

```python
class Card:
    def __init__(self, rank: str, suit: str)
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card'
    
    def rank_value(self) -> int
    def __str__(self) -> str
    def __eq__(self, other) -> bool
```

### Deck

```python
class Deck:
    def __init__(self, seed: int = None)
    
    def shuffle(self) -> None
    def deal(self) -> Card
    def deal_n(self, n: int) -> List[Card]
    def reset(self) -> None
```

### Action

```python
class Action:
    def __init__(self, action_type: ActionType, amount: int = 0)
    
    @classmethod
    def from_string(cls, action_str: str) -> 'Action'
    
    def __str__(self) -> str
```

---

## Testing

```bash
# Run engine tests
python -m pytest tests/test_engine.py -v

# Test hand evaluation speed
python scripts/test_hand_eval_speed.py

# Test game logic
python scripts/test_cli.py
```

---

## Performance Benchmarks

### Hand Evaluation

```
Standard evaluator:  1.2 μs/hand
Fast evaluator:      0.08 μs/hand
Speedup:             15×
```

### Full Hand Simulation

```
With all optimizations:  12-13 seconds per 36,000 hands
Throughput:              ~2,800 hands/second
                         ~10M hands/hour
```

### Memory Usage

```
PokerGame object:     ~5 KB
Feature cache:        ~50 KB
Hand history:         ~10 KB/hand
```

---

## Position and Blind Rules

### Button and Blind Positions

```python
# 6-player game
# Button = 0
# Small Blind = 1
# Big Blind = 2
# UTG (first to act preflop) = 3
# Action goes clockwise
```

### First to Act

- **Preflop**: Player after big blind (UTG)
- **Postflop**: Player after button (small blind if active)

```python
first_actor = game.state.first_to_act()
```

---

## Error Handling

```python
from engine import PokerGame, InvalidActionError

game = PokerGame(player_stacks=[1000]*6)
game.start_new_hand()

try:
    # Attempt invalid action
    game.apply_action(Action(ActionType.RAISE, 5))  # Below min raise
except InvalidActionError as e:
    print(f"Invalid action: {e}")
    
    # Get legal actions
    legal = game.get_legal_actions(player_idx)
    print(f"Legal actions: {legal}")
```

---

## Integration with Training

The engine is designed for efficient AI training:

```python
from engine import PokerGame, get_state_vector, FeatureCache
from training import PolicyNetwork, evaluate_matchup

# Evaluate agent fitness
fitness, hands = evaluate_matchup(
    genome_weights=agent.weights,
    opponent_weights=[opp1.weights, opp2.weights, ...],
    network_config=config,
    fitness_config=fitness_config
)

# Uses engine internally:
# 1. Creates PokerGame instances
# 2. Plays hands with PolicyNetwork agents
# 3. Tracks performance (BB/100)
# 4. Returns fitness score
```

See [training/README.md](../training/README.md) for details.

---

## Troubleshooting

### Issue: Slow hand evaluation
**Solution**: Ensure using `hand_eval_fast.py` (imported by default in `__init__.py`)

### Issue: Side pot errors
**Solution**: Verify all-in amounts are correctly recorded in pot before `close_betting_round()`

### Issue: Invalid action errors
**Solution**: Always check `get_legal_actions()` before applying actions

### Issue: Memory growth
**Solution**: Use GamePool for object reuse in long simulations

---

## Related Scripts

**Game Simulation**:
- **`scripts/train.py`**: Uses engine for self-play training
- **`scripts/match_agents.py`**: Head-to-head matches
- **`scripts/eval_baseline.py`**: Agent evaluation
- **`scripts/round_robin_agents_config.py`**: Tournament simulation

**Testing & Debugging**:
- **`scripts/test_cli.py`**: Interactive poker game
- **`scripts/test_ai_hands.py`**: Scenario testing
- **`scripts/test_ai_features.py`**: Feature extraction verification

**Analysis**:
- **`scripts/visualize_agent_behavior.py`**: Uses engine state for analysis
- **`scripts/plot_history.py`**: Training progress (hand counts, etc.)

---

## Future Enhancements

**Game Variants**:
- [ ] **Tournament support**: Increasing blinds, antes, payout structures
- [ ] **Omaha Hold'em**: 4 hole cards, use exactly 2
- [ ] **Short deck poker** (6+ Hold'em): 36-card deck, different hand rankings
- [ ] **Pot-Limit Omaha (PLO)**: Pot-sized raises only
- [ ] **Razz/Lowball variants**: Lowest hand wins

**Advanced Features**:
- [ ] **Deal insurance**: Side bets on outcomes
- [ ] **Run-it-twice**: Deal board multiple times after all-in
- [ ] **Time bank**: Thinking time limits per player
- [ ] **Straddles**: Optional blind raises
- [ ] **Button ante**: Alternative ante structure

**Performance & Analysis**:
- [ ] **Faster hand evaluation**: C++ extension (2-3× speedup)
- [ ] **Hand history export**: PGN-style format for analysis tools
- [ ] **Detailed timestamps**: Track decision time per action
- [ ] **Live hand visualization**: Web-based real-time display
- [ ] **Replay system**: Step through hands move-by-move

**Infrastructure**:
- [ ] **Game state serialization**: Save/load mid-hand
- [ ] **Undo/redo actions**: For debugging and analysis
- [ ] **Configurable deck**: Custom cards, wild cards
- [ ] **Multi-currency**: Tournament chips, cash game dollars

---

## Performance Tuning

**Current optimizations active**:
- ✅ Fast hand evaluator (13-16× speedup)
- ✅ Feature caching (1.5-2× speedup)
- ✅ NumPy RNG (1.05-1.1× speedup)
- ✅ Memory pooling for long simulations
- ✅ History tracking disabled by default

**Throughput**: ~2,800 hands/second (~10M hands/hour)

**To maximize performance**:
1. Use `hand_eval_fast.py` (imported by default)
2. Disable history: `game.history = None`
3. Use `FeatureCache` for repeated feature extraction
4. Use `GamePool` for object reuse in long runs
5. Minimize `print()` statements in hot loops

---

**For more information, see main [README.md](../README.md) and [training/README.md](../training/README.md)**
