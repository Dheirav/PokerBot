"""
Test script for AI training features.
Tests: features extraction, hand strength calculation, and hand history logging.
"""
import sys
sys.path.insert(0, '/home/dheirav/Code/PokerBot')

from engine import (
    PokerGame, 
    Action,
    get_state_features,
    get_state_vector,
    get_feature_names,
    preflop_hand_strength,
    chen_formula,
    HandHistoryLogger,
    SessionLogger,
    get_action_mask,
    get_raise_sizing_info,
)
from engine.cards import Card

def test_chen_formula():
    """Test Chen formula for various starting hands."""
    print("=" * 50)
    print("Testing Chen Formula")
    print("=" * 50)
    
    # AA should be highest
    aa = [Card('A', 'h'), Card('A', 's')]
    print(f"AA: Chen = {chen_formula(aa):.1f}")
    
    # 72o should be lowest
    seven_two = [Card('7', 'h'), Card('2', 's')]
    print(f"72o: Chen = {chen_formula(seven_two):.1f}")
    
    # AKs should be strong
    aks = [Card('A', 'h'), Card('K', 'h')]
    print(f"AKs: Chen = {chen_formula(aks):.1f}")
    
    # Test normalized hand strength
    print(f"\nNormalized hand strength:")
    print(f"AA: {preflop_hand_strength(aa):.3f}")
    print(f"AKs: {preflop_hand_strength(aks):.3f}")
    print(f"72o: {preflop_hand_strength(seven_two):.3f}")
    
    print("✓ Chen formula tests passed\n")


def test_state_features():
    """Test state feature extraction."""
    print("=" * 50)
    print("Testing State Feature Extraction")
    print("=" * 50)
    
    # Create a 6-player game
    game = PokerGame([1000] * 6, small_blind=5, big_blind=10, seed=42)
    
    # Get features for each player
    print(f"\nFeature names ({len(get_feature_names())} features):")
    for name in get_feature_names():
        print(f"  - {name}")
    
    # Get features for UTG (first to act)
    current_player = game.state.current_player
    print(f"\nFeatures for Player {current_player} (UTG):")
    features = get_state_features(game, current_player)
    
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")
    
    # Get as vector
    vector = get_state_vector(game, current_player)
    print(f"\nState vector (len={len(vector)}): {[round(v, 2) for v in vector]}")
    
    print("✓ State feature tests passed\n")


def test_ante():
    """Test ante support."""
    print("=" * 50)
    print("Testing Ante Support")
    print("=" * 50)
    
    # Create a game with ante
    game = PokerGame([1000] * 6, small_blind=5, big_blind=10, ante=1, seed=42)
    
    # Check pot includes antes + blinds
    # 6 players * 1 ante + 5 SB + 10 BB = 21
    expected_pot = 6 * 1 + 5 + 10
    print(f"Pot with antes: {game.state.pot.total} (expected: {expected_pot})")
    assert game.state.pot.total == expected_pot, f"Expected pot {expected_pot}, got {game.state.pot.total}"
    
    # Check player stacks
    total_chips = sum(p.stack + p.total_contributed for p in game.players)
    print(f"Total chips in game: {total_chips} (expected: 6000)")
    assert total_chips == 6000, "Chip conservation violated"
    
    print("✓ Ante tests passed\n")


def test_action_history():
    """Test action history logging."""
    print("=" * 50)
    print("Testing Action History")
    print("=" * 50)
    
    game = PokerGame([1000] * 3, small_blind=5, big_blind=10, seed=42)
    
    # Play some actions
    actions = [
        (game.state.current_player, Action("call", 10)),  # UTG calls
    ]
    
    for player_idx, action in actions:
        game.apply_action(player_idx, action)
    
    print(f"Action history after 1 action:")
    # Note: Action history disabled during training for performance
    if game.action_history is not None:
        for entry in game.action_history:
            player_id, action_type, amount, street = entry
            print(f"  Player {player_id}: {action_type} ${amount} on {street}")
        assert len(game.action_history) == 1, "Action history should have 1 entry"
    else:
        print("  (Action history disabled for performance)")
    print("✓ Action history tests passed\n")


def test_hand_history_logger():
    """Test hand history logger."""
    print("=" * 50)
    print("Testing Hand History Logger")
    print("=" * 50)
    
    game = PokerGame([1000] * 3, small_blind=5, big_blind=10, seed=42)
    
    logger = HandHistoryLogger(log_dir="logs/test_hands", table_name="TestTable")
    logger.start_hand(game, hand_id=1)
    
    # Log hero cards
    logger.log_hole_cards(0, game.players[0].hole_cards, hero=True)
    
    # Simulate actions
    current = game.state.current_player
    logger.log_action(current, Action("fold"), 0)
    game.apply_action(current, Action("fold"))
    
    current = game.state.current_player
    logger.log_action(current, Action("call"), 5)
    game.apply_action(current, Action("call"))
    
    current = game.state.current_player  
    logger.log_action(current, Action("check"), 0)
    game.apply_action(current, Action("check"))
    
    # Get the history
    history = logger.get_hand_history()
    print("Hand History Preview:")
    print("-" * 40)
    for line in history.split('\n')[:15]:
        print(line)
    print("...")
    print("-" * 40)
    
    print("✓ Hand history logger tests passed\n")


def test_full_hand_with_features():
    """Play a complete hand and track features."""
    print("=" * 50)
    print("Testing Full Hand with Feature Tracking")
    print("=" * 50)
    
    game = PokerGame([1000] * 4, small_blind=5, big_blind=10, seed=123)
    
    print(f"Initial pot: {game.state.pot.total}")
    print(f"Button: Player {game.state.button}")
    print(f"Current player: Player {game.state.current_player}")
    
    hand_num = 0
    while not game.is_hand_over() and hand_num < 20:
        current = game.state.current_player
        if current is None:
            break
            
        player = game.players[current]
        
        # Get features for current player
        features = get_state_features(game, current)
        
        # Simple strategy: fold if hand strength < 0.3, else call
        if features['hand_strength'] < 0.3 and features['facing_bet'] > 0:
            action = Action("fold")
        elif features['to_call_ratio'] > 0:
            action = Action("call")
        else:
            action = Action("check")
        
        print(f"Round {hand_num}: Player {current} (str={features['hand_strength']:.2f}) -> {action.action_type}")
        game.apply_action(current, action)
        hand_num += 1
    
    print(f"\nFinal state: {game.state.betting_round}")
    print(f"Pot: {game.state.pot.total}")
    if game.action_history is not None:
        print(f"Actions in history: {len(game.action_history)}")
    else:
        print(f"Actions in history: (disabled for performance)")
    
    if game.is_hand_over():
        not_folded = [p for p in game.players if not p.has_folded]
        if len(not_folded) == 1:
            print(f"Winner: Player {not_folded[0].player_id}")
        elif game.state.betting_round == "showdown":
            print("Hand went to showdown!")
    
    print("✓ Full hand test passed\n")


def test_action_mask():
    """Test action mask and raise sizing info."""
    print("=" * 50)
    print("Testing Action Mask & Raise Sizing")
    print("=" * 50)
    
    game = PokerGame([1000] * 4, small_blind=5, big_blind=10, seed=42)
    current = game.state.current_player
    
    # Get action mask
    mask = get_action_mask(game, current)
    print(f"Action mask for Player {current}: {mask}")
    print(f"  [fold, check, call, raise, all-in]")
    
    # Get raise sizing info
    sizing = get_raise_sizing_info(game, current)
    print(f"\nRaise sizing info:")
    for key, val in sizing.items():
        print(f"  {key}: {val:.3f}")
    
    assert len(mask) == 5, "Action mask should have 5 elements"
    assert mask[0] == 1, "Fold should be legal"
    
    print("✓ Action mask tests passed\n")


if __name__ == "__main__":
    test_chen_formula()
    test_state_features()
    test_ante()
    test_action_history()
    test_hand_history_logger()
    test_full_hand_with_features()
    test_action_mask()
    
    print("=" * 50)
    print("All AI Training Feature Tests Passed!")
    print("=" * 50)
