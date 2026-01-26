import sys
import random
from engine.game import PokerGame
from engine.actions import Action

def prompt_num_hands():
    while True:
        try:
            x = int(input("Enter number of hands to simulate: "))
            if x > 0:
                return x
            print("Please enter a positive integer.")
        except Exception:
            print("Invalid input. Please enter a number.")

def run_ai_hands(num_hands, num_players=2, stack=100, sb=5, bb=10, seed=None):
    results = [0] * num_players
    hand_logs = []
    failures = []
    showdown_count = 0

    for hand in range(num_hands):
        try:
            game = PokerGame([stack] * num_players, small_blind=sb, big_blind=bb, seed=seed)
            hand_log = []
            print(f"\n--- Hand {hand+1} ---")

            while not game.is_hand_over():
                # Print current state
                print(f"Community cards: {game.state.community_cards}")
                for p in game.players:
                    print(f"Player {p.player_id}: stack={p.stack}, bet={p.bet}, folded={p.has_folded}, all-in={p.is_all_in}")
                print(f"Pot: {game.state.pot.total}")
                print(f"Current betting round: {game.state.betting_round}")
                print(f"Current player: {game.state.current_player}")

                # If no current player (round is advancing), skip to next loop
                if game.state.current_player is None:
                    continue

                player = game.players[game.state.current_player]

                # Skip player if folded, all-in, or no stack
                if player.has_folded or player.is_all_in or player.stack == 0:
                    continue

                # Centralized legal action generation (professional, rule-correct)
                legal = []
                to_call = max(0, game.current_bet - player.bet)
                min_raise = max(bb, game.state.big_blind)
                # If player has no chips, no actions are legal
                if player.stack == 0:
                    legal = []
                # If nothing to call
                elif to_call == 0:
                    legal.append('check')
                    # Can only raise if enough for min raise
                    if player.stack >= min_raise:
                        legal.append('raise')
                    legal.append('fold')
                # If player can't cover the call (short stack)
                elif player.stack <= to_call:
                    legal.append('all-in')
                    legal.append('fold')
                # If player can call and possibly raise
                elif player.stack > to_call:
                    legal.append('call')
                    # Can only raise if enough for min raise
                    if player.stack > to_call + min_raise:
                        legal.append('raise')
                    legal.append('fold')

                # Weighted AI decision
                weights = []
                for act in legal:
                    if act == 'fold':
                        weights.append(0.0)
                    elif act == 'check':
                        weights.append(0.4)
                    elif act == 'call':
                        weights.append(0.4)
                    elif act == 'raise':
                        weights.append(0.15)
                    elif act == 'all-in':
                        weights.append(0.05)
                    else:
                        weights.append(0.01)
                total = sum(weights)
                weights = [w / total for w in weights]
                ai_action = random.choices(legal, weights=weights, k=1)[0]

                # Map to Action object
                if ai_action == 'fold':
                    action = Action(Action.FOLD)
                elif ai_action == 'check':
                    action = Action(Action.CHECK)
                elif ai_action == 'call':
                    action = Action(Action.CALL)
                elif ai_action == 'raise':
                    # Calculate valid raise range
                    min_valid = max(game.current_bet + min_raise, player.bet + to_call + min_raise)
                    max_valid = player.bet + player.stack
                    if max_valid <= min_valid:
                        action = Action(Action.ALL_IN)
                    else:
                        raise_to = random.randint(min_valid, max_valid)
                        action = Action(Action.RAISE, raise_to)
                elif ai_action == 'all-in':
                    action = Action(Action.ALL_IN)
                else:
                    action = Action(Action.FOLD)  # fallback

                # Execute action
                print(f"Player {player.player_id} action: {action}")
                hand_log.append(f"Player {player.player_id} action: {action}")
                game.apply_action(player.player_id, action)

            # After hand, count showdown if more than one player active
            active_players = [p for p in game.players if not p.has_folded]
            if len(active_players) > 1:
                showdown_count += 1

            # Resolve showdown safely
            results_this_hand = game.resolve_showdown()
            if results_this_hand:
                for pid, win in results_this_hand.items():
                    results[pid] += win

            # Log hand results
            hand_log.append("Hand results:")
            if results_this_hand:
                for pid, win in results_this_hand.items():
                    hand_log.append(f"Player {pid} wins {win}")
            else:
                hand_log.append("No showdown (all others folded)")

            hand_logs.append('\n'.join(hand_log))

        except Exception as e:
            import traceback
            failures.append(f"Hand {hand+1}: {e}\n{traceback.format_exc()}")

    return results, hand_logs, failures, showdown_count

def main():
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open("ai_hands_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    try:
        num_hands = prompt_num_hands()
        num_players = 5
        stack = 100
        sb = 5
        bb = 10
        seed = None
        print(f"Simulating {num_hands} hands with {num_players} AI players...")
        results, hand_logs, failures, showdown_count = run_ai_hands(num_hands, num_players, stack, sb, bb, seed)

        print("\n--- Hand-by-hand results ---")
        for log in hand_logs:
            print(log)

        print("\nTotal winnings after all hands:")
        for i, chips in enumerate(results):
            print(f"Player {i}: {chips}")

        print(f"\nNumber of hands with actual showdown between multiple players: {showdown_count}")

        if failures:
            print("\n--- Failures encountered ---")
            for fail in failures:
                print(fail)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()

if __name__ == "__main__":
    main()
