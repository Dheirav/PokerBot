import sys
import os
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch sys.argv to simulate CLI arguments
sys.argv = ["cli.py", "--players", "2", "--stack", "100", "--sb", "5", "--bb", "10", "--seed", "42"]

# Predefined user actions for the test (simulate a simple hand)
actions = [
    "call",   # preflop
    "check",  # flop
    "check",  # turn
    "check",  # river
]

def input_side_effect(prompt):
    print(prompt, end="")
    if actions:
        action = actions.pop(0)
        print(action)
        return action
    print("check")
    return "check"

# Capture output
output = StringIO()

with patch("builtins.input", side_effect=input_side_effect), patch("sys.stdout", output):
    import engine.cli
    engine.cli.main()

# Print the captured output
print(output.getvalue())
