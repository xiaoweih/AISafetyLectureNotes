"""
Extended Attack Suite for Defence Evaluation

This module provides 15 additional attack methods for comprehensive defence testing.
For usage, see Evaluate_defence.py with --extended flag.
"""

import os
import sys
import importlib.util

# Add internal attacks directory to path
_ATTACKS_DIR = os.path.join(os.path.dirname(__file__), '_attacks_internal')


def _load_attack_function(filepath):
    """Dynamically load attack function from file"""
    spec = importlib.util.spec_from_file_location("attack_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the attack function (starts with 'attack_')
    for name in dir(module):
        if name.startswith('attack_') and callable(getattr(module, name)):
            return getattr(module, name)
    return None


def get_all_attacks():
    """
    Returns list of 15 attack functions for extended defence evaluation.

    Returns:
        list: List of attack functions with signature (model, X, y, device)

    Example:
        attacks = get_all_attacks()
        for attack_fn in attacks:
            robust_acc = eval_adv_test(model, device, test_loader, attack_fn)
    """
    if not os.path.exists(_ATTACKS_DIR):
        raise FileNotFoundError(
            f"Attack functions directory not found: {_ATTACKS_DIR}\n"
            "Please ensure the extended attacks package is properly installed."
        )

    # List all attack files (bytecode .pyc files)
    attack_files = sorted([
        f for f in os.listdir(_ATTACKS_DIR)
        if f.startswith('attack_') and f.endswith('.pyc')
    ])

    if len(attack_files) == 0:
        raise RuntimeError("No attack functions found in attacks directory")

    # Load each attack function
    attacks = []
    for attack_file in attack_files:
        filepath = os.path.join(_ATTACKS_DIR, attack_file)
        try:
            attack_fn = _load_attack_function(filepath)
            if attack_fn is not None:
                attacks.append(attack_fn)
        except Exception as e:
            # Skip files that fail to load
            print(f"Warning: Could not load {attack_file}: {e}", file=sys.stderr)
            continue

    return attacks
