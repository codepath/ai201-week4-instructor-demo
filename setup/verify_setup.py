"""
Verify Setup
============
Run this the night before or morning of the demo.

Usage:
    python setup/verify_setup.py

Checks:
    1. Test messages load and cover all five scenarios
    2. Rate limiter fires correctly under load
    3. OpenRouter API key is set and returns a valid JSON moderation verdict
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from data.test_messages import ALL_SCENARIOS, CLEAN, OBVIOUS, RAID
from layers.rate_limiter import RateLimiter
from layers.logger import ModerationLogger
from utils.llm import LLMClient


def check_test_messages() -> bool:
    print("\n[1/3] Checking test messages...")
    required = {"clean", "obvious_violation", "edge_case", "injection", "raid"}
    missing = required - set(ALL_SCENARIOS.keys())
    if missing:
        print(f"  ✗ Missing scenarios: {missing}")
        return False

    for scenario, messages in ALL_SCENARIOS.items():
        print(f"  ✓ {scenario}: {len(messages)} messages")
    return True


def check_rate_limiter() -> bool:
    print("\n[2/3] Checking rate limiter...")
    limiter = RateLimiter(user_limit=3, server_limit=10, window_seconds=5)

    # Test per-user limit
    for i in range(3):
        result = limiter.check("user_test")
        if not result["allowed"]:
            print(f"  ✗ Rate limiter blocked too early (message {i+1})")
            return False
    result = limiter.check("user_test")
    if result["allowed"]:
        print("  ✗ Rate limiter should have fired after 3 messages — didn't")
        return False
    print(f"  ✓ Per-user limit fires correctly at 3 messages")

    # Test server-wide limit (raid simulation)
    limiter.reset()
    blocked = False
    for i in range(12):
        r = limiter.check(f"raider_{i}")
        if not r["allowed"]:
            blocked = True
            print(f"  ✓ Server-wide limit fired at message {i+1} of 12")
            break
    if not blocked:
        print("  ✗ Server-wide limit never fired during 12-user burst")
        return False

    return True


def check_llm() -> bool:
    print("\n[3/3] Checking OpenRouter LLM with JSON mode...")
    try:
        llm = LLMClient()
        print(f"  Model: {llm.model}")

        # Minimal system prompt for the check
        system = (
            "You are a Discord server moderator. Evaluate messages for a hip-hop fan server. "
            "Return a JSON verdict with decision (allow/timeout/ban), reason, confidence, and rule_triggered."
        )
        test_msg = "Kendrick's discography is unmatched fr"

        result = llm.moderate(system, test_msg)

        required_keys = {"decision", "reason", "confidence", "rule_triggered"}
        if not required_keys.issubset(result.keys()):
            print(f"  ✗ Missing keys in response: {required_keys - result.keys()}")
            return False
        if result["decision"] not in ("allow", "timeout", "ban"):
            print(f"  ✗ Invalid decision value: '{result['decision']}'")
            return False

        print(f"  ✓ LLM returned valid JSON: decision='{result['decision']}', confidence='{result['confidence']}'")
        return True

    except ValueError as e:
        print(f"  ✗ {e}")
        return False
    except Exception as e:
        print(f"  ✗ OpenRouter request failed: {e}")
        return False


def main():
    print("=" * 60)
    print("AI 201 — Week 4 Demo: Pre-Demo Verification")
    print("=" * 60)

    results = [
        check_test_messages(),
        check_rate_limiter(),
        check_llm(),
    ]

    print()
    if all(results):
        print("=" * 60)
        print("✓ All checks passed. Open demo.ipynb to begin.")
        print("=" * 60)
    else:
        print("✗ Some checks failed — fix the errors above before the demo.")
        sys.exit(1)


if __name__ == "__main__":
    main()
