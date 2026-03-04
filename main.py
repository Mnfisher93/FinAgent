"""
Quantitative Fintech Agent — Interactive Chat Interface

Supports: Claude (Anthropic) • GPT-4 (OpenAI) • Gemini (Google) • Grok (xAI)

Run: uv run main.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load existing .env if present
load_dotenv()


# ============================================================================
# Terminal colors
# ============================================================================

CYAN = "\033[1;36m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ============================================================================
# API key auto-detection
# ============================================================================

KEY_PREFIXES = {
    "sk-ant-": ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
    "sk-":    ("OPENAI_API_KEY",    "OpenAI (GPT)"),
    "AIza":   ("GEMINI_API_KEY",    "Google (Gemini)"),
    "xai-":   ("XAI_API_KEY",       "xAI (Grok)"),
}


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        📊  Quantitative Fintech Agent  📊                   ║
║        AI-Powered Financial Analyst                          ║
║                                                              ║
║   Supports: Claude • GPT-4 • Gemini • Grok                 ║
║                                                              ║
║   "What's AAPL trading at?"                                 ║
║   "Price of Bitcoin and Ethereum"                           ║
║   "Compare SPY, QQQ, and BTC-USD"                          ║
║   "Analyze NVDA's performance"                              ║
║   "Show me the options chain for Tesla"                     ║
║   "Show me Polymarket predictions"                          ║
║                                                              ║
║   Commands: 'clear' to reset • 'quit' to exit              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# First-run setup — one paste, zero menus
# ============================================================================

def _has_valid_key() -> bool:
    """Check if any valid API key is set in the environment."""
    keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"]
    for key in keys:
        val = os.environ.get(key, "").strip()
        if val and val != "your-api-key-here":
            return True
    return False


def _detect_provider(api_key: str) -> tuple[str, str] | None:
    """Detect provider from API key prefix. Returns (env_key, label) or None."""
    # Check sk-ant- before sk- (both start with sk-)
    for prefix, (env_key, label) in KEY_PREFIXES.items():
        if api_key.startswith(prefix):
            return env_key, label
    return None


def _save_env(env_key: str, api_key: str):
    """Save the API key to .env file."""
    env_path = Path(".env")
    lines = [f"{env_key}={api_key}"]
    env_path.write_text("\n".join(lines) + "\n")
    os.environ[env_key] = api_key


def run_first_time_setup():
    """One-paste setup — auto-detects provider from key prefix."""
    print(f"""
  {BOLD}👋 Welcome! Paste your API key to get started.{RESET}
  {DIM}Supports: Anthropic · OpenAI · Google Gemini · xAI Grok{RESET}
  {DIM}Don't have one? Get a free key at aistudio.google.com/apikey{RESET}
""")

    api_key = input(f"  {BOLD}API key: {RESET}").strip()

    if not api_key:
        print(f"\n  {RED}No key entered. Run the program again to retry.{RESET}\n")
        sys.exit(1)

    result = _detect_provider(api_key)
    if not result:
        print(f"\n  {RED}Couldn't detect provider from key format.{RESET}")
        print(f"  {DIM}Expected prefixes: sk-ant-... (Anthropic) | sk-... (OpenAI) | AIza... (Gemini) | xai-... (xAI){RESET}\n")
        sys.exit(1)

    env_key, label = result
    _save_env(env_key, api_key)

    print(f"""
  {GREEN}✅ Detected: {label}{RESET}
  {DIM}Saved to .env — you won't be asked again.{RESET}
""")


# ============================================================================
# Main
# ============================================================================

def main():
    print(BANNER)

    # First-run setup if no key found
    if not _has_valid_key():
        run_first_time_setup()
        print(BANNER)

    # Import and create agent (after env is set)
    from agent import FinancialAgent

    try:
        agent = FinancialAgent()
    except ValueError as e:
        print(f"\n  {RED}❌ {e}{RESET}")
        sys.exit(1)

    print(f"  {GREEN}✅ Ready! Ask me anything about stocks, crypto, or markets.{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}  You → {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Goodbye!\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  👋 Goodbye!\n")
            break
        if user_input.lower() == "clear":
            agent.clear()
            print("  History cleared.\n")
            continue

        print()
        try:
            response = agent.chat(user_input)
            print(f"{YELLOW}  Agent →{RESET} {response}\n")
        except Exception as e:
            print(f"{RED}  Error →{RESET} {e}\n")


if __name__ == "__main__":
    main()
