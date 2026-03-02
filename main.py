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
# Provider config (mirrors agent.py PROVIDERS for setup purposes)
# ============================================================================

SETUP_PROVIDERS = {
    "1": {
        "name": "anthropic",
        "label": "Anthropic (Claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-20250514", "claude-haiku-3-20250414"],
        "url": "https://console.anthropic.com/",
        "free": "$5 credit on signup",
    },
    "2": {
        "name": "openai",
        "label": "OpenAI (GPT)",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "url": "https://platform.openai.com/api-keys",
        "free": "Pay-as-you-go",
    },
    "3": {
        "name": "gemini",
        "label": "Google (Gemini)",
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
        "url": "https://aistudio.google.com/apikey",
        "free": "✅ Free tier",
    },
    "4": {
        "name": "xai",
        "label": "xAI (Grok)",
        "env_key": "XAI_API_KEY",
        "models": ["grok-4", "grok-3", "grok-3-fast"],
        "url": "https://console.x.ai/",
        "free": "Free credits on signup",
    },
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
# First-run setup — prompts for API key if none found
# ============================================================================

def _has_valid_key() -> bool:
    """Check if any valid API key is set in the environment."""
    keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"]
    for key in keys:
        val = os.environ.get(key, "").strip()
        if val and val != "your-api-key-here":
            return True
    return False


def _save_env(env_key: str, api_key: str, model: str | None = None):
    """Save the API key (and optional model) to .env file."""
    env_path = Path(".env")
    lines = [f"{env_key}={api_key}"]
    if model:
        lines.append(f"MODEL={model}")
    env_path.write_text("\n".join(lines) + "\n")
    # Also set in current process so agent picks it up
    os.environ[env_key] = api_key
    if model:
        os.environ["MODEL"] = model


def run_first_time_setup():
    """Interactive first-run setup — pick provider, paste key, choose model."""
    print(f"""
  {BOLD}👋 Welcome! Let's get you set up.{RESET}
  {DIM}This only happens once — your settings will be saved automatically.{RESET}

  {BOLD}Which AI provider do you want to use?{RESET}
""")

    for num, p in SETUP_PROVIDERS.items():
        print(f"    {GREEN}[{num}]{RESET}  {p['label']}  {DIM}— {p['free']}{RESET}")

    print()
    choice = input(f"  {BOLD}Enter choice (1-4): {RESET}").strip()

    if choice not in SETUP_PROVIDERS:
        print(f"\n  {RED}Invalid choice. Run the program again to retry.{RESET}\n")
        sys.exit(1)

    provider = SETUP_PROVIDERS[choice]

    print(f"""
  {BOLD}Great! You chose {provider['label']}.{RESET}

  {DIM}Get your API key at:{RESET} {provider['url']}
  {DIM}Copy the key and paste it below.{RESET}
""")

    api_key = input(f"  {BOLD}Paste your API key: {RESET}").strip()

    if not api_key:
        print(f"\n  {RED}No key entered. Run the program again to retry.{RESET}\n")
        sys.exit(1)

    # Model selection
    models = provider["models"]
    print(f"""
  {BOLD}Which model do you want to use?{RESET}
  {DIM}(The first option is recommended){RESET}
""")

    for i, m in enumerate(models, 1):
        rec = f"  {DIM}← recommended{RESET}" if i == 1 else ""
        print(f"    {GREEN}[{i}]{RESET}  {m}{rec}")

    print()
    model_choice = input(f"  {BOLD}Enter choice (1-{len(models)}, or Enter for recommended): {RESET}").strip()

    if model_choice and model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
        selected_model = models[int(model_choice) - 1]
    else:
        selected_model = models[0]

    # Save to .env
    _save_env(provider["env_key"], api_key, selected_model)

    print(f"""
  {GREEN}✅ Saved!{RESET} Your settings are in {DIM}.env{RESET}
  {DIM}Provider: {provider['label']}{RESET}
  {DIM}Model:    {selected_model}{RESET}
  {DIM}You won't be asked again unless you delete .env{RESET}
""")

    return selected_model


# ============================================================================
# Main
# ============================================================================

def main():
    print(BANNER)

    # First-run setup if no key found
    selected_model = None
    if not _has_valid_key():
        selected_model = run_first_time_setup()
        # Replay banner so user sees fresh start with their config
        print(BANNER)

    # Now import and create agent (after env is set)
    from agent import FinancialAgent

    try:
        agent = FinancialAgent(model=selected_model)
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
