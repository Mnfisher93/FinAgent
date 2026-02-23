"""
Quantitative Fintech Agent — Interactive Chat Interface

Supports: Claude (Anthropic) • GPT-4 (OpenAI) • Gemini (Google) • Grok (xAI)

Run: uv run main.py
"""

import sys
from agent import FinancialAgent


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
║   "Show me Polymarket predictions"                          ║
║                                                              ║
║   Commands: 'clear' to reset • 'quit' to exit              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    try:
        agent = FinancialAgent()
    except ValueError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    print("  ✅ Ready! Ask me anything about stocks, crypto, or markets.\n")

    while True:
        try:
            user_input = input("\033[1;36m  You → \033[0m").strip()
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
            print(f"\033[1;33m  Agent →\033[0m {response}\n")
        except Exception as e:
            print(f"\033[1;31m  Error →\033[0m {e}\n")


if __name__ == "__main__":
    main()
