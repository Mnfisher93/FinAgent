"""
Run the TradingView Webhook Receiver.

Usage:
    uv run python run_webhook.py

Then configure TradingView alerts to POST to:
    http://your-server:5000/webhook

Use ngrok for external access:
    ngrok http 5000
"""

from webhooks.tradingview import run_server

if __name__ == "__main__":
    run_server()
