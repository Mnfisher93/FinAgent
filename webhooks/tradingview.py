"""
TradingView Webhook Receiver

Receives alert payloads from TradingView via HTTP POST, authenticates them
with a passphrase, logs them to disk, and optionally feeds them into Claude
for AI-powered analysis.

Usage:
    uv run python run_webhook.py

TradingView alert message format (JSON):
    {
        "passphrase": "your-secret-here",
        "ticker": "AAPL",
        "action": "buy",
        "price": 264.58,
        "indicator": "SMA Cross 50/200",
        "timeframe": "1D",
        "message": "Golden Cross detected on AAPL"
    }
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────
WEBHOOK_PASSPHRASE = os.environ.get("WEBHOOK_PASSPHRASE", "finagent-secret")
WEBHOOK_PORT = int(os.environ.get("WEBHOOK_PORT", "5000"))
ALERTS_FILE = Path(__file__).parent.parent / "data" / "alerts.json"
ENABLE_AI_ANALYSIS = os.environ.get("WEBHOOK_AI_ANALYSIS", "false").lower() == "true"

# Ensure data directory exists
ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Alert Storage ───────────────────────────────────────────────────────

def load_alerts() -> list[dict]:
    """Load existing alerts from disk."""
    if ALERTS_FILE.exists():
        with open(ALERTS_FILE) as f:
            return json.load(f)
    return []


def save_alert(alert: dict) -> None:
    """Append an alert to the alerts file."""
    alerts = load_alerts()
    alerts.append(alert)
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2, default=str)


def get_ai_analysis(alert: dict) -> str | None:
    """Feed the alert into Claude for analysis (if enabled)."""
    if not ENABLE_AI_ANALYSIS:
        return None
    try:
        from agent import FinancialAgent
        agent = FinancialAgent()
        prompt = (
            f"TradingView alert received:\n"
            f"  Ticker: {alert.get('ticker', 'N/A')}\n"
            f"  Action: {alert.get('action', 'N/A')}\n"
            f"  Price: {alert.get('price', 'N/A')}\n"
            f"  Indicator: {alert.get('indicator', 'N/A')}\n"
            f"  Timeframe: {alert.get('timeframe', 'N/A')}\n"
            f"  Message: {alert.get('message', 'N/A')}\n\n"
            f"Please analyze this signal. Look up the current price and recent "
            f"performance, then give your assessment of whether this is a strong "
            f"signal. Be concise."
        )
        return agent.chat(prompt)
    except Exception as e:
        return f"AI analysis error: {e}"


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Receive a TradingView alert.

    Expects JSON body with at minimum a 'passphrase' field for auth.
    All other fields are stored as-is.
    """
    # Parse payload
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # Authenticate
    passphrase = data.get("passphrase", "")
    if passphrase != WEBHOOK_PASSPHRASE:
        print(f"  ❌ Unauthorized webhook attempt (bad passphrase)")
        return jsonify({"error": "Unauthorized"}), 401

    # Remove passphrase from stored data
    alert = {k: v for k, v in data.items() if k != "passphrase"}
    alert["received_at"] = datetime.now(timezone.utc).isoformat()

    # Log to console
    ticker = alert.get("ticker", "???")
    action = alert.get("action", "???")
    price = alert.get("price", "???")
    indicator = alert.get("indicator", "")
    print(f"\n  📨 Alert: {action.upper()} {ticker} @ ${price}")
    if indicator:
        print(f"     Indicator: {indicator}")
    if alert.get("message"):
        print(f"     Message: {alert['message']}")

    # AI Analysis (if enabled)
    ai_response = get_ai_analysis(alert)
    if ai_response:
        alert["ai_analysis"] = ai_response
        print(f"  🤖 AI: {ai_response[:200]}...")

    # Save
    save_alert(alert)
    print(f"  💾 Saved ({len(load_alerts())} total alerts)\n")

    return jsonify({
        "status": "received",
        "ticker": ticker,
        "action": action,
        "ai_analysis": ai_response,
    }), 200


@app.route("/alerts", methods=["GET"])
def get_alerts():
    """View all stored alerts."""
    alerts = load_alerts()
    limit = request.args.get("limit", 50, type=int)
    return jsonify({
        "total": len(alerts),
        "alerts": alerts[-limit:],
    })


@app.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    """Clear all stored alerts."""
    passphrase = request.get_json(force=True).get("passphrase", "")
    if passphrase != WEBHOOK_PASSPHRASE:
        return jsonify({"error": "Unauthorized"}), 401
    with open(ALERTS_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"status": "cleared"})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "ai_analysis": ENABLE_AI_ANALYSIS,
        "alerts_count": len(load_alerts()),
    })


# ── Main ────────────────────────────────────────────────────────────────

def run_server():
    """Start the webhook server."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        📡  TradingView Webhook Receiver  📡                 ║
║                                                              ║
║   Listening on: http://0.0.0.0:{WEBHOOK_PORT:<24}          ║
║   Webhook URL:  http://localhost:{WEBHOOK_PORT}/webhook       ║
║   View alerts:  http://localhost:{WEBHOOK_PORT}/alerts        ║
║   AI Analysis:  {'✅ Enabled' if ENABLE_AI_ANALYSIS else '❌ Disabled (set WEBHOOK_AI_ANALYSIS=true)'}          ║
║                                                              ║
║   For external access, use: ngrok http {WEBHOOK_PORT:<17}   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=WEBHOOK_PORT, debug=False)


if __name__ == "__main__":
    run_server()
