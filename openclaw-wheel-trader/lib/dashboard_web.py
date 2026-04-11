"""
Web Dashboard Server — Flask API + HTML dashboard at localhost:5050.

Usage:
  python main.py dashboard
  python main.py dashboard --port 8080
"""

from pathlib import Path

from flask import Flask, jsonify, render_template

from lib.dashboard_data import (
    get_full_dashboard_state, get_quant_scores,
    get_portfolio_summary, get_positions_table,
    get_events, get_trade_history, get_circuit_breaker_status,
)

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    return jsonify(get_full_dashboard_state())


@app.route("/api/quant")
def api_quant():
    return jsonify(get_quant_scores())


@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(get_portfolio_summary())


@app.route("/api/positions")
def api_positions():
    return jsonify(get_positions_table())


@app.route("/api/events")
def api_events():
    return jsonify(get_events(30))


@app.route("/api/history")
def api_history():
    return jsonify(get_trade_history())


@app.route("/api/breakers")
def api_breakers():
    return jsonify(get_circuit_breaker_status())


def run_dashboard(port: int = 5050):
    """Start the dashboard web server."""
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
