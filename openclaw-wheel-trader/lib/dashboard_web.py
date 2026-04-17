"""
Web Dashboard Server — Flask API + HTML dashboard at localhost:5051.

Default port is 5051 so it does NOT collide with the sibling polybot project
(which defaults to 5050). If both defaulted to 5050, starting one would
silently steal the port from the other.

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


def run_dashboard(port: int = 5051):
    """Start the dashboard web server.

    Fails fast with a helpful message if the port is already bound (e.g. polybot
    on 5050, or a stale traderbot dashboard on 5051), so the two projects can't
    silently step on each other.
    """
    import errno
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            print(f"ERROR: Port {port} is already in use on 127.0.0.1.")
            print(f"       Another dashboard may already be running "
                  f"(polybot uses 5050, traderbot uses 5051).")
            print(f"       Check with:  lsof -i :{port}")
            print(f"       Or pick a different port:  python main.py dashboard --port <N>")
            raise SystemExit(2)
        raise
    finally:
        probe.close()

    print(f"Traderbot Dashboard: http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
