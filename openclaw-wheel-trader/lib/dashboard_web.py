"""
Web Dashboard Server — Flask API + HTML dashboard at localhost:5051.

Default port is 5051 so it does NOT collide with the sibling polybot project
(which defaults to 5050). If both defaulted to 5050, starting one would
silently steal the port from the other.

Usage:
  python main.py dashboard
  python main.py dashboard --port 8080
  python main.py dashboard --host 0.0.0.0   # LAN/Tailscale expose (no auth — trusted networks only)

Phone access: docs/MOBILE_ACCESS.md (recommended path is `tailscale serve`,
which needs no --host change).
"""

from pathlib import Path

from flask import Flask, jsonify, render_template

from lib.dashboard_data import (
    get_full_dashboard_state, get_quant_scores,
    get_portfolio_summary, get_positions_table,
    get_events, get_trade_history, get_circuit_breaker_status,
    get_agent_thinking, get_goal_progress, get_markov_summary,
    get_hermes_state,
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


@app.route("/api/all_open_trades")
def api_all_open_trades():
    """Unified open-trades view across every sibling bot.
    Same response shape across polybot, cryptobot, wheel-trader.
    """
    from lib.all_bots_positions import get_all_bots_open_positions
    return jsonify(get_all_bots_open_positions())


@app.route("/api/events")
def api_events():
    return jsonify(get_events(30))


@app.route("/api/history")
def api_history():
    return jsonify(get_trade_history())


@app.route("/api/breakers")
def api_breakers():
    return jsonify(get_circuit_breaker_status())


@app.route("/api/thinking")
def api_thinking():
    return jsonify(get_agent_thinking())


@app.route("/api/goal-progress")
def api_goal_progress():
    return jsonify(get_goal_progress())


@app.route("/api/markov")
def api_markov():
    from flask import request
    ticker = request.args.get("ticker", "SPY").upper()
    refresh = request.args.get("refresh", "0") == "1"
    return jsonify(get_markov_summary(ticker, refresh=refresh))


@app.route("/api/hermes")
def api_hermes():
    return jsonify(get_hermes_state())


def run_dashboard(port: int = 5051, host: str = "127.0.0.1"):
    """Start the dashboard web server.

    Fails fast with a helpful message if the port is already bound (e.g. polybot
    on 5050, or a stale traderbot dashboard on 5051), so the two projects can't
    silently step on each other.

    Binds to 127.0.0.1 by default so the dashboard is reachable only from this
    machine. For phone access prefer `tailscale serve`, which proxies to the
    localhost bind — no non-default host needed (docs/MOBILE_ACCESS.md). A
    non-localhost host exposes the dashboard, unauthenticated, to whatever
    network that interface is on.
    """
    import errno
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        probe.bind((host, port))
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            print(f"ERROR: Port {port} is already in use on {host}.")
            print(f"       Another dashboard may already be running "
                  f"(polybot uses 5050, traderbot uses 5051).")
            print(f"       Check with:  lsof -i :{port}")
            print(f"       Or pick a different port:  python main.py dashboard --port <N>")
            raise SystemExit(2)
        raise
    finally:
        probe.close()

    if host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: Dashboard bound to a non-localhost address with NO authentication.")
        print("         Anyone who can reach that network interface can view portfolio data.")
        print("         Prefer `tailscale serve` (keeps the localhost-only bind);")
        print("         see docs/MOBILE_ACCESS.md.")

    print(f"Traderbot Dashboard: http://{'localhost' if host == '127.0.0.1' else host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
