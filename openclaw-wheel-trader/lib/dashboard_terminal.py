"""
Terminal Dashboard — Rich-based colored status display.

Usage:
  python main.py status        # Fast: portfolio + positions + breakers
  python main.py status --full # Includes quant scores (slower)
"""

from lib.dashboard_data import (
    get_portfolio_summary, get_positions_table, get_open_orders,
    get_circuit_breaker_status, get_quant_scores,
)


def render_terminal_dashboard(include_quant: bool = False):
    """Render the full terminal dashboard using Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()

    # --- Portfolio Header ---
    portfolio = get_portfolio_summary()

    if "error" in portfolio:
        console.print(f"[bold red]Error connecting to Alpaca:[/] {portfolio['error']}")
        return

    mode_badge = "[bold yellow]PAPER[/]" if portfolio["mode"] == "paper" else "[bold red]LIVE[/]"
    phase_text = f"Phase {portfolio['phase']}: {portfolio['phase_label']}"
    regime_text = portfolio["regime"].upper()

    pl = portfolio["daily_pl"]
    pl_color = "green" if pl >= 0 else "red"
    pl_str = f"[{pl_color}]${pl:+,.2f}[/]"

    # %-gain-to-date vs baseline equity
    baseline = portfolio.get("baseline_equity", 0)
    dollar_gain = portfolio.get("dollar_gain_to_date", 0)
    pct_gain = portfolio.get("pct_gain_to_date", 0)
    gain_color = "green" if dollar_gain >= 0 else "red"
    baseline_date = (portfolio.get("baseline_set_at") or "")[:10]
    gain_line = (
        f"  Gain vs. ${baseline:,.2f} baseline"
        + (f" (since {baseline_date})" if baseline_date else "")
        + f": [{gain_color}]${dollar_gain:+,.2f}[/] "
          f"[{gain_color}]{pct_gain:+.2%}[/]"
    )

    header = Text.from_markup(
        f"  Portfolio: [bold]${portfolio['portfolio_value']:,.2f}[/]  "
        f"Cash: ${portfolio['cash']:,.2f}  "
        f"Buying Power: ${portfolio['buying_power']:,.2f}\n"
        f"  Mode: {mode_badge}  {phase_text}  Regime: [cyan]{regime_text}[/]  Daily P/L: {pl_str}\n"
        f"{gain_line}"
    )
    console.print(Panel(header, title="[bold gold1]OPENCLAW WHEEL TRADER[/]", border_style="gold1"))

    # --- Positions Table ---
    positions = get_positions_table()
    if positions and not (len(positions) == 1 and "error" in positions[0]):
        table = Table(title="Positions", box=box.ROUNDED, border_style="blue")
        table.add_column("Ticker", style="bold")
        table.add_column("Type", style="dim")
        table.add_column("Shares", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P/L $", justify="right")
        table.add_column("P/L %", justify="right")
        table.add_column("Target", justify="right", style="dim")
        table.add_column("Stop", justify="right", style="dim")
        table.add_column("Score", justify="center")

        for p in positions:
            if "error" in p:
                continue
            pnl = p.get("pnl", 0)
            pnl_pct = p.get("pnl_pct", 0)
            pnl_color = "green" if pnl >= 0 else "red"
            pending = " [dim](pending)[/]" if p.get("pending") else ""

            table.add_row(
                f"{p['ticker']}{pending}",
                p.get("type", ""),
                str(p.get("shares", 0)),
                f"${p.get('entry_price', 0):.2f}",
                f"${p.get('current_price', 0):.2f}" if p.get("current_price") else "-",
                f"[{pnl_color}]${pnl:+.2f}[/]",
                f"[{pnl_color}]{pnl_pct:+.1%}[/]",
                f"${p.get('target', 0):.2f}" if p.get("target") else "-",
                f"${p.get('stop', 0):.2f}" if p.get("stop") else "-",
                f"{p.get('score', 0)}/9",
            )
        console.print(table)
    else:
        console.print("[dim]No open positions[/]")

    # --- Pending Orders ---
    orders = get_open_orders()
    if orders and not (len(orders) == 1 and "error" in orders[0]):
        console.print(f"\n[bold]Pending Orders:[/] {len(orders)}")
        for o in orders:
            if "error" in o:
                continue
            console.print(f"  {o.get('symbol', ''):8s} {o.get('side', ''):4s} "
                         f"qty={o.get('qty', '')} {o.get('status', '')}")

    # --- Circuit Breakers ---
    cb = get_circuit_breaker_status()
    if "error" not in cb:
        breakers = cb.get("breakers", {})
        lines = []
        for name, b in breakers.items():
            pct = b.get("pct_used", 0)
            if b.get("tripped"):
                bullet = "[bold red]TRIPPED[/]"
            elif pct > 0.6:
                bullet = "[yellow]WARNING[/]"
            else:
                bullet = "[green]OK[/]"
            label = name.replace("_", " ").title()
            lines.append(f"  {bullet:20s} {label}: {b.get('current', 0)} / {b.get('limit', 0)}")

        paper = "[green]Yes[/]" if cb.get("paper_mode") else "[bold red]NO[/]"
        lines.append(f"\n  Paper Mode: {paper}")

        console.print(Panel("\n".join(lines), title="Circuit Breakers", border_style="yellow"))

    # --- Quant Scores (optional) ---
    if include_quant:
        console.print("\n[dim]Loading quant scores...[/]")
        scores = get_quant_scores()
        if scores and not (len(scores) == 1 and "error" in scores[0]):
            qt = Table(title="Quant Scores", box=box.SIMPLE, border_style="magenta")
            qt.add_column("#", justify="right", style="dim")
            qt.add_column("Ticker", style="bold")
            qt.add_column("Price", justify="right")
            qt.add_column("1Y Ret", justify="right")
            qt.add_column("MaxDD", justify="right")
            qt.add_column("Sharpe", justify="right")
            qt.add_column("Vol", justify="right")
            qt.add_column("Score", justify="right")
            qt.add_column("Verdict", justify="center")

            for i, s in enumerate(scores, 1):
                if "error" in s:
                    continue
                verdict = s.get("verdict", "")
                v_style = {"STRONG": "bold green", "OK": "yellow", "WEAK": "dim", "AVOID": "bold red"}.get(verdict, "")

                qt.add_row(
                    str(i),
                    s.get("ticker", ""),
                    f"${s.get('price', 0):.2f}",
                    f"{s.get('return_1y', 0):+.0%}",
                    f"{s.get('max_drawdown', 0):+.0%}",
                    f"{s.get('sharpe', 0):.2f}",
                    f"{s.get('volatility', 0):.0%}",
                    f"{s.get('quant_score', 0):.1f}",
                    f"[{v_style}]{verdict}[/]",
                )
            console.print(qt)
    else:
        console.print("\n[dim]Run with --full for quant scores[/]")
