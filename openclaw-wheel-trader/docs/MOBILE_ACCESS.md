# Viewing the dashboards on your phone

All the bot dashboards are plain web pages served by Flask on the Mac, bound to
`127.0.0.1` (localhost) on purpose — they have **no login**, so they must never
be reachable from the open internet. That still leaves three good ways to get
them on your phone. **Option A (Tailscale serve) is the recommended one**: it
needs no code or config changes, keeps the localhost-only bind, and gives you
HTTPS.

The dashboards themselves are mobile-responsive (cards stack into one column,
wide tables scroll sideways inside their card), so once you can reach them,
they're usable on a phone screen.

## The dashboards

| Dashboard | Port | Started by |
|-----------|------|------------|
| Wheel trader (this repo) | 5051 | `python main.py dashboard` — kept alive by the `ai.openclaw.dashboard` LaunchAgent |
| Polybot (prediction markets) | 5050 | `python main.py dashboard` in the polybot repo |
| Kalshi 15-min trader | 5053 | `python main.py kalshi-dashboard` in the polybot repo |

Any other web UI running on the same Mac (e.g. an assistant's gateway/control
page on its own localhost port) can be exposed to your phone the same way —
just point one of the commands below at its port.

## Option A — Tailscale serve (recommended)

Tailscale puts your Mac and phone on a private WireGuard network ("tailnet").
`tailscale serve` then proxies an HTTPS URL on that tailnet to a localhost port
on the Mac. Nothing is exposed to your LAN or the internet, the dashboards keep
their `127.0.0.1` bind, and traffic is encrypted end to end.

One-time setup:

1. Install Tailscale on the Mac (`brew install --cask tailscale` or from
   tailscale.com) and sign in.
2. Install the Tailscale app on your phone (iOS/Android) and sign in to the
   **same account/tailnet**.
3. In the Tailscale admin console, enable **HTTPS certificates**
   (DNS → HTTPS Certificates → Enable). Needed once per tailnet so `serve` can
   mint TLS certs.

Then on the Mac (serve allows HTTPS on ports 443, 8443, and 10000 — three
ports, three dashboards):

```sh
tailscale serve --bg --https=443   http://127.0.0.1:5051   # wheel trader
tailscale serve --bg --https=8443  http://127.0.0.1:5050   # polybot
tailscale serve --bg --https=10000 http://127.0.0.1:5053   # kalshi 15-min
```

On your phone (with the Tailscale app connected), open:

- `https://<mac-name>.<tailnet>.ts.net` — wheel trader
- `https://<mac-name>.<tailnet>.ts.net:8443` — polybot
- `https://<mac-name>.<tailnet>.ts.net:10000` — kalshi

(`tailscale status` prints the Mac's exact name; `tailscale serve status`
shows what's being served; `tailscale serve reset` clears it. `--bg` keeps the
proxies running in the background and they survive reboots.)

Notes:

- **Do not use `tailscale funnel`** — funnel publishes to the public internet.
  These dashboards have no auth; `serve` (tailnet-only) is the correct tool.
- Don't use `--set-path /trader`-style path mounting: the dashboards fetch
  `/api/...` with absolute paths, so they only work served from the root of
  their own port.

## Option B — bind to the Tailscale IP (no serve, plain HTTP)

If you'd rather skip `serve`, bind the dashboard directly to the Mac's
Tailscale address. It's then reachable only over the tailnet (traffic is still
WireGuard-encrypted in transit), but not from your LAN:

```sh
python main.py dashboard --host "$(tailscale ip -4)"
```

Phone: `http://100.x.y.z:5051` (the same IP the command printed).

## Option C — home Wi-Fi only (last resort)

`--host 0.0.0.0` exposes the dashboard to every device on whatever networks the
Mac is on. Only do this on a Wi-Fi network you trust, and never port-forward
these ports on your router:

```sh
python main.py dashboard --host 0.0.0.0
```

Phone (same Wi-Fi): `http://<mac-lan-ip>:5051` — find the IP with
`ipconfig getifaddr en0`.

macOS will ask "Do you want python to accept incoming network connections?"
the first time you bind beyond localhost — click Allow.

## Make it feel like an app

Open the dashboard in the phone browser, then:

- **iPhone (Safari):** Share → **Add to Home Screen**
- **Android (Chrome):** ⋮ menu → **Add to Home screen**

The pages carry the standalone-web-app meta tags, so the icon launches
full-screen without browser chrome, dark themed.

## LaunchAgent note

The `ai.openclaw.dashboard` LaunchAgent (`scripts/run_dashboard.sh`) starts the
trader dashboard localhost-only — exactly what Option A wants, so with
Tailscale serve there is nothing to change. If you insist on Option B/C
permanently, add the `--host` flag to the `exec python main.py dashboard` line
in that script, understanding the exposure trade-off above.
