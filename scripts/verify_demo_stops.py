"""
Demo verification of the fill-based stop move (Broker._move_stop) on OKX Europe X-Perps.
python scripts/verify_demo_stops.py [--coin ETH]

SIMULATED TRADING ONLY. Narrow, owner-approved exception to the "never run demo" rule (CLAUDE.md).
  0. skip (exit 0) if no OKX credentials are configured; the values are never printed
  1. build the broker in demo mode and ABORT unless ccxt's sandbox flag is set and every request
     carries the OKX simulated-trading header (x-simulated-trading: 1)
  2. resolve the coin's X-Perp in the demo market list; skip if X-Perps are not offered in demo
  3. open the minimum size (1 contract) with an estimate-based stop, let the broker move the stop
     to fill -/+ distance, verify: exactly one stop, at the fill-based level, old one cancelled
  4. close the position (reduce-only) and verify the account is flat with no stop orders left
Uses a temporary trade database; the bot's own state is not touched.
"""
import argparse
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIM_HEADER = ("x-simulated-trading", "1")


class Abort(Exception):
    pass


def credentials_present(exchange_id: str) -> bool:
    """True if key, secret and passphrase variables are set (after .env loading). Values unused."""
    p = exchange_id.upper()
    return all(os.getenv(f"{p}_{k}") for k in ("API_KEY", "SECRET", "PASSPHRASE"))


def assert_simulated(broker):
    ex = broker.exchange
    if broker.mode != "demo":
        raise Abort(f"broker mode is {broker.mode!r}, expected 'demo'")
    if not getattr(ex, "isSandboxModeEnabled", False):
        raise Abort("ccxt sandbox mode is not enabled")
    if (ex.headers or {}).get(SIM_HEADER[0]) != SIM_HEADER[1]:
        raise Abort(f"request header {SIM_HEADER[0]}={SIM_HEADER[1]} is missing")


def _stops(ex, symbol):
    out = []
    for params in ({"stop": True}, {"ordType": "conditional"}):
        try:
            out += ex.fetch_open_orders(symbol, params=params)
        except Exception:                                # noqa: BLE001 - variant not supported
            pass
    seen, uniq = set(), []
    for o in out:
        if o["id"] not in seen:
            seen.add(o["id"])
            uniq.append(o)
    return uniq


def run(coin: str = "ETH", stop_dist_pct: float = 0.05) -> int:
    from dotenv import load_dotenv
    load_dotenv()                                        # the bot's normal credential loading
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    if not credentials_present(config["exchange"]):
        print("SKIP: no demo credentials configured")
        return 0

    os.environ["TRADING_MODE"] = "demo"
    import execution.order_manager as om
    om.DB_PATH = os.path.join(tempfile.mkdtemp(), "verify_demo.db")
    from execution.broker import Broker, xperp_symbol

    broker = Broker(config)
    assert_simulated(broker)                             # BEFORE any request
    ex = broker.exchange
    ex.load_markets()
    assert_simulated(broker)
    symbol = xperp_symbol(ex.markets, coin)
    if symbol is None:
        print(f"SKIP: no {coin} X-Perp offered in simulated trading")
        return 0

    market = ex.market(symbol)
    coins = market["contractSize"] * max(market["limits"]["amount"]["min"] or 1, 1)
    price = float(ex.fetch_ticker(symbol)["last"])
    dist = price * stop_dist_pct
    estimate_sl = price - dist
    checks = {}
    try:
        broker.place_order(symbol=symbol, side="buy", amount=coins, sl_price=estimate_sl,
                           entry_price=price, strategy="verify_demo", sl_dist=dist)
        pos = broker.om.get_position(symbol)
        fill = pos["entry_price"]
        target = float(ex.price_to_precision(symbol, fill - dist))
        stops = _stops(ex, symbol)
        trig = [float(o.get("triggerPrice") or o.get("stopLossPrice") or 0) for o in stops]
        checks["one stop order"] = len(stops) == 1
        checks["stop at fill-based level"] = any(abs(t - target) <= market["precision"]["price"] for t in trig)
        checks["estimate-based stop cancelled"] = not any(
            abs(t - float(ex.price_to_precision(symbol, estimate_sl))) < 1e-12 for t in trig) or target == estimate_sl
        checks["stored stop = fill-based"] = abs(pos["sl_price"] - (fill - dist)) < market["precision"]["price"]
    finally:
        broker.close_position(symbol, price, "verify_demo")
        broker.cancel_trigger_orders(symbol)
        positions = [p for p in ex.fetch_positions([symbol]) if float(p.get("contracts") or 0) != 0]
        checks["flat after close"] = not positions
        checks["no stop orders left"] = not _stops(ex, symbol)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="ETH")
    a = ap.parse_args()
    try:
        sys.exit(run(a.coin))
    except Abort as e:
        print(f"ABORT: {e}")
        sys.exit(2)
