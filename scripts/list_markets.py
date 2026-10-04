"""
Lists the derivative markets of the configured exchange for one base asset.
Use it to find the exact symbol for config/pairs.yaml (e.g. the OKX Europe X-Perp).

  python scripts/list_markets.py ETH
"""
import sys
import yaml
import ccxt

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

base = sys.argv[1].upper() if len(sys.argv) > 1 else "ETH"
exchange = getattr(ccxt, config["exchange"])({"enableRateLimit": True})
exchange.load_markets()

print(f"{config['exchange']} derivative markets for {base}:\n")
for m in exchange.markets.values():
    if m["base"] == base and (m.get("swap") or m.get("future")):
        print(f"{m['symbol']:<34} id={m['id']:<30} contractSize={m.get('contractSize')} "
              f"linear={m.get('linear')} expiry={m.get('expiryDatetime')} "
              f"ruleType={m['info'].get('ruleType')}")
