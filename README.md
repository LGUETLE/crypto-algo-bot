# crypto-algo-bot 🤖

Systematic futures trading bot (Python + ccxt). Exchange-agnostic: OKX Europe (X-Perps), OKX, Bitget, Hyperliquid (experimental).

## Features
- Strategies: Donchian breakout, EMA crossover, RSI mean reversion (pluggable via `BaseStrategy`)
- ATR-based stop loss, trailing stop, higher-timeframe trend filter ("guardian")
- Risk-based position sizing + portfolio risk budget across all open positions
- Backtest engine with fees, slippage, funding, walk-forward test and IS/OOS optimizer
- Three run modes: `paper` (simulated), `demo` (exchange demo account), `live`
- Exchange-side stop loss on every entry, reconciliation with the exchange, orphan detection
- Persistent state in SQLite (restart-safe), kill switch on daily loss, Telegram alerts
- Tax export CSV

---

## Quick start

```bash
git clone https://github.com/LGUETLE/crypto-algo-bot.git
cd crypto-algo-bot
# requires Python 3.12+ (pandas-ta 0.4.x)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # add your keys
pytest tests/ -v                           # everything must be green
python main.py --mode backtest             # backtest first
python scripts/list_markets.py ETH         # find the venue's symbol for config/pairs.yaml
python main.py --mode paper                # then paper trading
python main.py --mode demo                 # then the exchange demo account (if available)
python main.py --mode live                 # only after weeks of clean paper/demo results
```

## Workflow

1. Backtest `python main.py --mode backtest`
2. Optimize `python main.py --mode optimize` (keep a final hold-out period untouched)
3. Paper `python main.py --mode paper` (4-8 weeks, compare with the backtest)
4. Demo `python main.py --mode demo` (tests the real order path: SL, trailing, reduce-only exits)
5. Live with a very small amount first

## Project structure

```
crypto-algo-bot/
├── main.py                      Entry point (--mode paper|demo|live|backtest|optimize)
├── config/
│   ├── settings.yaml            Exchange, strategy, risk, backtest parameters
│   └── pairs.yaml               Active pairs (venue symbol + Binance data symbol)
├── data/fetcher.py              Binance history (cached parquet) + live data
├── features/
│   ├── engineer.py              30+ technical features
│   └── macro.py                 Higher-timeframe trend filter (no look-ahead)
├── strategies/                  base_strategy, donchian_breakout, ema_crossover, rsi_mean_reversion
├── backtesting/                 engine, metrics, optimizer
├── execution/
│   ├── broker.py                Exchange-agnostic ccxt broker (paper / demo / live)
│   └── order_manager.py         Persistent positions + exchange sync (SQLite)
├── risk/
│   ├── position_sizer.py        ATR-based, risk-based sizing
│   ├── stops.py                 Trailing stop logic (shared by backtest and live)
│   └── guard.py                 Kill switch, position slots, portfolio risk budget
├── notify/telegram_bot.py       Telegram alerts
├── scripts/list_markets.py      Find the right derivative symbol on the venue
└── tests/                       pytest suite
```

## Risk rules

- Risk per trade: `max_risk_per_trade` of equity, measured to the stop
- Portfolio cap: `max_total_risk_pct` = sum of all open stop-out losses (crypto assets are strongly correlated)
- Kill switch: no new entries after `max_daily_loss_pct` daily loss; open positions keep being managed
- API keys without withdrawal permission, isolated margin where available, max 3-5x leverage
- Always backtest -> paper -> demo -> small live

## Adding a strategy

```python
# strategies/my_strategy.py
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def add_indicators(self, df): ...
    def generate_signals(self, df): ...   # must create df["signal"] (1 / -1 / 0); optional long_exit / short_exit
```

Then register it in `get_strategy()` in `main.py` and set `strategy: my_strategy` in `config/settings.yaml`.

## Tax export

```python
from execution.order_manager import OrderManager
om = OrderManager()
om.export_tax_csv(year=2026)   # -> logs/tax_export_2026.csv
print(om.get_summary(year=2026, mode="live"))
```
