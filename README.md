# crypto-algo-bot 🤖

Systematic futures trading bot (Python + ccxt). Exchange-agnostic: OKX Europe (X-Perps), OKX, Bitget, Hyperliquid (experimental).

## Research summary

Two strategies have been evaluated with a look-ahead-free engine, realistic costs (fees, slippage,
historical funding), walk-forward validation, Deflated Sharpe Ratios and a one-time final hold-out.
Full method, numbers and limitations: [RESEARCH.md](RESEARCH.md).

| Study | Verdict |
|---|---|
| ETH Donchian breakout, 1h ([Part 1](RESEARCH.md#part-1-eth-donchian-breakout-1h)) | **Concluded: no edge after costs.** −69 % on the research period; about 0 % even without any costs |
| Cross-asset time-series momentum, daily portfolio ([Part 2](RESEARCH.md#part-2-cross-asset-time-series-momentum-daily-portfolio), [preregistration](docs/preregistration.md)) | **No demonstrated edge** under the preregistered rule (DSR 0.86 < 0.95; hold-out Sharpe −0.09). The evidence supports a drawdown-reducing overlay, not demonstrated alpha. Forward paper trading as `tsmom-v1` |

Nothing in this repository is a performance claim beyond these documented, reproducible runs.

## Features
- Strategies: Donchian breakout, EMA crossover, RSI mean reversion (pluggable via `BaseStrategy`)
- ATR-based stop loss, trailing stop, higher-timeframe trend filter ("guardian")
- Risk-based position sizing + portfolio risk budget across all open positions
- Backtest engine without look-ahead (signal on a bar's close, fill at the next bar's open), fees,
  slippage and **historical funding** applied at the real funding timestamps
- Research pipeline: **walk-forward optimisation** (rolling and anchored), a **final hold-out** that is
  only read on request and logged, **Probabilistic / Deflated Sharpe Ratio**, **BTC benchmarks**
  (buy & hold, vol-matched) and an append-only **research log** of every run
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
pip install -r requirements-dev.txt        # runtime deps + pytest
cp .env.example .env                       # add your keys (not needed for backtests)
pytest -m "not slow" -q                    # fast suite (what CI runs); pytest tests/ -v runs all
python main.py --mode backtest             # backtest first
python scripts/list_markets.py ETH         # find the venue's symbol for config/pairs.yaml
python main.py --mode paper                # then paper trading
python main.py --mode demo                 # then the exchange demo account (if available)
python main.py --mode live                 # only after weeks of clean paper/demo results
```

## Workflow

1. Backtest `python main.py --mode backtest` (configured parameters, research period only)
2. Costs `python main.py --mode costs` (gross vs net, turnover, break-even cost, cost scenarios)
3. Optimize `python main.py --mode optimize` (walk-forward optimisation on the research period)
4. Final hold-out, **once**: `python main.py --mode optimize --final-holdout`
5. Paper `python main.py --mode paper` (4-8 weeks, compare with the backtest)
6. Demo `python main.py --mode demo` (tests the real order path: SL, trailing, reduce-only exits)
7. Live with a very small amount first

Research results, method and limitations: see [RESEARCH.md](RESEARCH.md).

Portfolio study (cross-asset time-series momentum, daily, point-in-time universe of liquid perps):
`python main.py --mode tsmom` (research period) and, once, `--final-holdout`. The design and trial
count are frozen in [docs/preregistration.md](docs/preregistration.md), parameters in
`config/tsmom.yaml`. Research only: this strategy is not wired into the live bot.

**Forward paper trading (`tsmom-v1`).** The frozen configuration (preregistration Amendment 2) runs
daily via GitHub Actions (`.github/workflows/paper-tsmom.yml`, 00:20 UTC) on public OKX Europe
X-Perp data. Data, state, `summary.json`, `report.md` and a daily `log.jsonl` are committed to the
branch [`paper-state`](../../tree/paper-state), never to main. The track record starts at the open of
2026-11-25; quarterly reports, formal verdict after 12 months. No orders are sent anywhere.

## OKX Europe X-Perps

X-Perps are listed by the EEA API (`eea.okx.com`) as instType **FUTURES** with `ruleType: xperp`
(instId e.g. `ETH-USD_UM_XPERP-310404`, settle USD, 0.001 ETH per contract, lot = 1 contract).
ccxt parses them (symbol `ETH/USD:USD-310404`) once futures markets are loaded, which the broker does
for `exchange: myokx`, `market_type: future`. `python scripts/list_markets.py ETH` lists them. The
demo environment uses different instIds (e.g. `-310328`); resolve symbols with
`execution.broker.xperp_symbol()`. Funding settles every 8 hours (RESEARCH.md, Appendix A).
`scripts/verify_demo_stops.py` checks the fill-based stop move in simulated trading only.

## Research methodology

**Data.** Binance USD-M perpetual candles and realised funding rates are the price-history proxy for
the trading venue. `config/settings.yaml` fixes the window (`backtest.start_date` / `end_date`), so runs
are reproducible; downloads are cached in `data/cache/` and fixed past windows are never refreshed. The
fetcher uses ccxt (Binance REST) and falls back to the official archive `data.binance.vision`
(SHA-256 verified) where the REST API is not reachable.

**Backtest timing and costs.** Timestamps are candle open times. A signal computed on the close of bar
*t* is filled at the open of bar *t+1*. As in the live bot, an opposite signal only closes a position.
Commission (0.05 % per side, OKX Europe X-Perps taker) and slippage apply per side; funding is settled
at each funding timestamp against the position held at that moment (longs pay positive rates).
Without a funding history the engine falls back to the constant `funding_rate_8h` and logs a
warning. Trade PnL is net of both fees and funding. Strategies get raw OHLCV plus the guardian
columns, exactly like the live bot.

**Stops = what the exchange does.** Stop loss, trailing stop and take profit are simulated intrabar
from the entry bar on by `risk/stops.bar_stop_check`, shared by the backtest and the paper broker:
the stop in force is checked first (a gap fills at the open), the trailing stop ratchets with the
favourable extreme and fires inside the bar if the adverse extreme reaches it. When the order of high
and low matters, the backtest replays that bar over 1-minute candles (`backtest.intrabar_timeframe`);
without them the adverse order is assumed. Live entries set the stop from the actual fill price
(fill -/+ ATR stop distance) and size with the strategy's own ATR.

**Metrics.** Sharpe/Sortino/volatility are annualised with the bar timeframe (1h: sqrt(8760), 365-day
year, risk-free rate `backtest.risk_free_rate`, default 0). CAGR uses calendar time.

**Walk-forward optimisation** (`optimize:` in the config): for each window the parameter grid is
searched on the in-sample part only (selection by IS Sharpe, minimum trades, drawdown limit), the best
parameters are traded on the next out-of-sample part, and the OOS pieces are compounded into one
equity curve. Rolling (fixed 12-month IS) and anchored (expanding IS) variants, 3-month OOS steps.
The report shows the chosen parameters per window (stability), IS Deflated Sharpe Ratio per window,
and PSR / DSR of the stitched OOS curve (Bailey & Lopez de Prado). `BacktestEngine.split_consistency_check()`
(formerly `walk_forward()`) only re-runs *fixed* parameters on blocks of data and is not an OOS test.

**Final hold-out.** Data from `backtest.holdout_start` on is cut off in backtest and optimize unless
`--final-holdout` is passed. The hold-out model is fitted on the last IS window before the hold-out
with the same settings, frozen, and traded once. Every hold-out evaluation is logged; a repeated one
raises a `HoldoutReuseWarning`.

**Benchmarks.** Every report compares the strategy with BTC buy & hold, BTC scaled to the strategy's
volatility, and the traded asset's buy & hold over the identical bars: return, CAGR, Sharpe, max
drawdown, Calmar, excess return and information ratio.

**Cost report.** `--mode costs` decomposes a run into gross PnL at mid prices, slippage, fees and
funding, reports turnover and the break-even cost per round trip, and re-runs it with no costs, fees
only, fees + slippage, the configured costs and 2x stressed costs.

**Research log.** Every backtest, cost report, grid search, walk-forward study and hold-out
evaluation is appended to `logs/research_log.jsonl` with parameters, period, metrics, number of
trials and git commit.

## Project structure

```
crypto-algo-bot/
├── main.py                      Entry point (--mode paper|demo|live|backtest|optimize)
├── config/
│   ├── settings.yaml            Exchange, strategy, risk, backtest parameters
│   └── pairs.yaml               Active pairs (venue symbol + Binance data symbol)
├── data/fetcher.py              Binance history + funding (ccxt, archive fallback, cached parquet)
├── data/universe.py             OKX X-Perp snapshot, Binance symbol inventory, daily panels
├── docs/preregistration.md      Frozen design of the TSMOM study (written before any run)
├── features/
│   ├── engineer.py              30+ technical features (exploration; not used by the strategies)
│   ├── macro.py                 Higher-timeframe trend filter (no look-ahead)
│   └── timeframes.py            ccxt timeframe -> Timedelta
├── strategies/                  base_strategy, donchian_breakout, ema_crossover, rsi_mean_reversion, registry
├── backtesting/
│   ├── engine.py                Event-driven backtest (next-open fills, funding, stops)
│   ├── metrics.py               Metrics with timeframe-aware annualisation
│   ├── optimizer.py             Grid search, walk-forward optimisation
│   ├── stats.py                 Probabilistic / Deflated Sharpe Ratio
│   ├── costs.py                 Gross vs net cost report, cost scenarios
│   ├── benchmark.py             BTC buy & hold, vol-matched BTC, information ratio
│   ├── research.py              Pipeline: data + hold-out guard, backtest/optimize/hold-out reports
│   ├── portfolio.py             Multi-asset daily portfolio engine (TSMOM), point-in-time membership
│   ├── tsmom_research.py        TSMOM study: variants, WFO, DSR (N = 7), benchmarks, hold-out
│   └── research_log.py          Append-only JSONL research log
├── execution/
│   ├── broker.py                Exchange-agnostic ccxt broker (paper / demo / live)
│   └── order_manager.py         Persistent positions + exchange sync (SQLite)
├── risk/
│   ├── position_sizer.py        ATR-based, risk-based sizing
│   ├── stops.py                 Intrabar stop / trailing / TP simulation (shared by backtest and paper)
│   └── guard.py                 Kill switch, position slots, portfolio risk budget
├── notify/telegram_bot.py       Telegram alerts
├── paper/tsmom_paper.py         Daily forward paper run of tsmom-v1 (GitHub Actions)
├── scripts/
│   ├── list_markets.py          Find the right derivative symbol on the venue (incl. X-Perps)
│   ├── compare_funding.py       OKX X-Perp vs Binance funding (public data)
│   ├── power_analysis.py        Sharpe standard errors / years needed (RESEARCH.md P2.5)
│   ├── verify_demo_stops.py     Simulated-trading check of the fill-based stop move
│   └── audit_holdout_bar.py     One-time audit of the ETH hold-out (already run, logged)
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

Then register it in `strategies/registry.py`, set `strategy: my_strategy` in `config/settings.yaml`, and
add its parameter grid under `optimize.param_grid`. Signals must only use closed bars:
`tests/test_lookahead.py` checks that signals on `df[:n]` equal the full-history signals.

## Tax export

```python
from execution.order_manager import OrderManager
om = OrderManager()
om.export_tax_csv(year=2026)   # -> logs/tax_export_2026.csv
print(om.get_summary(year=2026, mode="live"))
```
