# Research report

| Study | Status |
|---|---|
| [Part 1: ETH Donchian breakout, 1h](#part-1-eth-donchian-breakout-1h) | **Concluded: no edge after costs (and none before costs)** |
| [Part 2: Cross-asset time-series momentum, daily portfolio](#part-2-cross-asset-time-series-momentum-daily-portfolio) | **No demonstrated edge** under the preregistered rule (DSR 0.86 < 0.95; hold-out Sharpe −0.09), but positive research-period results and a much smaller loss than all benchmarks in the falling hold-out year |

All numbers in this file come from the runs listed in each part's reproducibility section, stated
with command, commit and date (UTC). Capital is 10,000 USDT in every run.

---

# Part 1: ETH Donchian breakout (1h)

**Result in one line:** the strategy has no edge. With the reference stop model (1-minute replay),
the configured parameters return −68.9 % over the research period and +0.4 % even with zero trading
costs. The walk-forward-optimised version loses −62.8 % out of sample (PSR 0.03, DSR 0.00). The
single hold-out evaluation lost −30.1 %.

## 1. Hypothesis

Hourly Donchian channel breakouts on ETH perpetual futures capture trend continuation with a positive
expectancy **after** realistic costs. The entries are filtered by:

- trend strength (ADX),
- volatility regime (ATR above its 20-bar average),
- volume,
- a daily trend filter ("guardian").

Risk management: each trade risks 2 % of equity to an ATR stop, plus a trailing stop (arms at +1.5 %,
trails 0.8 %).

Null hypothesis: the true Sharpe ratio after costs is <= 0.

## 2. Data

| Item | Value |
|---|---|
| Instrument | ETH/USDT USD-M perpetual (Binance), a price proxy for the OKX Europe X-Perp that is actually traded |
| Bars | 1h, timestamps = candle open time (UTC) |
| Intrabar | 1m candles of the same instrument, used to replay stops when the high/low order matters |
| Guardian | 1d candles; a daily candle becomes visible only after it has closed |
| Funding | realised Binance funding rates, 8-hourly. **OKX Europe X-Perps settle funding hourly**, so Binance funding is a proxy for the level, not the schedule |
| Benchmark | BTC/USDT USD-M perpetual, 1h, same bars |
| Window | 2021-01-01 to 2026-11-01 (end exclusive) |
| Research period | 2021-01-01 to 2025-11-01 |
| Final hold-out | 2025-11-01 to 2026-11-01 (12 months) |
| Source | `data.binance.vision`, Binance's public archive, SHA-256 verified. The Binance REST API returns HTTP 451 from the research machine; the archive holds the same data |

Data quality, checked 2026-11-24:

- ETH and BTC 1h each have 51,120 candles; ETH 1m has 3,067,200 candles. These are the full calendars
  with no gaps.
- There are no NaNs and no inconsistent OHLC rows. Each 1h series has one zero-volume bar.
- Funding has 6,390 events, all exactly 8 h apart.

## 3. Cost model

| Cost | Assumption |
|---|---|
| Commission | **0.05 % per side** (OKX Europe X-Perps standard tier: 0.02 % maker / 0.05 % taker; the bot uses market orders) |
| Slippage | 0.02 % per side |
| Funding | historical rate at each funding timestamp on the position held at that moment; longs pay positive rates |

Position size is in coins: 2 % of equity at risk to a stop at 2 × ATR(14), with notional capped by
3x leverage. Leverage only changes the locked margin, not PnL.

## 4. Method

1. **Engine without look-ahead.**
   - A signal on the close of bar *t* is filled at the open of bar *t+1*. An opposite signal only
     closes a position, as in the live bot.
   - Truncation tests show that signals on `df[:n]` equal the full-history signals for every bar
     `< n`, for all strategies and the guardian merge.
   - The live loop decides on closed candles only. Signals on a 299-candle live window equal the
     full-history signals.
   - An independent code review found no look-ahead and no hold-out leakage.
2. **Stops are modelled like the exchange executes them** (`risk/stops.bar_stop_check`, shared by the
   backtest and the paper broker):
   - The stop in force is checked first; a gap through it fills at the open.
   - The trailing stop ratchets with the favourable extreme and fires inside the bar if the
     adverse extreme reaches it.
   - When the high/low order of a 1h bar matters, the backtest replays that hour over its 1m candles.
   - The live bot sets the stop from the actual fill price and sizes with the strategy's own ATR.
3. **Walk-forward optimisation (WFO)** on the research period.
   - Windows: 12-month in-sample (IS), 3-month out-of-sample (OOS), step 3 months. The rolling
     variant is primary; the anchored variant is a robustness check.
   - Grid (108 combinations): `dc_enter` {20, 30, 40, 50} × `dc_exit` {5, 10, 15} ×
     `adx_threshold` {20, 25, 30} × `atr_min_mult` {0.3, 0.5, 0.7}.
   - Selection: the highest IS Sharpe among combinations with >= 10 trades and an IS max drawdown
     better than −50 %.
   - OOS pieces are compounded, and open positions are carried across windows.
   - The grid and rules were fixed in `config/settings.yaml` before the first real-data run.
4. **Statistics.** I use PSR(SR > 0) and the Deflated Sharpe Ratio (Bailey & López de Prado 2014).
   - Per IS window: N = 108 combinations, with V = variance of their Sharpe ratios.
   - Stitched OOS: N = number of distinct WFO configurations ever run on this data. This is 6 here:
     2 per cost/stop model (see 5.4). V = the mean IS trial-Sharpe variance.
5. **Final hold-out**, evaluated once with a pre-registered procedure: refit the grid with the rolling
   procedure on the last 12 months before the hold-out, freeze the parameters, and trade the
   hold-out.
6. **Benchmarks**, over identical bars:
   - BTC buy & hold,
   - BTC scaled to the strategy's volatility (ex-post scale, no costs),
   - ETH buy & hold.

   None of the benchmarks pays funding.

## 5. Results: research period (2021-01-01 to 2025-10-31)

Unless stated otherwise, these come from runs at commit `65091f7` on 2026-11-24 (1m stop replay,
0.05 % fees).

### 5.1 Configured parameters (dc 40/5, ADX 25, ATR ratio 0.7), no fitting

| | Return | CAGR | Sharpe | Max DD | Calmar | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Strategy | −68.9 % | −21.5 % | −1.00 | −77.5 % | −0.28 | 994 |
| BTC buy & hold | +249.1 % | +29.5 % | 0.73 | −77.2 % | 0.38 | |
| BTC vol-matched (x0.36) | +92.4 % | +14.5 % | 0.73 | −38.2 % | 0.38 | |
| ETH buy & hold | +404.1 % | +39.8 % | 0.82 | −81.4 % | 0.49 | |

Further detail on the strategy:

- Information ratio vs BTC buy & hold: −1.03.
- Win rate 58.7 %, profit factor 0.81.
- Exits: 582 trailing stop, 224 stop loss, 188 signal.

### 5.2 Cost report (`python main.py --mode costs --symbol ETH`)

| Component | USDT | % of start capital |
|---|---:|---:|
| Gross PnL at mid prices | −520 | −5.2 % |
| − slippage | 1,736 | 17.4 % |
| − fees | 4,341 | 43.4 % |
| + funding | −291 | −2.9 % |
| **= Net PnL** | **−6,889** | **−68.9 %** |

The strategy trades 994 round trips, an average notional of 4,367 USDT per trade and a turnover of
348 × equity per year. The break-even cost per round trip is **−0.019 %**, against the assumed
0.14 %: the signal loses money before any trading cost.

| Scenario | Return | Sharpe | Profit factor |
|---|---:|---:|---:|
| No costs | +0.4 % | 0.11 | 1.00 |
| Fees only | −58.9 % | −0.74 | 0.85 |
| Fees + slippage | −67.6 % | −0.96 | 0.82 |
| Configured (+ funding) | −68.9 % | −1.00 | 0.81 |
| Stressed (2x fees + slippage) | −90.8 % | −2.15 | 0.66 |

### 5.3 Walk-forward optimisation, stitched OOS (2022-01-01 to 2025-10-31)

| | Return | CAGR | Sharpe | Max DD | Trades | PSR(SR>0) | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rolling (primary) | −62.8 % | −22.7 % | −0.98 | −70.6 % | 781 | 0.029 | 0.000 |
| Anchored | −46.1 % | −14.9 % | −1.05 | −48.2 % | 330 | 0.022 | 0.001 |
| BTC buy & hold | +136.8 % | +25.2 % | 0.69 | −67.4 % | | | |
| BTC vol-matched to rolling (x0.45) | +68.2 % | +14.5 % | 0.69 | −37.8 % | | | |
| ETH buy & hold | +4.5 % | +1.2 % | 0.36 | −76.7 % | | | |

Information ratio of the rolling WFO: −1.03 vs BTC buy & hold and −1.18 vs vol-matched BTC.

Rolling WFO, chosen parameters per window:

| # | OOS window | dc_enter | dc_exit | ADX | ATR ratio | IS Sharpe | IS DSR | OOS return | OOS Sharpe | Trades |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2022-01 – 2022-03 | 50 | 15 | 20 | 0.7 | −0.40 | 0.04 | −5.6 % | −0.82 | 66 |
| 2 | 2022-04 – 2022-06 | 50 | 15 | 20 | 0.7 | −0.67 | 0.03 | −11.3 % | −1.81 | 83 |
| 3 | 2022-07 – 2022-09 | 30 | 15 | 25 | 0.5 | −0.36 | 0.04 | +0.1 % | 0.12 | 74 |
| 4 | 2022-10 – 2022-12 | 50 | 15 | 25 | 0.5 | −0.38 | 0.01 | −26.1 % | −4.25 | 48 |
| 5 | 2023-01 – 2023-03 | 30 | 15 | 25 | 0.5 | −1.45 | 0.00 | −15.6 % | −2.75 | 48 |
| 6 | 2023-04 – 2023-06 | 50 | 15 | 25 | 0.7 | −1.75 | 0.00 | −7.7 % | −1.24 | 34 |
| 7 | 2023-07 – 2023-09 | 50 | 10 | 30 | 0.7 | −1.49 | 0.00 | −11.4 % | −2.53 | 21 |
| 8 | 2023-10 – 2023-12 | 50 | 10 | 30 | 0.7 | −1.96 | 0.00 | −4.0 % | −0.69 | 26 |
| 9 | 2024-01 – 2024-03 | 50 | 10 | 30 | 0.3 | −1.24 | 0.00 | −3.4 % | −0.49 | 39 |
| 10 | 2024-04 – 2024-06 | 50 | 10 | 25 | 0.3 | −0.86 | 0.00 | −12.3 % | −2.19 | 48 |
| 11 | 2024-07 – 2024-09 | 50 | 15 | 20 | 0.7 | −1.15 | 0.00 | +2.3 % | 0.47 | 60 |
| 12 | 2024-10 – 2024-12 | 50 | 10 | 25 | 0.3 | −0.06 | 0.06 | −5.8 % | −0.91 | 47 |
| 13 | 2025-01 – 2025-03 | 50 | 10 | 25 | 0.5 | 0.19 | 0.10 | +8.6 % | 1.67 | 42 |
| 14 | 2025-04 – 2025-06 | 30 | 10 | 25 | 0.7 | 0.27 | 0.23 | −3.3 % | −0.47 | 62 |
| 15 | 2025-07 – 2025-09 | 40 | 5 | 25 | 0.3 | 1.00 | 0.15 | +12.9 % | 2.13 | 55 |
| 16 | 2025-10 (1 month) | 30 | 5 | 25 | 0.5 | 1.45 | 0.39 | −6.3 % | −2.94 | 28 |

Observations:

- The best IS combination had a negative IS Sharpe in 12 of 16 windows, and no IS selection is
  significant after deflation (maximum IS DSR 0.39).
- Parameters are unstable: 11 distinct sets in 16 windows.
- **`atr_min_mult` has no effect:** every window has 36 distinct outcomes out of 108 combinations,
  so the chosen ATR-ratio value is an arbitrary tie-break.
- **Anchored:** from window 6 on, no combination met the −50 % IS drawdown limit, so the variant
  stayed flat. A fixed drawdown cap on an expanding window is a design flaw of that pre-registered
  rule. It is reported, not changed.

### 5.4 Sensitivity to the stop model (configured parameters, research period)

| Stop model | Commit | Fees | Return | Return with no costs | Rolling WFO OOS |
|---|---|---:|---:|---:|---:|
| Trailing stop checked from the next bar on (original engine) | `98c2258` / `aaa6225` | 0.06 % | −66.0 % | +5.3 % | −69.6 % |
| Adverse order on the full 1h bar | `5b43d54` | 0.05 % | +50.6 % | +479.1 % | +7.2 % |
| **1m replay where the order matters (reference)** | `65091f7` | 0.05 % | **−68.9 %** | **+0.4 %** | **−62.8 %** |

The hourly adverse rule exits at the freshly ratcheted trail whenever the bar's low reaches it. On
rising bars whose low was the open, this books a near-high exit that usually never happened. A
result that changes sign with the intrabar assumption is not usable. Replaying those hours over 1m
candles settles it: no edge.

## 6. Final hold-out (2025-11-01 to 2026-10-31), evaluated once

- **Run:** 2026-11-24T00:26:58Z, commit `98c2258` (original engine, 0.06 % fees, trailing stop
  checked from the next bar on), `python main.py --mode optimize --symbol ETH --final-holdout`.
- **Fit:** rolling procedure on [2024-11-01, 2025-11-01). It selected `dc_enter 30, dc_exit 5,
  adx_threshold 25, atr_min_mult 0.3`, with an IS Sharpe of 0.66.

| | Return | Sharpe | Max DD | Calmar | Vol | Excess return | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strategy | −30.1 % | −1.18 | −37.4 % | −0.81 | 27.3 % | | |
| BTC buy & hold | −29.4 % | −0.59 | −47.6 % | −0.62 | 43.1 % | −0.8 % | −0.13 |
| BTC vol-matched (x0.63) | −17.9 % | −0.59 | −32.5 % | −0.55 | 27.3 % | −12.2 % | −0.43 |
| ETH buy & hold | −41.5 % | −0.62 | −61.1 % | −0.68 | 58.7 % | +11.4 % | 0.07 |

Strategy detail:

- 164 trades, profit factor 0.79, fees 1,972.91 USDT, funding −35.82 USDT.
- Price PnL including slippage is about −1,003 USDT, derived as net PnL (−3,011.76) + fees −
  funding.
- PSR(SR > 0) = 0.139, computed from the logged return moments.

**Audit of the largest bar** (owner-authorised, one time; logged as hold-out evaluation #2 with
`prior_evaluations = 1`):

- Run on 2026-11-24 at commit `66606fa` with `scripts/audit_holdout_bar.py`; no parameter changed.
- The +13.2 % equity bar at 2026-08-19 15:00 UTC is a real market move: ETH +5.9 % and BTC +4.0 % in
  the same hour on ~6x volume. The strategy held a 2.2x-notional long, which the trailing stop
  closed at the next open.
- No data error, so nothing was re-run.

**Not re-evaluated.** The hold-out was evaluated before the stop-model, fee and walk-forward changes
of section 5.4. Re-running it with the new engine would be a third look at the same data, so the
hold-out is considered spent. The conclusion does not depend on it: under the reference stop model
the research period shows no edge even before costs.

## 7. Conclusion: concluded, no edge after costs

| Test | Result |
|---|---|
| Fixed parameters, research period | −68.9 % (no costs: +0.4 %) |
| Rolling WFO, OOS | −62.8 %, PSR 0.03, DSR 0.00 |
| Hold-out (original engine) | −30.1 % |

Passive BTC at the same volatility dominated in every period. The ETH Donchian 1h strategy is
**concluded: no edge after costs**, and its gross edge is indistinguishable from zero. It should not
be traded.

Earlier "validated" results in this repository came from an engine with look-ahead fills and
leveraged PnL. For example, the comment in `config/pairs.yaml` mentioned an OOS Sharpe of 1.37. Those
results are void.

## 8. Limitations

- **Hold-out contamination is possible.** The strategy design, grid ranges and fixed settings
  existed on 2026-10-03, with a lookback that overlapped the hold-out. Contamination would bias
  results upward, so it cannot explain the negative result.
- **Intrabar order within one minute** still uses the adverse rule. Its effect is small at 1m
  relative to a 0.8 % trailing distance.
- **Proxy data.** Prices and funding are from Binance; the bot trades OKX Europe X-Perps, whose
  funding settles hourly.
- **Not modelled:** the daily-loss kill switch and the portfolio risk cap (with one pair and 2 % risk
  per trade the cap never binds). Minimum order sizes, partial fills and liquidation are also not
  modelled.
- **Benchmarks** pay no funding. The vol-matched line uses an ex-post scale and has no costs.
- **DSR trial count** comes from the local, gitignored research log. Trials made before this study or
  elsewhere are not included.

## 9. Reproducibility

Config: `config/settings.yaml` and `config/pairs.yaml` at the stated commit, with ETH as the only
active pair. Every run is in `logs/research_log.jsonl`.

| What | Command | Commit | Time (UTC) |
|---|---|---|---|
| 5.1 backtest | `python main.py --mode backtest --symbol ETH` | `65091f7` | batch started 2026-11-24 01:04, run in this order |
| 5.2 cost report | `python main.py --mode costs --symbol ETH` | `65091f7` | same batch |
| 5.3 WFO | `python main.py --mode optimize --symbol ETH` | `65091f7` | same batch |
| 5.4 original engine | same commands, plus the removed `scripts/cost_decomposition.py` | `98c2258`, `aaa6225` | 2026-11-24 00:20–00:28 |
| 5.4 hourly adverse rule | backtest / costs / optimize | `5b43d54` | batch started 2026-11-24 00:56 |
| 6 hold-out | `python main.py --mode optimize --symbol ETH --final-holdout` | `98c2258` | 2026-11-24 00:27 |
| 6 audit | `python scripts/audit_holdout_bar.py` | `66606fa` (script committed unchanged as `bda5f86`) | 2026-11-24 00:50 |

---

# Part 2: Cross-asset time-series momentum (daily, portfolio)

Design, parameters, variants, trial count and decision rule:
[docs/preregistration.md](docs/preregistration.md). The preregistration was committed as `9d608ac`
(2026-11-24 01:12 UTC), **before any real-data run**; it has no amendments.

## P2.1 Setup (summary of the preregistration)

| Item | Value |
|---|---|
| Universe | Point-in-time: Binance USD-M USDT perps, ranked by trailing 30-day quote volume, 12 members (stay while rank <= 15), >= 90 days of history |
| Primary candidates | Coin is a crypto X-Perp on OKX Europe today (snapshot `config/okx_xperps_snapshot.json`, 168 coins, retrieved 2026-11-24) **or** delisted from Binance (kept to avoid survivorship bias). 180 candidates, 57 were members at some point, among them LUNA, MATIC, TOMO and SXP |
| Signal | Time-series momentum over 7/14/28/56 days: `sign` or `volscaled`; `long_flat` or `long_short` |
| Sizing | Inverse volatility, 30 % p.a. portfolio vol target (60-day covariance), 25 % cap per asset, 2x gross cap, 0.05 no-trade band; daily rebalance at the open after the signal close |
| Costs | 0.05 % taker per side; slippage 0.02 % (BTC, ETH) / 0.05 % (others) per side; historical Binance funding (OKX X-Perps settle hourly, so Binance is a proxy) |
| Trials | N = 7: V1-V4 fixed variants, W1 rolling WFO, W2 anchored WFO (both choose among V1-V4 by IS Sharpe, IS 12 m / OOS 3 m), R1 = W1 on the Binance-wide universe |
| Decision rule | Edge only if W1 has PSR >= 0.95 **and** DSR >= 0.95 on the stitched OOS, **and** a hold-out Sharpe > 0 |

## P2.2 Results: research period

Fixed variants V1-V4 run 2021-04-01 to 2025-10-31 (from the first day with a full universe).
WFO procedures run on their stitched OOS, 2022-01-01 to 2025-10-31. Command
`python main.py --mode tsmom`, commit `aee0410`, run 2026-11-24 01:48 UTC.

| Config | Return | CAGR | Sharpe | Max DD | Vol | Turnover (x/yr) | Fees | Slippage | Funding | PSR | DSR (N=7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 sign / long-flat | +91.6 % | +15.2 % | 0.71 | −40.0 % | 24.1 % | 42.2 | 1,216 | 1,007 | −905 | 0.936 | 0.818 |
| V2 sign / long-short | +140.1 % | +21.0 % | 0.81 | −27.8 % | 28.7 % | 90.0 | 3,561 | 2,998 | −1,402 | 0.958 | 0.868 |
| V3 volscaled / long-flat | +12.9 % | +2.7 % | 0.23 | −53.7 % | 26.0 % | 25.3 | 540 | 464 | −864 | 0.690 | 0.453 |
| V4 volscaled / long-short | +62.4 % | +11.1 % | 0.51 | −27.8 % | 29.4 % | 55.4 | 1,787 | 1,514 | −1,493 | 0.860 | 0.681 |
| **W1 rolling WFO (primary)** | **+108.4 %** | **+21.1 %** | **0.84** | **−27.5 %** | 27.4 % | 63.2 | 1,885 | 1,601 | −986 | **0.950** | **0.861** |
| W2 anchored WFO | +77.1 % | +16.1 % | 0.65 | −27.8 % | 29.4 % | 94.4 | 2,478 | 2,091 | −838 | 0.900 | 0.765 |
| R1 W1 on Binance-wide universe | +77.5 % | +16.1 % | 0.66 | −33.0 % | 29.0 % | 64.9 | 1,928 | 1,625 | +471 | 0.903 | 0.769 |

Fees, slippage and funding are in USDT on 10,000 USDT starting capital. The DSR uses N = 7 and
V = variance of the seven per-day Sharpe ratios (1.17e-4).

Benchmarks for W1, on identical days (2022-01-01 to 2025-10-31):

| | Return | CAGR | Sharpe | Max DD | Calmar | Excess return | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 strategy | +108.4 % | +21.1 % | 0.84 | −27.5 % | 0.77 | | |
| BTC buy & hold | +137.1 % | +25.2 % | 0.69 | −66.9 % | 0.38 | −28.7 % | −0.25 |
| Equal-weight universe | −74.1 % | −29.7 % | −0.07 | −86.0 % | −0.34 | +182.5 % | 0.37 |
| BTC vol-matched (x0.53) | +79.3 % | +16.4 % | 0.69 | −41.9 % | 0.39 | +29.1 % | 0.11 |

Variants chosen per W1 window:
V1 V2 V2 V2 V2 V2 V1 V1 V1 V1 V1 V1 V1 V2 V1 V1.

The sign-based variants V1 and V2 won almost every window. The vol-scaled variants were never chosen
by W1, but R1 chose V4 in four windows.

Observations:

- The research period is encouraging but not conclusive. W1's PSR is exactly at the threshold
  (0.9504); its DSR of 0.861 is below the preregistered 0.95.
- **The decision rule is therefore already not met on the research period.**
- W1 beats vol-matched BTC only slightly (information ratio 0.11). It trails BTC buy & hold in return,
  with far smaller drawdowns.
- Survivorship: on the Binance-wide universe (R1), the same procedure earns less (Sharpe 0.66 vs
  0.84). Part of the primary result may come from the OKX-today filter.

## P2.3 Final hold-out (2025-11-01 to 2026-10-31), evaluated once

- **Run:** `python main.py --mode tsmom --final-holdout`, commit `aee0410`, hold-out evaluation
  logged 2026-11-24 02:00 UTC, `prior_evaluations = 0`. The command first re-ran the research study,
  which reproduced every number in P2.2 exactly.
- **Fit (W1 procedure):** IS window [2024-11-01, 2025-11-01). IS Sharpe ratios were V1 1.29, V2 0.77,
  V3 1.14 and V4 0.23, so **V1 (sign, long-flat)** was frozen and traded on the hold-out.

| | Return | Sharpe | Max DD | Calmar | Excess return | IR |
|---|---:|---:|---:|---:|---:|---:|
| TSMOM (V1) | −5.7 % | −0.09 | −22.4 % | −0.26 | | |
| BTC buy & hold | −29.2 % | −0.57 | −47.0 % | −0.62 | +23.5 % | 0.60 |
| Equal-weight universe | −41.1 % | −0.53 | −54.9 % | −0.75 | +35.4 % | 0.64 |
| BTC vol-matched (x0.60) | −16.7 % | −0.57 | −30.3 % | −0.55 | +11.0 % | 0.49 |

Strategy detail:

- Volatility 26.2 %, turnover 51 × equity per year.
- Fees 239, slippage 201 and funding −54 USDT; gross PnL before costs −76 USDT.
- PSR(SR > 0) = 0.46.

## P2.4 Conclusion: no demonstrated edge (decision rule not met)

| Criterion (preregistered) | Required | Result |
|---|---|---|
| W1 stitched OOS PSR | >= 0.95 | 0.950 (met, at the threshold) |
| W1 stitched OOS DSR (N = 7) | >= 0.95 | **0.861 (not met)** |
| Hold-out Sharpe | > 0 | **−0.09 (not met)** |

Under the preregistered rule, the TSMOM portfolio has **no demonstrated edge**. The evidence is
nonetheless clearly different from Part 1:

- The research-period returns are positive across all seven configurations.
- The signal is positive before costs on the research period.
- Drawdowns are much smaller than those of BTC or an equal-weight basket.
- In a falling hold-out year it lost 5.7 % while BTC lost 29 % and the universe basket 41 %.

That is the profile of a trend-following, risk-reducing overlay, not of a stand-alone return source
with demonstrated alpha. No parameter was changed after the hold-out.

## P2.5 Limitations

- **Universe.** The OKX filter can only use today's list (all X-Perps were listed in 2026).
  Binance-delisted coins are kept, but coins still on Binance that are not on OKX Europe are excluded,
  which can be optimistic. FTT, for example, was relisted on Binance and is therefore not in the
  primary universe. R1 (Binance-wide, no OKX filter) shows a lower Sharpe of 0.66 vs 0.84.
- **Funding.** Binance 8-hourly funding is a proxy for OKX's hourly funding. Within a day, all
  events are charged at the day's close price.
- **Execution.**
  - The rebalance fills at the daily open with fixed slippage, with no market impact or
    minimum-size modelling.
  - Delisted coins are closed at their last close (optimistic for coins that collapsed into a
    delisting).
  - Liquidity is ranked on `volume x close` (a quote-volume proxy).
- **Statistics.** DSR uses only the 7 preregistered configurations. Part 1 and other ideas tested on
  the same market history are not included, so the true multiple-testing burden is higher.
- **Single hold-out year.** One year of a falling market is a small sample (PSR 0.46).

## P2.6 Reproducibility

| What | Command | Commit | Time (UTC) |
|---|---|---|---|
| Preregistration | `docs/preregistration.md` | `9d608ac` | 2026-11-24 01:12 |
| OKX X-Perp snapshot | `data.universe.okx_xperp_coins(refresh=True)` -> `config/okx_xperps_snapshot.json` | `aee0410` | 2026-11-24 01:18 |
| Research study (P2.2) | `python main.py --mode tsmom` | `aee0410` | 2026-11-24 01:48 |
| Hold-out (P2.3) | `python main.py --mode tsmom --final-holdout` | `aee0410` | 2026-11-24 01:55 (hold-out logged 02:00) |
