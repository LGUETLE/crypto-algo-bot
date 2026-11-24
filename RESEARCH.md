# Research report: ETH Donchian breakout (1h)

**Result in one line:** after costs, the strategy has no edge. This holds for the configured
parameters, for walk-forward optimised parameters out of sample, and on the untouched 12-month hold-out.
Even with zero trading costs, the configured parameters return only +5.3 % over almost five years.

All numbers in this file come from the runs listed in [section 9](#9-reproducibility) (2026-11-24 UTC,
commit `98c2258` unless stated otherwise). Capital is 10,000 USDT in every run.

---

## 1. Hypothesis

Hourly Donchian channel breakouts on ETH perpetual futures capture trend continuation with a positive
expectancy **after** realistic costs. The entries are filtered by:

- trend strength (ADX),
- volatility regime (ATR above its 20-bar average),
- volume,
- a daily trend filter ("guardian").

Risk management: each trade risks 2 % of equity to an ATR stop, plus a trailing stop.

Null hypothesis: the true Sharpe ratio after costs is <= 0.

## 2. Data

| Item | Value |
|---|---|
| Instrument | ETH/USDT USD-M perpetual (Binance). It is a price proxy for the OKX Europe X-Perp that is actually traded |
| Bars | 1h. Timestamps are candle open times (UTC) |
| Guardian | 1d candles of the same instrument. A daily candle becomes visible only after it has closed |
| Funding | Realised Binance funding rates, 8-hourly (00/08/16 UTC) |
| Benchmark | BTC/USDT USD-M perpetual, 1h, on the same bars |
| Window | 2021-01-01 to 2026-11-01 (end exclusive) |
| Research period | 2021-01-01 to 2025-11-01 |
| Final hold-out | 2025-11-01 to 2026-11-01 (12 months) |
| Source | `data.binance.vision`, Binance's public archive. Downloads are SHA-256 verified. The Binance REST API returns HTTP 451 from the machine that produced these runs; the archive holds the same klines and funding history |

Data quality, checked 2026-11-24:

- ETH and BTC 1h each have 51,120 candles. That is the full calendar, with no gaps.
- There are no NaNs and no inconsistent OHLC rows. Each series has one zero-volume bar.
- Funding has 6,390 events, all exactly 8 h apart.

## 3. Cost model

| Cost | Assumption |
|---|---|
| Commission | 0.06 % per side of notional (taker). This is `backtest.commission`; the venue's fee tier is **not verified** |
| Slippage | 0.02 % per side (`backtest.slippage`) |
| Funding | The historical rate at each funding timestamp, applied to the position held at that moment. Longs pay positive rates |
| Stops | Filled at the stop level, or at the open if the bar gaps through it |

Position size is in coins. Each trade risks 2 % of equity to a stop at 2 × ATR(14), and notional is
capped by 3x leverage. Leverage only changes the locked margin, not PnL.

## 4. Method

1. **Engine without look-ahead.**
   - A signal on the close of bar *t* is filled at the open of bar *t+1*.
   - Truncation tests show that signals computed on `df[:n]` equal the full-history signals for every
     bar `< n`. This holds for all strategies and for the daily guardian merge.
   - The live loop decides on closed candles only (tested).
   - Donchian signals computed on a 299-candle live window equal the full-history signals.
   - As in the live bot, an opposite signal only closes a position. An independent code review found
     no look-ahead and no hold-out leakage.
2. **Walk-forward optimisation (WFO)** on the research period.
   - Windows: 12-month in-sample (IS), 3-month out-of-sample (OOS), step 3 months. The rolling
     variant is primary; the anchored variant is a robustness check.
   - Grid (108 combinations): `dc_enter` {20, 30, 40, 50}, `dc_exit` {5, 10, 15},
     `adx_threshold` {20, 25, 30}, `atr_min_mult` {0.3, 0.5, 0.7}.
   - In each window the whole grid runs on the IS part only. The combination with the highest IS
     Sharpe is then traded on the next OOS part. To qualify, a combination needs at least 10 trades
     and an IS max drawdown better than −50 %.
   - The OOS pieces are compounded into one equity curve.
   - The grid and selection rules were fixed in `config/settings.yaml` before the first run on real
     data. The grid is the one that was already in the repository.
3. **Statistics.** I use the Probabilistic Sharpe Ratio, PSR(SR > 0), and the Deflated Sharpe Ratio
   (DSR) of Bailey & López de Prado (2014). DSR accounts for skewness, kurtosis, track-record length
   and the number of trials.
   - Per IS window: the DSR of the selected combination, with N = 108 and the variance of the 108
     trial Sharpe ratios.
   - For the stitched OOS curve: N = 2 (the WFO variants run on this data), with V = the mean variance
     of the IS trial Sharpe ratios.
4. **Final hold-out**, evaluated once with a pre-registered procedure:
   1. Refit the grid with the rolling procedure on the last 12 months before the hold-out.
   2. Freeze the selected parameters.
   3. Trade them over the full hold-out.
5. **Benchmarks**, over identical bars:
   - BTC buy & hold,
   - BTC scaled to the strategy's volatility (ex-post scale, no costs),
   - ETH buy & hold.

   None of the benchmarks pays funding.

## 5. Results: research period (2021-01-01 to 2025-10-31)

### 5.1 Configured parameters (dc 40/5, ADX 25, ATR ratio 0.7), no fitting

| | Return | CAGR | Sharpe | Max DD | Calmar | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Strategy | −66.0 % | −20.0 % | −0.90 | −72.9 % | −0.27 | 795 |
| BTC buy & hold | +249.0 % | +29.5 % | 0.73 | −77.2 % | 0.38 | |
| BTC vol-matched (x0.37) | +94.0 % | +14.7 % | 0.73 | −38.7 % | 0.38 | |
| ETH buy & hold | +404.0 % | +39.8 % | 0.82 | −81.4 % | 0.49 | |

Further detail on the strategy:

- Information ratio vs BTC buy & hold: −1.00.
- Win rate 51.3 %, profit factor 0.81.
- Exits: 454 trailing stop, 180 stop loss, 161 signal.

**Cost decomposition** (`scripts/cost_decomposition.py`, commit `aaa6225`):

| Run | Return | Sharpe | Price PnL (incl. slippage) | Fees | Funding |
|---|---:|---:|---:|---:|---:|
| All costs (historical funding) | −66.0 % | −0.90 | −1,487 | 4,837 | −271 |
| No fees/slippage, funding on | +1.6 % | 0.12 | +521 | 0 | −358 |
| No costs at all | +5.3 % | 0.16 | +534 | 0 | 0 |

The signal has close to zero edge before costs. About 800 trades × ~0.16 % round-trip cost consume
more than the whole gross result.

### 5.2 Walk-forward optimisation, stitched OOS (2022-01-01 to 2025-10-31)

| | Return | CAGR | Sharpe | Max DD | Trades | PSR(SR>0) | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rolling (primary) | −69.6 % | −26.7 % | −1.16 | −74.2 % | 675 | 0.013 | 0.002 |
| Anchored | −47.2 % | −15.4 % | −1.14 | −52.8 % | 258 | 0.017 | 0.004 |
| BTC buy & hold | +136.7 % | +25.2 % | 0.69 | −67.4 % | | | |
| BTC vol-matched to rolling (x0.47) | +70.5 % | +14.9 % | 0.69 | −38.8 % | | | |
| ETH buy & hold | +4.5 % | +1.1 % | 0.36 | −76.7 % | | | |

Information ratio of the rolling WFO: −1.12 vs BTC buy & hold, −1.31 vs vol-matched BTC.

Rolling WFO, chosen parameters per window:

| # | OOS window | dc_enter | dc_exit | ADX | ATR ratio | IS Sharpe | IS DSR | OOS return | OOS Sharpe | Trades |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2022-01 – 2022-03 | 40 | 5 | 25 | 0.3 | −0.41 | 0.02 | −17.7 % | −4.01 | 46 |
| 2 | 2022-04 – 2022-06 | 30 | 5 | 25 | 0.7 | −1.35 | 0.00 | +3.9 % | 0.96 | 48 |
| 3 | 2022-07 – 2022-09 | 30 | 10 | 25 | 0.5 | −0.28 | 0.00 | −9.9 % | −1.95 | 57 |
| 4 | 2022-10 – 2022-12 | 50 | 10 | 25 | 0.7 | −0.99 | 0.00 | −27.3 % | −5.39 | 39 |
| 5 | 2023-01 – 2023-03 | 20 | 5 | 20 | 0.3 | −1.64 | 0.00 | −0.9 % | 0.07 | 61 |
| 6 | 2023-04 – 2023-06 | 20 | 5 | 20 | 0.3 | −1.01 | 0.01 | −17.8 % | −2.42 | 58 |
| 7 | 2023-07 – 2023-09 | 50 | 5 | 20 | 0.3 | −0.85 | 0.01 | −16.5 % | −3.36 | 25 |
| 8 | 2023-10 – 2023-12 | 50 | 5 | 20 | 0.3 | −1.28 | 0.00 | −5.1 % | −0.68 | 40 |
| 9 | 2024-01 – 2024-03 | 50 | 5 | 25 | 0.3 | −0.46 | 0.02 | −4.0 % | −0.55 | 42 |
| 10 | 2024-04 – 2024-06 | 50 | 5 | 25 | 0.5 | −0.85 | 0.00 | −16.0 % | −2.82 | 43 |
| 11 | 2024-07 – 2024-09 | 50 | 10 | 25 | 0.5 | −1.32 | 0.00 | +8.6 % | 1.39 | 34 |
| 12 | 2024-10 – 2024-12 | 50 | 10 | 25 | 0.7 | −0.18 | 0.05 | −7.5 % | −1.26 | 38 |
| 13 | 2025-01 – 2025-03 | 50 | 10 | 25 | 0.3 | −0.21 | 0.05 | +3.1 % | 0.69 | 32 |
| 14 | 2025-04 – 2025-06 | 30 | 10 | 25 | 0.3 | 0.12 | 0.13 | +0.5 % | 0.21 | 46 |
| 15 | 2025-07 – 2025-09 | 40 | 5 | 25 | 0.7 | 0.66 | 0.14 | +8.1 % | 1.32 | 46 |
| 16 | 2025-10 (1 month) | 40 | 5 | 25 | 0.5 | 0.97 | 0.28 | −7.6 % | −3.48 | 20 |

Observations:

- **The best in-sample combination had a negative IS Sharpe in 13 of 16 windows.** The optimiser
  selects the least bad combination, and it is still a losing one.
- No IS selection is significant after deflation: the maximum IS DSR is 0.28.
- Parameter stability is low: 13 distinct parameter sets in 16 windows, with a change rate of 87 %.
- **`atr_min_mult` has no effect anywhere in the grid.** Every window has exactly 36 distinct
  outcomes out of 108 combinations, so the filter never binds at 0.3, 0.5 or 0.7. The effective grid
  is 36 combinations, and the reported `atr_min_mult` is an arbitrary tie-break. A nominal
  N = 108 makes the IS DSR more conservative than necessary; with negative Sharpe ratios this does
  not change any conclusion.
- **Anchored:** from window 7 on, no combination met the pre-registered −50 % IS drawdown limit, so
  the anchored variant stayed flat for 2023-07 to 2025-10. With an expanding IS window, a fixed
  drawdown cap becomes ever harder to meet. This is a design flaw of that rule. It was reported, not
  changed after the fact.
- **Forced exits at OOS window ends** (positions the live bot would keep): 2 trades with net −29.79
  USDT (rolling) and 1 trade with −3.76 (anchored). The effect is negligible.

## 6. Results: final hold-out (2025-11-01 to 2026-10-31), evaluated once

- **Fit:** rolling procedure on [2024-11-01, 2025-11-01). It selected `dc_enter 30, dc_exit 5,
  adx_threshold 25, atr_min_mult 0.3`, with an IS Sharpe of 0.66.
- **Run:** 2026-11-24T00:26:58Z, commit `98c2258`, clean tree.
- **Log:** `prior_evaluations` = 0, historical funding.

| | Return | Sharpe | Max DD | Calmar | Vol | Excess return | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strategy | −30.1 % | −1.18 | −37.4 % | −0.81 | 27.3 % | | |
| BTC buy & hold | −29.4 % | −0.59 | −47.6 % | −0.62 | 43.1 % | −0.8 % | −0.13 |
| BTC vol-matched (x0.63) | −17.9 % | −0.59 | −32.5 % | −0.55 | 27.3 % | −12.2 % | −0.43 |
| ETH buy & hold | −41.5 % | −0.62 | −61.1 % | −0.68 | 58.7 % | +11.4 % | 0.07 |

Strategy detail:

- 164 trades, win rate 44.5 %, profit factor 0.79.
- Fees 1,972.91 USDT and funding −35.82 USDT.
- Price PnL including slippage is about −1,003 USDT. This is derived as net PnL (−3,011.76) +
  fees − funding.
- PSR(SR > 0) = 0.139. I computed it from the logged return moments: per-bar SR −0.0126, skew 13.1,
  kurtosis 531, T = 8,760.

The hold-out was a falling market: BTC −29 %, ETH −42 %. The strategy trades both directions and
still lost about as much as BTC buy & hold, and more than vol-matched BTC. Again, it lost money
before fees.

**Unaudited outlier.** The hold-out equity has one bar of +13.2 % (best bar), which drives the
extreme skew and kurtosis. I did not audit this bar. Re-running the hold-out outside the logged
procedure was deliberately not done. Accounting reconciliation (sum of trade PnL = equity change) is
covered by tests and was verified on full, windowed and stitched research runs.

## 7. Conclusion

The hypothesis is rejected on every test:

| Test | Result |
|---|---|
| Fixed parameters, research period | −66 % |
| WFO stitched OOS, rolling | −70 %, PSR 0.01 |
| WFO stitched OOS, anchored | −47 %, PSR 0.02 |
| Final hold-out | −30 %, PSR 0.14 |

The gross signal is close to zero: +5.3 % over almost five years without any costs. The strategy
trades about 165 times per year (795 trades in 4.8 years; 164 in the hold-out), and costs of ~0.16 % per round trip make the result clearly
negative. Passive BTC exposure at the same volatility dominated in all periods tested.

Earlier "validated" results in this repository came from an engine that filled at the signal bar's
close and multiplied PnL by leverage. For example, the comment in `config/pairs.yaml` mentioned an OOS
Sharpe of 1.37. Those results should be considered void.

## 8. Limitations

- **Hold-out contamination is possible.** The strategy design, the grid ranges and the fixed settings
  already existed in the repository on 2026-10-03. That includes the trailing stop at 1.5 %/0.8 %, the
  ATR stop × 2 and the configured parameters. The configuration at the time looked back 1,095 days,
  which overlaps the hold-out. Whether they were tuned on that data cannot be verified. The hold-out
  model was refitted mechanically on pre-hold-out data only. Contamination would bias results
  *upward*, so it cannot explain the negative result.
- **Intrabar order (OHLC).** The trailing stop is ratcheted with the bar's high (long) and checked
  from the next bar on, as in the paper broker. In 517 bars of the research-period backtest, the
  ratcheted stop also lay inside the same bar's range. If the high came first, the live exchange-side
  trailing stop would already have fired. This affects many of the 454 trailing exits, in an unknown
  direction. Lower-timeframe data would resolve it.
- **Proxy data.** Prices and funding are from Binance; the bot trades OKX Europe X-Perps. Fees there
  are not verified.
- **Not modelled:** the daily-loss kill switch and the portfolio risk cap. With one pair and 2 % risk
  per trade, the cap never binds. Exchange minimum order sizes, partial fills and liquidation are
  also not modelled. Stop slippage on gaps is approximated by the bar open.
- **Benchmarks** pay no funding. The vol-matched line uses an ex-post scale and has no costs.
- **DSR trial count.** N counts WFO variants in the local, gitignored research log only. Trials made
  before this study, by hand, in other clones or on other symbols/strategies are not included. The
  stitched-OOS DSR uses the IS trial-Sharpe variance as V, a proxy population.
- **WFO stitching** closes open positions at OOS window ends. The effect was measured as negligible
  (see 5.2).

## 9. Reproducibility

All runs used `config/settings.yaml` and `config/pairs.yaml` at the stated commit, with ETH as the
only active pair. Every run is appended to `logs/research_log.jsonl`, with parameters, period,
metrics, trial counts and git commit.

| What | Command | Commit | Time (UTC) |
|---|---|---|---|
| Fixed-parameter backtest (5.1) | `python main.py --mode backtest --symbol ETH` | `98c2258` | 2026-11-24 00:20 |
| Cost decomposition (5.1) | `python scripts/cost_decomposition.py --symbol ETH` | `aaa6225` (no engine change since `98c2258`) | 2026-11-24 00:28 |
| WFO, rolling + anchored (5.2) | `python main.py --mode optimize --symbol ETH` | `98c2258` | 2026-11-24 00:21–00:24 |
| Final hold-out (6) | `python main.py --mode optimize --symbol ETH --final-holdout` | `98c2258` | 2026-11-24 00:27 |

Notes on the log and repeat runs:

- The same WFO also ran at `0f2edae`, before the review fixes, with identical results. The two
  variants ran three times in total, so N = 2 distinct variants.
- The final-holdout command re-runs the deterministic WFO before the single hold-out evaluation.
- Re-running the final-holdout command now raises a `HoldoutReuseWarning`, and the run is logged as a
  repeated evaluation.

## 10. Next steps (decisions for the owner)

1. **Strategy direction.** Hourly Donchian breakouts with this filter and stop set do not pay for
   their costs. Options include:
   - lower turnover (4h/1d bars, wider stops/trailing),
   - maker execution,
   - a different signal family.

   Any new idea needs a **new** hold-out. The current one is spent.
2. **Trailing-stop modelling.** Use 1m/5m data for exits, or switch to a bar-close trailing stop in
   live, so that backtest and live agree on intrabar order.
3. **Parity items** that need changes in live code:
   - ATR length used for sizing,
   - SL computed from the fill price,
   - paper stop fills on gaps.
4. **Grid design.** Remove the inert `atr_min_mult`. Size the drawdown filter relative to the IS
   length.
