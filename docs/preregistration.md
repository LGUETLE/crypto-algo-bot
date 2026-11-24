# Preregistration: cross-asset time-series momentum (TSMOM) on crypto perpetuals

- Status: **frozen**. Written and committed before any real-data run of this strategy.
- Amendments are append-only (section 12), each with a date and a reason. The sections above them
  are never edited after the first real-data run.
- Author: research pipeline of this repository, on behalf of the owner.
- Date: 2026-11-24.

## 1. Hypothesis

A diversified, volatility-targeted portfolio of liquid crypto perpetuals, positioned by
time-series momentum (the sign or strength of each asset's own recent return), earns a positive
risk-adjusted return **after** taker fees, slippage and historical funding. It adds value over
passive BTC exposure at the same volatility.

Null hypothesis: the strategy's true Sharpe ratio after costs is <= 0.

## 2. Data

| Item | Definition |
|---|---|
| Prices | Binance USD-M USDT perpetuals, **daily bars** (00:00 UTC open), from `data.binance.vision`, SHA-256 verified. Binance REST is blocked (HTTP 451) on the research machine |
| Funding | Binance realised funding rates (8-hourly) per asset, as a proxy. OKX Europe X-Perps settle funding **hourly**; this difference is a known limitation |
| Window | 2021-01-01 to 2026-11-01 (end exclusive) |
| Research period | 2021-01-01 to 2025-11-01 |
| Final hold-out | **2025-11-01 to 2026-10-31**, evaluated **once** at the very end |
| Liquidity measure | daily quote volume, approximated as `volume x close` |

Symbol mapping:

- A Binance symbol maps to its base coin by stripping `USDT` and a leading `1000000`, `1000` or `1M`
  (for example `1000PEPEUSDT` -> PEPE).
- Contract multipliers do not affect returns.

## 3. Universe (point in time, no survivorship bias)

**Candidates:** every Binance USD-M USDT perpetual in the archive that meets one of two conditions.

- (a) Its base coin is a **crypto X-Perp currently listed on OKX Europe**: `ruleType = xperp`,
  `instCategory = 1` on `eea.okx.com /api/v5/public/instruments?instType=FUTURES`, list retrieved
  on the day of the first run.
- (b) It **has been delisted from Binance** before 2026-10. A delisted symbol is one whose last
  monthly archive file predates 2026-10.
  - Rationale: all OKX Europe X-Perps were listed in 2026, so OKX listing cannot be checked
    point in time for 2021-2025.
  - Dropping coins that died (LUNA, FTT, ...) would be exactly the survivorship bias. They stay
    in and are treated as tradable while they were liquid.
- Excluded in all cases: stablecoins and index products (USDC, BUSD, TUSD, FDUSD, USDP, DAI, BTCDOM,
  DEFI, FOOTBALL, BLUEBIRD and any other non-coin index).

**Eligibility on day t** (all information up to the close of t):

- at least 90 daily bars of history,
- a bar on day t,
- a trailing 30-day average quote volume.

**Membership:**

- Rank the eligible candidates by trailing 30-day average quote volume.
- Members stay while their rank is <= 15 (hysteresis).
- Free places are filled with the best-ranked non-members until there are 12 members.
- The universe therefore holds 12-15 coins.

**Leaving the universe:**

- An asset that leaves (illiquid or rank > 15) gets target weight 0 at the next rebalance.
- An asset whose data ends (delisted) is closed at its last available close, with normal costs.
  This is an optimistic assumption, listed under limitations.

## 4. Signal (decided on the close of day t, traded at the open of day t+1)

Lookbacks: L in {7, 14, 28, 56} days. Daily log returns r. Volatility sigma_d = standard deviation
of r over the last 60 days.

- **sign** variant: `s = mean_L sign(ln(C_t / C_{t-L}))`, in [-1, 1].
- **volscaled** variant: `z_L = ln(C_t / C_{t-L}) / (sigma_d * sqrt(L))`,
  `s = mean_L clip(z_L, -2, 2) / 2`, in [-1, 1].
- **long_flat** direction: `s+ = max(s, 0)`.
- **long_short** direction: s as is.

## 5. Portfolio construction

| Parameter | Value (fixed, not optimised) |
|---|---|
| Asset volatility | sigma_i = std of daily simple returns over 60 days x sqrt(365) |
| Raw weight | w_i = s_i / sigma_i (inverse volatility) |
| Portfolio volatility target | **30 % p.a.** Scale k = 0.30 / sqrt(w' Sigma w), where Sigma = 60-day sample covariance of the members' daily returns, annualised x 365 (pairwise complete) |
| Cap per asset | abs(w_i) <= 0.25 (notional / equity) |
| Gross cap | sum abs(w_i) <= 2.0; all weights are scaled down proportionally if it is exceeded |
| Order of operations | raw weights -> vol-target scaling -> per-asset cap -> gross cap (realised vol may end up below target) |
| No-trade band | trade asset i only if abs(w_target - w_current) > **0.05**. A target of exactly 0 (signal off or left the universe) always closes the position |
| Rebalance | daily, at the open of day t+1, after the signal on the close of day t |
| Capital | 10,000 USDT, PnL compounding; positions are in coins (units), w_current drifts with prices |

## 6. Costs

| Cost | Value |
|---|---|
| Commission | 0.05 % per side of traded notional (OKX Europe X-Perps taker, standard tier) |
| Slippage | 0.02 % per side for BTC and ETH, 0.05 % per side for all other coins |
| Funding | historical Binance rate for each event in (open_t, open_{t+1}] on the position held during day t; payment = -units x price x rate, with price = close of day t (approximation) |

Reported for each run: turnover (traded notional / average equity per year), gross vs net PnL,
fees, slippage and funding.

## 7. Variants and trial count

The core variants, each a fixed configuration on the research period:

| # | Signal | Direction |
|---:|---|---|
| V1 | sign | long_flat |
| V2 | sign | long_short |
| V3 | volscaled | long_flat |
| V4 | volscaled | long_short |

Selection procedures (walk-forward, choosing among V1-V4):

| # | Procedure |
|---:|---|
| W1 | rolling WFO: IS 12 months, OOS 3 months, step 3 months; per window pick the variant with the highest IS Sharpe; OOS equity compounded, positions carried across windows |
| W2 | anchored WFO, same rules with an expanding IS |

Robustness (universe):

| # | Test |
|---:|---|
| R1 | W1 on the Binance-wide point-in-time universe: all USDT perps, with no OKX filter |

**Trial count N = 7** (V1-V4, W1, W2, R1). The Deflated Sharpe Ratio uses N = 7 and V = variance of
the per-day Sharpe ratios of these 7 configurations on the research period.

Nothing else is varied: lookbacks, volatility windows, target, caps, band, universe size and costs
are fixed above. Any additional configuration is an amendment and increases N.

## 8. Validation

- WFO as in Part 1 of RESEARCH.md: windows on calendar months, data at or after the IS end never
  used for selection. Truncation tests prove that the weights on day t depend only on data up to
  the close of day t.
- Statistics: PSR(SR > 0) and DSR (Bailey & Lopez de Prado 2014) for the stitched OOS curves and
  the fixed variants.

## 9. Final hold-out (once)

- **Model:** procedure W1. Select the variant with the highest Sharpe on the last 12 months before
  2025-11-01, freeze it, and trade 2025-11-01 to 2026-10-31 on the primary universe.
- **Logging:** the evaluation is logged; a repeat triggers a warning.
- **No changes afterwards:** no parameter or code change after the hold-out evaluation affects
  the reported result.

## 10. Benchmarks (identical days)

- BTC buy & hold.
- Equal-weight buy & hold of the same point-in-time universe: equal weights across members,
  rebalanced on the first day of each month and when membership changes, without costs or funding.
- BTC scaled to the strategy's volatility (ex-post scale, no costs).

Reported: return, CAGR, Sharpe, max drawdown, Calmar, excess return, information ratio.

## 11. Decision rule

The strategy is considered to have an edge only if both of these hold:

- the stitched OOS of W1 has PSR(SR > 0) >= 0.95 and DSR >= 0.95, and
- the hold-out Sharpe is > 0.

Anything else is reported as no demonstrated edge.

## 12. Amendments (append-only)

### Amendment 1 (2026-11-24): funding settlement interval

- **Reason:** section 2 states that OKX Europe X-Perps settle funding hourly. Measured from the
  public funding-history endpoint (`scripts/compare_funding.py`, RESEARCH.md Appendix A), they
  settle **every 8 hours** (00/08/16 UTC).
- **Impact:** none on any reported number, because the study uses Binance funding.

### Amendment 2 (2026-11-24): forward paper trading of `tsmom-v1`

Written after the study and its single hold-out evaluation. It adds an independent forward test;
nothing above is changed.

**Frozen configuration `tsmom-v1`:**

| Item | Value |
|---|---|
| Variant | **V1: sign signal, long_flat.** This is the variant the preregistered W1 procedure selected on [2024-11-01, 2025-11-01), the last IS window before the hold-out. Static: no re-selection |
| Parameters | `config/tsmom.yaml` unchanged: lookbacks 7/14/28/56, vol and covariance windows 60, vol target 30 %, cap 0.25, gross cap 2.0, band 0.05, 12 members (stay <= 15), 30-day volume ranking, >= 90 days of history |
| Universe | candidates = the 168 crypto X-Perps in `config/okx_xperps_snapshot.json` (OKX Europe, retrieved 2026-11-24). Coins listed later are not added. A coin that stops trading is closed at its last close |
| Data | OKX Europe X-Perps public endpoints only: daily candles (`bar=1Dutc`), volume = coins x close, realised 8-hourly funding. This differs from the study's Binance proxy; the rules are identical |
| Costs | commission 0.05 % per side; slippage 0.02 % (BTC, ETH) / 0.05 % (others) per side; funding as charged by the X-Perp |
| Capital | 10,000 USDT (paper) |
| Start | weights decided on the close of 2026-11-24 are traded at the **open of 2026-11-25** (00:00 UTC). This is the first day of the track record |
| Execution | GitHub Actions, daily after 00:00 UTC (`.github/workflows/paper-tsmom.yml`). Data, state and the daily log are stored on branch `paper-state`. Each run replays the frozen engine from the start date over the stored data |

**Evaluation rule:**

- **Quarterly** (after 2027-02-25, 2027-05-25, 2027-08-25, 2027-11-25), report:
  - return,
  - annualised Sharpe with its standard error SE = sqrt((1 + SR_d² / 2) / T) x sqrt(365) and the
    95 % interval,
  - maximum drawdown,
  - turnover, fees, slippage and funding,
  - benchmarks over identical days: BTC buy & hold (OKX BTC X-Perp), equal-weight universe and
    vol-matched BTC, each with excess return and information ratio.
- **Formal verdict after 12 months** (on the data up to the close of 2027-11-24):
  - **"not supported":** the upper bound of the 95 % Sharpe interval is below 0.66, the
    less-biased Binance-wide estimate from the study.
  - **"edge confirmed":** Sharpe / SE >= 1.645. With one year of data this requires a Sharpe of
    about 1.64 or more; RESEARCH.md P2.5 explains why a moderate edge cannot be confirmed in 12 months.
  - **"inconclusive, consistent with the study":** every other outcome.
  - The drawdown-overlay claim is assessed separately: strategy max drawdown vs BTC buy & hold max
    drawdown over the same days.
- **No changes:** no parameter, universe, data or cost change during the 12 months. Any change
  creates `tsmom-v2` with a new start date and its own 12-month clock.
