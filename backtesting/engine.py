"""
Backtest Engine - long and short
================================
Timing: a signal on the close of bar t is filled at the OPEN of bar t+1 (entries, signal
exits and reversals). Stops/take profit are checked intrabar from the entry bar on.

Features:
  - ATR-based stop loss (via PositionSizer, identical to live trading)
  - Trailing stop (via risk.stops.update_trailing, identical to live trading)
  - Optional fixed take profit
  - Signal exit and strategy exit channels (long_exit / short_exit)
  - Fees, slippage, simple funding model
  - Split consistency check (fixed parameters; walk-forward optimisation lives in optimizer.py)

Accounting: `size` is in coins, so PnL = price difference * size.
Leverage only reduces the margin that is locked, it does NOT multiply PnL.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from backtesting.metrics import calc_all, infer_periods_per_year, periods_per_year, print_report
from features.timeframes import to_timedelta
from risk.position_sizer import PositionSizer
from risk.stops import update_trailing

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol:       str
    side:         str                      # "long" | "short"
    entry_time:   pd.Timestamp
    entry_price:  float
    size:         float
    sl_price:     float
    tp_price:     Optional[float]
    trail_active: bool                     = False
    exit_time:    Optional[pd.Timestamp]   = None
    exit_price:   Optional[float]          = None
    pnl_usdt:     Optional[float]          = None   # NET: price PnL - both fees + funding
    exit_reason:  str                      = ""
    entry_fee:    float                    = 0.0
    exit_fee:     float                    = 0.0
    funding:      float                    = 0.0    # received (+) / paid (-) while open


@dataclass
class BacktestResult:
    equity:  pd.Series
    trades:  pd.DataFrame
    metrics: dict
    config:  dict


class BacktestEngine:
    def __init__(self, config: dict):
        risk = config["risk"]
        self.config          = config
        self.commission      = config["backtest"]["commission"]
        self.slippage        = config["backtest"]["slippage"]
        self.funding_8h      = config["backtest"].get("funding_rate_8h", 0.0)   # fallback only
        self._warned_const   = False
        self.leverage        = risk["leverage"]
        self.trail_act       = risk.get("trailing_activation")
        self.trail_cb        = risk.get("trailing_callback")
        self.sizer           = PositionSizer(config)
        self.initial_capital = 10_000.0
        self.timeframe       = config.get("timeframe")
        self.risk_free_rate  = config["backtest"].get("risk_free_rate", 0.0)

    def periods_per_year(self, index=None) -> float:
        """Annualisation factor: from the configured timeframe, else from the bar spacing."""
        if self.timeframe:
            return periods_per_year(self.timeframe)
        return infer_periods_per_year(index)

    # ── Main loop ─────────────────────────────────────────────────────────

    @staticmethod
    def prepare(df: pd.DataFrame, strategy) -> pd.DataFrame:
        """Indicators + signals on the whole frame (all features are causal, see test_lookahead)."""
        df = strategy.add_indicators(df.copy())
        df = strategy.generate_signals(df)
        if "atr_14" not in df.columns:
            df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        for col in ("long_exit", "short_exit"):
            df[col] = df[col].fillna(False).astype(bool) if col in df.columns else False
        df["signal"] = df["signal"].fillna(0).astype(int)
        return df

    def run(self, df, strategy, symbol="BTC/USDT:USDT", capital=None, verbose=True,
            start=None, end=None, funding: Optional[pd.Series] = None) -> BacktestResult:
        """
        Event order per bar t (timestamps are candle OPEN times):
          1. orders decided on the close of bar t-1 are filled at the OPEN of bar t:
             either a signal exit (exit flag or opposite signal) OR, when flat, an entry.
             Like the live bot (main.run_cycle), an opposite signal only closes the position;
             a new entry needs a signal on a later bar.
          2. stop loss / take profit are checked against bar t's range (incl. the entry bar)
          3. the trailing stop is ratcheted with bar t's high/low (risk.stops, same as live)
          4. equity is marked at the close of bar t
          5. the signal on the close of bar t becomes the pending order for bar t+1
        A signal on the last bar is never filled (there is no next open).

        Funding: `funding` = realised rates indexed by funding time (see DataFetcher.funding_history).
        An event at time T is settled at the start of the first bar with open >= T, BEFORE that
        bar's fills, against the position held at T, at the close of the bar ending at T:
        payment = -side * size * price * rate (positive rate: longs pay). Without a series, a
        constant backtest.funding_rate_8h at 00/08/16 UTC is used and a warning is logged.
        Approximation: with bars longer than the funding interval, a position stopped out
        intrabar is not charged for events inside its exit bar.

        start / end (optional, end exclusive) restrict TRADING to a window. Indicators are still
        computed on all rows of `df`, so history before `start` serves as warm-up.
        """
        capital = capital or self.initial_capital
        full = self.prepare(df, strategy)

        mask = pd.Series(True, index=full.index)
        if start is not None:
            mask &= full.index >= pd.Timestamp(start)
        if end is not None:
            mask &= full.index < pd.Timestamp(end)
        pos_idx = np.flatnonzero(mask.to_numpy())
        if len(pos_idx) == 0:
            empty = pd.Series(dtype=float, name="equity")
            return BacktestResult(equity=empty, trades=pd.DataFrame(), metrics={}, config=self.config)
        first, last = pos_idx[0], pos_idx[-1]
        if last - first + 1 != len(pos_idx):
            raise ValueError("start/end must select a contiguous block of bars")

        o, h, l, c = (full[k].to_numpy(dtype=float).tolist() for k in ("open", "high", "low", "close"))
        sig = full["signal"].to_numpy().tolist()
        lx  = full["long_exit"].to_numpy().tolist()
        sx  = full["short_exit"].to_numpy().tolist()
        atr = full["atr_14"].to_numpy(dtype=float).tolist()
        idx = full.index

        fund = self._funding_per_bar(idx, first, last, funding)

        cash, position = float(capital), None
        equity_vals, closed_trades = [], []
        ambiguous = 0          # bars where the ratcheted trailing stop also lies inside the bar range

        for i in range(first, last + 1):
            ts = idx[i]
            # Pending order from the previous bar's close (also when it lies just before `start`)
            p_sig, p_lx, p_sx, p_atr = (sig[i - 1], lx[i - 1], sx[i - 1], atr[i - 1]) if i > 0 \
                else (0, False, False, float("nan"))

            # ── 0. Funding events since the previous bar's open ───────────
            if position is not None and fund[i] != 0.0:
                d = 1 if position.side == "long" else -1
                pay = -d * position.size * c[i - 1] * fund[i]
                cash += pay
                position.funding += pay

            # ── 1. Fills at the open ──────────────────────────────────────
            exited = False
            if position is not None:
                is_long  = position.side == "long"
                opposite = p_sig == (-1 if is_long else 1)
                if opposite or (p_lx if is_long else p_sx):
                    position = self._close(position, ts, o[i], "Signal")
                    cash += self._released(position)
                    closed_trades.append(position)
                    position, exited = None, True

            if position is None and not exited and p_sig in (1, -1) and cash > 100:
                side = "long" if p_sig == 1 else "short"
                ep   = o[i] * (1 + self.slippage if side == "long" else 1 - self.slippage)
                sz   = self.sizer.calculate(cash, ep, side, None if np.isnan(p_atr) else p_atr)
                fee  = sz["size"] * ep * self.commission
                # Margin booked exactly as it is released in _margin() (the sizer's value is rounded)
                margin = sz["size"] * ep / self.leverage
                if margin + fee <= cash * 0.95:
                    cash -= margin + fee
                    position = Trade(symbol=symbol, side=side, entry_time=ts,
                                     entry_price=ep, size=sz["size"],
                                     sl_price=sz["sl_price"], tp_price=sz["tp_price"],
                                     entry_fee=fee)

            # ── 2./3. Intrabar stop / take profit, then trailing ──────────
            if position is not None:
                raw_exit, reason = self._check_exit(position, o[i], h[i], l[i])
                if raw_exit is not None:
                    position = self._close(position, ts, raw_exit, reason)
                    cash += self._released(position)
                    closed_trades.append(position)
                    position = None
                else:
                    position.sl_price, position.trail_active = update_trailing(
                        position.side, position.entry_price, position.sl_price,
                        position.trail_active, h[i], l[i], self.trail_act, self.trail_cb)
                    # OHLC ambiguity: if the high came before the low, an exchange-side trailing
                    # stop would already have fired inside this bar. Like the paper broker, the
                    # engine only checks the new level from the next bar on; count how often
                    # this matters (reported as metrics["ambiguous_trailing_bars"]).
                    if position.trail_active and ((l[i] <= position.sl_price) if position.side == "long"
                                                  else (h[i] >= position.sl_price)):
                        ambiguous += 1

            # ── 4. Equity at the close ────────────────────────────────────
            unreal, locked = 0.0, 0.0
            if position is not None:
                d = 1 if position.side == "long" else -1
                unreal = d * (c[i] - position.entry_price) * position.size
                locked = self._margin(position)
            equity_vals.append(cash + locked + unreal)

        # Close a position still open at the end of the window at the final close (with costs)
        if position is not None:
            if fund[last + 1] != 0.0:              # events inside the final bar
                d = 1 if position.side == "long" else -1
                pay = -d * position.size * c[last] * fund[last + 1]
                cash += pay
                position.funding += pay
            position = self._close(position, idx[last], c[last], "End")
            cash += self._released(position)
            closed_trades.append(position)
            equity_vals[-1] = cash

        equity  = pd.Series(equity_vals, index=idx[first:last + 1], name="equity")
        trades  = self._to_df(closed_trades)
        metrics = calc_all(equity, trades, capital, periods_per_year=self.periods_per_year(idx),
                           risk_free_rate=self.risk_free_rate)
        metrics["ambiguous_trailing_bars"] = ambiguous

        if verbose:
            if not trades.empty:
                for side in ("long", "short"):
                    t = trades[trades["side"] == side]
                    print(f"\n  {side.capitalize():<5} trades: {len(t):>3} | PnL: ${t['pnl_usdt'].sum():>8.0f}")
                print(f"  Exits: { {k: int(v) for k, v in trades['exit_reason'].value_counts().items()} }")
            print_report(metrics)

        return BacktestResult(equity=equity, trades=trades, metrics=metrics, config=self.config)

    # ── Exit logic ────────────────────────────────────────────────────────

    @staticmethod
    def _check_exit(pos: Trade, open_, high, low):
        """Intrabar exits. Returns (raw_exit_price, reason) or (None, ""). Stop first (conservative)."""
        is_long = pos.side == "long"

        if (low <= pos.sl_price) if is_long else (high >= pos.sl_price):
            # On a gap through the stop we are filled at the open, not at the stop
            raw = min(open_, pos.sl_price) if is_long else max(open_, pos.sl_price)
            return raw, ("TrailingSL" if pos.trail_active else "SL")

        if pos.tp_price is not None and \
           ((high >= pos.tp_price) if is_long else (low <= pos.tp_price)):
            return pos.tp_price, "TP"

        return None, ""

    def _close(self, pos: Trade, ts, raw_price, reason) -> Trade:
        """Sets exit price/fee and the NET trade PnL (price PnL - entry fee - exit fee + funding)."""
        is_long = pos.side == "long"
        exit_px = raw_price * (1 - self.slippage if is_long else 1 + self.slippage)
        d       = 1 if is_long else -1
        pos.exit_fee  = exit_px * pos.size * self.commission
        pos.exit_time, pos.exit_price, pos.exit_reason = ts, exit_px, reason
        pos.pnl_usdt  = self._gross(pos) - pos.entry_fee - pos.exit_fee + pos.funding
        return pos

    @staticmethod
    def _gross(pos: Trade) -> float:
        d = 1 if pos.side == "long" else -1
        return d * (pos.exit_price - pos.entry_price) * pos.size

    def _released(self, pos: Trade) -> float:
        """Cash returned on close. Entry fee and funding were already booked when they occurred."""
        return self._margin(pos) + self._gross(pos) - pos.exit_fee

    def _margin(self, pos: Optional[Trade]) -> float:
        return pos.size * pos.entry_price / self.leverage if pos else 0.0

    def _funding_per_bar(self, idx: pd.DatetimeIndex, first: int, last: int,
                         funding: Optional[pd.Series]) -> np.ndarray:
        """
        Sum of funding rates to settle at the start of each bar (array of len(idx) + 1; the last
        slot collects events inside the final bar). Event at T -> first bar with open >= T.
        """
        out = np.zeros(len(idx) + 1)
        if len(idx) == 0:
            return out
        bar = to_timedelta(self.timeframe) if self.timeframe else (idx[1] - idx[0] if len(idx) > 1
                                                                    else pd.Timedelta("1h"))
        lo, hi = idx[max(first - 1, 0)], idx[last] + bar
        if funding is None:
            if not self.funding_8h:
                return out
            if not self._warned_const:
                logger.warning(f"No historical funding series given: using a CONSTANT funding rate of "
                               f"{self.funding_8h:+.4%} per 8h (00/08/16 UTC). Results are approximate.")
                self._warned_const = True
            times = pd.date_range(lo.floor("8h"), hi, freq="8h")
            funding = pd.Series(self.funding_8h, index=times)
        f = funding[(funding.index > lo) & (funding.index <= hi)]
        if f.empty:
            return out
        pos = np.searchsorted(idx.asi8, f.index.as_unit(idx.unit).asi8, side="left")
        np.add.at(out, pos, f.to_numpy(dtype=float))
        out[: first] = 0.0          # never settle before trading starts
        return out

    @staticmethod
    def _to_df(trades) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "symbol": t.symbol, "side": t.side,
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "size": t.size, "pnl_usdt": t.pnl_usdt, "exit_reason": t.exit_reason,
            "fees": t.entry_fee + t.exit_fee, "funding": t.funding,
        } for t in trades])

    # ── Split consistency check (fixed parameters, NO re-fit) ─────────────

    def split_consistency_check(self, df, strategy, symbol="BTC/USDT:USDT", n_splits=5,
                                funding: Optional[pd.Series] = None) -> dict:
        """
        Cuts the data into n_splits consecutive blocks and runs the SAME (fixed) parameters on the
        first part (labelled IS) and the last walk_forward_pct of each block (labelled OOS).
        Nothing is fitted, so this only checks whether a fixed parameter set behaves consistently
        across time. It is NOT a walk-forward optimisation: see
        ParameterOptimizer.walk_forward_optimize() for that.
        """
        split_size = len(df) // n_splits
        is_pct     = 1 - self.config["backtest"]["walk_forward_pct"]
        results    = []
        logger.info(f"Split consistency check: {n_splits} splits, fixed parameters")

        for i in range(n_splits):
            part = df.iloc[i * split_size:(i + 1) * split_size]
            sp   = int(len(part) * is_pct)
            if sp < 100 or len(part) - sp < 30:
                continue
            upto  = df[df.index <= part.index[-1]]          # earlier rows serve as indicator warm-up
            ir  = self.run(upto, strategy, symbol, verbose=False, start=part.index[0],
                           end=part.index[sp], funding=funding)
            oor = self.run(upto, strategy, symbol, verbose=False, start=part.index[sp], funding=funding)
            results.append({
                "split": i + 1,
                "is_sharpe":  ir.metrics.get("sharpe", 0),
                "oos_sharpe": oor.metrics.get("sharpe", 0),
                "is_return":  ir.metrics.get("total_return", 0),
                "oos_return": oor.metrics.get("total_return", 0),
                "is_trades":  ir.metrics.get("total_trades", 0),
                "oos_trades": oor.metrics.get("total_trades", 0),
            })

        df_res = pd.DataFrame(results)
        self._print_split_check(df_res)
        return {
            "results": df_res,
            "avg_oos_sharpe": df_res["oos_sharpe"].mean() if not df_res.empty else 0,
            "avg_oos_return": df_res["oos_return"].mean() if not df_res.empty else 0,
            "consistency":    (df_res["oos_return"] > 0).mean() if not df_res.empty else 0,
        }

    @staticmethod
    def _print_split_check(df):
        print("\n" + "=" * 65)
        print("  SPLIT CONSISTENCY CHECK (fixed parameters, no re-fit)")
        print("=" * 65)
        print(f"  {'Split':>5} | {'IS Sharpe':>10} | {'OOS Sharpe':>10} | {'IS Return':>10} | {'OOS Return':>10}")
        print(f"  {'-' * 60}")
        for _, r in df.iterrows():
            ok = "OK" if r["oos_return"] > 0 else "--"
            print(f"  {int(r['split']):>5} | {r['is_sharpe']:>10.2f} | {r['oos_sharpe']:>10.2f} | "
                  f"{r['is_return']:>9.1%} | {r['oos_return']:>9.1%} {ok}")
        c = (df["oos_return"] > 0).mean() if not df.empty else 0
        print("=" * 65)
        print(f"\n  Share of positive 'OOS' blocks: {c:.0%} (same parameters everywhere - not an OOS test)\n")
