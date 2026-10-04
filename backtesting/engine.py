"""
Backtest Engine - long and short
================================
Features:
  - ATR-based stop loss (via PositionSizer, identical to live trading)
  - Trailing stop (via risk.stops.update_trailing, identical to live trading)
  - Optional fixed take profit
  - Signal exit and strategy exit channels (long_exit / short_exit)
  - Fees, slippage, simple funding model
  - Walk-forward test

Accounting: `size` is in coins, so PnL = price difference * size.
Leverage only reduces the margin that is locked, it does NOT multiply PnL.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as ta

from backtesting.metrics import calc_all, print_report
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
    pnl_usdt:     Optional[float]          = None
    exit_reason:  str                      = ""


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
        self.funding_8h      = config["backtest"].get("funding_rate_8h", 0.0)
        self.leverage        = risk["leverage"]
        self.trail_act       = risk.get("trailing_activation")
        self.trail_cb        = risk.get("trailing_callback")
        self.sizer           = PositionSizer(config)
        self.initial_capital = 10_000.0

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self, df, strategy, symbol="BTC/USDT:USDT",
            capital=None, verbose=True) -> BacktestResult:

        capital = capital or self.initial_capital
        df = strategy.add_indicators(df.copy())
        df = strategy.generate_signals(df)

        if "atr_14" not in df.columns:
            df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        for col in ("long_exit", "short_exit"):
            df[col] = df[col].fillna(False).astype(bool) if col in df.columns else False

        bar_hours = df.index.to_series().diff().median().total_seconds() / 3600 if len(df) > 1 else 1.0

        cash, position = capital, None
        equity_curve, closed_trades = [], []

        for ts, row in df.iterrows():
            price = row["close"]

            # ── Manage open position ──────────────────────────────────────
            if position is not None:
                is_long = position.side == "long"
                d = 1 if is_long else -1

                # Funding: longs pay, shorts receive (constant assumed rate)
                cash -= d * position.size * price * self.funding_8h * bar_hours / 8

                raw_exit, reason = self._check_exit(position, row)
                if raw_exit is not None:
                    position = self._close(position, ts, raw_exit, reason)
                    cash += self._margin(position) + position.pnl_usdt
                    closed_trades.append(position)
                    position = None
                else:
                    position.sl_price, position.trail_active = update_trailing(
                        position.side, position.entry_price, position.sl_price,
                        position.trail_active, row["high"], row["low"],
                        self.trail_act, self.trail_cb)

            # ── Entry ─────────────────────────────────────────────────────
            if position is None and cash > 100 and row["signal"] in (1, -1):
                side = "long" if row["signal"] == 1 else "short"
                ep   = price * (1 + self.slippage if side == "long" else 1 - self.slippage)
                atr  = row["atr_14"]
                sz   = self.sizer.calculate(cash, ep, side, None if pd.isna(atr) else atr)
                fee  = sz["size"] * ep * self.commission

                if sz["margin"] + fee <= cash * 0.95:
                    cash -= sz["margin"] + fee
                    position = Trade(symbol=symbol, side=side, entry_time=ts,
                                     entry_price=ep, size=sz["size"],
                                     sl_price=sz["sl_price"], tp_price=sz["tp_price"])

            # ── Equity ────────────────────────────────────────────────────
            unreal, locked = 0.0, 0.0
            if position:
                d = 1 if position.side == "long" else -1
                unreal = d * (price - position.entry_price) * position.size
                locked = self._margin(position)
            equity_curve.append((ts, cash + locked + unreal))

        # Close the last open position at the final close
        if position:
            position = self._close(position, df.index[-1], df["close"].iloc[-1], "End")
            closed_trades.append(position)

        equity  = pd.Series([e for _, e in equity_curve],
                            index=[t for t, _ in equity_curve], name="equity")
        trades  = self._to_df(closed_trades)
        metrics = calc_all(equity, trades, capital)

        if verbose:
            if not trades.empty:
                for side in ("long", "short"):
                    t = trades[trades["side"] == side]
                    print(f"\n  {side.capitalize():<5} trades: {len(t):>3} | PnL: ${t['pnl_usdt'].sum():>8.0f}")
                print(f"  Exits: {dict(trades['exit_reason'].value_counts())}")
            print_report(metrics)

        return BacktestResult(equity=equity, trades=trades, metrics=metrics, config=self.config)

    # ── Exit logic ────────────────────────────────────────────────────────

    @staticmethod
    def _check_exit(pos: Trade, row):
        """Returns (raw_exit_price, reason) or (None, ""). Stop is checked first (conservative)."""
        is_long = pos.side == "long"

        if (row["low"] <= pos.sl_price) if is_long else (row["high"] >= pos.sl_price):
            # On a gap through the stop we are filled at the open, not at the stop
            raw = min(row["open"], pos.sl_price) if is_long else max(row["open"], pos.sl_price)
            return raw, ("TrailingSL" if pos.trail_active else "SL")

        if pos.tp_price is not None and \
           ((row["high"] >= pos.tp_price) if is_long else (row["low"] <= pos.tp_price)):
            return pos.tp_price, "TP"

        opposite = row["signal"] == (-1 if is_long else 1)
        if opposite or row["long_exit" if is_long else "short_exit"]:
            return row["close"], "Signal"

        return None, ""

    def _close(self, pos: Trade, ts, raw_price, reason) -> Trade:
        is_long = pos.side == "long"
        exit_px = raw_price * (1 - self.slippage if is_long else 1 + self.slippage)
        d       = 1 if is_long else -1
        pnl     = d * (exit_px - pos.entry_price) * pos.size - exit_px * pos.size * self.commission
        pos.exit_time, pos.exit_price = ts, exit_px
        pos.pnl_usdt, pos.exit_reason = round(pnl, 4), reason
        return pos

    def _margin(self, pos: Optional[Trade]) -> float:
        return pos.size * pos.entry_price / self.leverage if pos else 0.0

    @staticmethod
    def _to_df(trades) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "symbol": t.symbol, "side": t.side,
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "size": t.size, "pnl_usdt": t.pnl_usdt, "exit_reason": t.exit_reason,
        } for t in trades])

    # ── Walk-forward ──────────────────────────────────────────────────────

    def walk_forward(self, df, strategy, symbol="BTC/USDT:USDT", n_splits=5) -> dict:
        split_size = len(df) // n_splits
        is_pct     = 1 - self.config["backtest"]["walk_forward_pct"]
        results    = []
        logger.info(f"Walk-forward: {n_splits} splits, {is_pct:.0%} in-sample")

        for i in range(n_splits):
            part = df.iloc[i * split_size:(i + 1) * split_size]
            sp   = int(len(part) * is_pct)
            in_s, out_s = part.iloc[:sp], part.iloc[sp:]
            if len(in_s) < 100 or len(out_s) < 30:
                continue
            ir  = self.run(in_s,  strategy, symbol, verbose=False)
            oor = self.run(out_s, strategy, symbol, verbose=False)
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
        self._print_wf(df_res)
        return {
            "results": df_res,
            "avg_oos_sharpe": df_res["oos_sharpe"].mean() if not df_res.empty else 0,
            "avg_oos_return": df_res["oos_return"].mean() if not df_res.empty else 0,
            "consistency":    (df_res["oos_return"] > 0).mean() if not df_res.empty else 0,
        }

    @staticmethod
    def _print_wf(df):
        print("\n" + "=" * 65)
        print("  WALK-FORWARD RESULTS")
        print("=" * 65)
        print(f"  {'Split':>5} | {'IS Sharpe':>10} | {'OOS Sharpe':>10} | {'IS Return':>10} | {'OOS Return':>10}")
        print(f"  {'-' * 60}")
        for _, r in df.iterrows():
            ok = "OK" if r["oos_return"] > 0 else "--"
            print(f"  {int(r['split']):>5} | {r['is_sharpe']:>10.2f} | {r['oos_sharpe']:>10.2f} | "
                  f"{r['is_return']:>9.1%} | {r['oos_return']:>9.1%} {ok}")
        c = (df["oos_return"] > 0).mean() if not df.empty else 0
        print("=" * 65)
        print(f"\n  OOS consistency: {c:.0%}")
        print("  Robust - paper trading recommended" if c >= 0.6 else
              "  Moderate - keep optimizing" if c >= 0.4 else
              "  Not consistent enough")
        print()
