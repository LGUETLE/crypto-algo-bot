"""Cost report: gross vs net decomposition, turnover, break-even cost per round trip."""
import pandas as pd
import pytest

from backtesting.costs import cost_breakdown
from backtesting.engine import BacktestEngine
from tests.test_lookahead import CONFIG, ScriptedStrategy, ohlcv

CFG = {**CONFIG, "backtest": {"commission": 0.0005, "slippage": 0.0003, "funding_rate_8h": 0.0}}


def _res():
    df = ohlcv(400)
    strat = ScriptedStrategy(CFG, entries={5: 1, 60: -1, 61: -1, 150: 1, 151: 1}, exits={300})
    return BacktestEngine(CFG).run(df, strat, verbose=False)


def test_trades_record_slippage_cost():
    t = _res().trades
    row = t.iloc[0]
    raw_entry = row["entry_price"] / (1 + 0.0003)
    raw_exit  = row["exit_price"] / (1 - 0.0003)
    assert row["slippage"] == pytest.approx(row["size"] * ((row["entry_price"] - raw_entry) + (raw_exit - row["exit_price"])))


def test_decomposition_reconciles():
    r = _res()
    c = cost_breakdown(r.trades, r.equity, 10_000, CFG)
    assert c["gross_pnl"] - c["fees"] - c["slippage"] + c["funding"] == pytest.approx(c["net_pnl"])
    assert c["net_pnl"] == pytest.approx(r.equity.iloc[-1] - 10_000)
    assert c["round_trips"] == len(r.trades)


def test_break_even_cost_and_turnover():
    r = _res()
    c = cost_breakdown(r.trades, r.equity, 10_000, CFG)
    notional = (r.trades["size"] * r.trades["entry_price"]).sum()
    assert c["break_even_cost_rt"] == pytest.approx((c["gross_pnl"] + c["funding"]) / notional)
    assert c["assumed_cost_rt"] == pytest.approx(2 * (0.0005 + 0.0003))
    years = (r.equity.index[-1] - r.equity.index[0] + pd.Timedelta("1h")) / pd.Timedelta(days=365)
    traded = (r.trades["size"] * (r.trades["entry_price"] + r.trades["exit_price"])).sum()
    assert c["turnover_per_year"] == pytest.approx(traded / r.equity.mean() / years)
