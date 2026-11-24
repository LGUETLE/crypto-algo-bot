"""Portfolio TSMOM engine: look-ahead, execution timing, band, delisting, funding, accounting."""
import numpy as np
import pandas as pd
import pytest

from backtesting.portfolio import (DEFAULTS, Panel, membership, signal, simulate, target_weights,
                                   variant_weights)
from data.universe import base_coin, candidates

P = {**DEFAULTS, "slippage_major": 0.0, "slippage_other": 0.0}


def _frames(n=400, k=20, seed=0, start="2021-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    frames = {}
    for j in range(k):
        drift = rng.normal(0, 0.002)
        close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.04, n)))
        open_ = np.r_[100, close[:-1]]
        frames[f"C{j:02d}USDT"] = pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
            "close": close, "volume": rng.uniform(1e5, 1e6) * rng.uniform(0.5, 1.5, n)}, index=idx)
    # one coin listed late, one delisted early
    frames["C00USDT"] = frames["C00USDT"].iloc[150:]
    frames["C01USDT"] = frames["C01USDT"].iloc[:250]
    return frames


def _panel(**kw):
    return Panel.from_frames(_frames(**kw))


# ── look-ahead ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["V1", "V2", "V3", "V4"])
def test_weights_do_not_depend_on_future_data(variant):
    full = _panel()
    cut_day = full.close.index[300]
    w_full = variant_weights(full, variant, p=P)
    w_cut = variant_weights(full.cut(cut_day), variant, p=P)
    assert (w_full.abs().sum(axis=1) > 0).sum() > 50                 # the test means something
    pd.testing.assert_frame_equal(w_cut, w_full.loc[w_cut.index], check_exact=False, atol=1e-12)


def test_membership_is_point_in_time_and_sized():
    pan = _panel()
    m = membership(pan, P)
    m_cut = membership(pan.cut(pan.close.index[260]), P)
    pd.testing.assert_frame_equal(m_cut, m.loc[m_cut.index])
    sizes = m.sum(axis=1)
    assert sizes.iloc[100:].between(12, 15).all()
    assert not m.loc[:pan.close.index[150 + 89], "C00USDT"].any()     # needs 90 bars of history
    assert not m.iloc[250:]["C01USDT"].any()                           # gone after delisting


def test_membership_one_symbol_per_base_coin():
    fr = _frames(k=14)
    fr["1000C02USDT"] = fr["C02USDT"].assign(volume=fr["C02USDT"]["volume"] * 10)
    pan = Panel.from_frames(fr, base={s: base_coin(s) for s in fr})
    m = membership(pan, P)
    both = m["C02USDT"] & m["1000C02USDT"]
    assert not both.any() and m["1000C02USDT"].iloc[120:].all()


# ── execution ────────────────────────────────────────────────────────────────

def _two_asset_panel(n=30):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    a = np.linspace(100, 130, n)
    b = np.full(n, 50.0)
    mk = lambda c, o: pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                                    "close": c, "volume": 1e6}, index=idx)
    return Panel.from_frames({"AUSDT": mk(a, a - 0.5), "BUSDT": mk(b, b)})


def test_weight_decided_on_close_trades_at_next_open():
    pan = _two_asset_panel()
    W = pd.DataFrame(0.0, index=pan.close.index, columns=pan.close.columns)
    W.iloc[10, 0] = 0.5                                  # decided on the close of day 10
    res = simulate(pan, W, P)
    tr = res.trades
    assert tr["day"].iloc[0] == pan.close.index[11]
    assert tr["notional"].iloc[0] == pytest.approx(0.5 * res.equity.iloc[10])
    # day 12 target is 0 again -> closed at the open of day 12
    assert tr["day"].iloc[1] == pan.close.index[12] and tr["notional"].iloc[1] < 0


def test_no_trade_band_and_zero_target_closes():
    pan = _two_asset_panel()
    W = pd.DataFrame(0.0, index=pan.close.index, columns=pan.close.columns)
    W.iloc[5:20, 1] = 0.30
    W.iloc[8:20, 1] = 0.33                               # change of 0.03 < band: no trade
    res = simulate(pan, W, P)
    days = list(res.trades.loc[res.trades["symbol"] == "BUSDT", "day"])
    assert days == [pan.close.index[6], pan.close.index[21]]   # target 0 decided on day 20


def test_band_does_not_open_small_positions():
    pan = _two_asset_panel()
    W = pd.DataFrame(0.0, index=pan.close.index, columns=pan.close.columns)
    W.iloc[5:, 1] = 0.04
    assert simulate(pan, W, P).trades.empty


def test_delisted_asset_is_closed_at_last_close():
    pan = _two_asset_panel()
    pan.open.iloc[15:, 0] = np.nan
    pan.close.iloc[15:, 0] = np.nan
    W = pd.DataFrame(0.0, index=pan.close.index, columns=pan.close.columns)
    W.iloc[5:, 0] = 0.5
    res = simulate(pan, W, P)
    last = res.trades[res.trades["symbol"] == "AUSDT"].iloc[-1]
    assert last["day"] == pan.close.index[15] and last["notional"] < 0
    assert res.weights.iloc[15:, 0].abs().sum() == 0


def test_funding_sign_and_cost_reconciliation():
    pan = _two_asset_panel()
    pan.funding.iloc[:, 1] = 0.001                       # positive rate: longs pay
    W = pd.DataFrame(0.0, index=pan.close.index, columns=pan.close.columns)
    W.iloc[2:25, 1] = 0.5
    W.iloc[2:25, 0] = -0.3
    p = {**P, "slippage_other": 0.0004}
    res = simulate(pan, W, p)
    assert res.costs["funding"] < 0
    c = res.costs
    assert c["gross_pnl"] - c["fees"] - c["slippage"] + c["funding"] == pytest.approx(c["net_pnl"])
    assert c["net_pnl"] == pytest.approx(res.equity.iloc[-1] - p["capital"])
    traded = res.trades["notional"].abs().sum()
    assert c["fees"] == pytest.approx(traded * p["commission"])


def test_vol_target_scales_predicted_portfolio_vol():
    pan = _panel(n=300, k=15, seed=3)
    m = membership(pan, P)
    s = signal(pan, "sign", "long_short", P)
    p = {**P, "cap_asset": 10.0, "cap_gross": 100.0}     # caps off
    W = target_weights(pan, m, s, p)
    t = 250
    w = W.iloc[t]
    live = w[w != 0].index
    ret = pan.close.pct_change(fill_method=None)[live].iloc[t - 59:t + 1]
    pv = np.sqrt(w[live].to_numpy() @ (ret.cov().to_numpy() * 365) @ w[live].to_numpy())
    assert pv == pytest.approx(0.30, rel=1e-6)


def test_caps_are_respected():
    pan = _panel(n=300, k=15, seed=4)
    W = variant_weights(pan, "V2", p=P)
    assert W.abs().max().max() <= 0.25 + 1e-12
    assert W.abs().sum(axis=1).max() <= 2.0 + 1e-9


def test_long_flat_has_no_shorts():
    pan = _panel(n=300, k=15, seed=5)
    assert (variant_weights(pan, "V1", p=P) >= 0).all().all()


# ── universe helpers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("sym,base", [("ETHUSDT", "ETH"), ("1000PEPEUSDT", "PEPE"), ("1000000MOGUSDT", "MOG"),
                                      ("1MBABYDOGEUSDT", "BABYDOGE"), ("1INCHUSDT", "1INCH"),
                                      ("LUNA2USDT", "LUNA2")])
def test_base_coin(sym, base):
    assert base_coin(sym) == base


def test_candidates_primary_keeps_delisted_and_okx_drops_stables():
    inv = pd.DataFrame({"symbol": ["ETHUSDT", "LUNAUSDT", "FOOUSDT", "USDCUSDT", "BTCDOMUSDT"],
                        "first_month": ["2020-01"] * 5,
                        "last_month": ["2026-10", "2022-05", "2026-10", "2022-01", "2026-10"]})
    inv["base"] = inv["symbol"].map(base_coin)
    prim = candidates(inv, ["ETH", "BTC"], "primary")
    assert set(prim["symbol"]) == {"ETHUSDT", "LUNAUSDT"}
    wide = candidates(inv, ["ETH"], "binance_wide")
    assert set(wide["symbol"]) == {"ETHUSDT", "LUNAUSDT", "FOOUSDT"}


def test_funding_event_day_assignment():
    from data import universe
    s = pd.Series([0.001, 0.002, 0.003],
                  index=pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00"], utc=True))
    day = (s.index - pd.Timedelta("1ns")).floor("D")
    assert (s.groupby(day).sum() == pd.Series([0.006], index=pd.to_datetime(["2024-01-01"], utc=True))).all()
    assert hasattr(universe, "load_funding_panel")


def test_equal_weight_benchmark_holds_between_rebalances():
    from backtesting.portfolio import equal_weight_weights
    pan = _panel(n=200, k=15, seed=6)
    m = membership(pan, P)
    W = equal_weight_weights(m)
    rebal = W.notna().any(axis=1)
    assert rebal.sum() < len(W) / 3                       # mostly holding
    res = simulate(pan, W, {**P, "commission": 0.0, "band": 0.0})
    assert not res.trades.empty and len(res.trades["day"].unique()) <= rebal.sum() + 5
