"""OKX Europe X-Perps: market loading and request building from recorded PUBLIC responses.
No network, no orders: ccxt's HTTP methods are replaced by fixtures / capture functions."""
import json
import os

import ccxt
import pandas as pd
import pytest

import execution.order_manager as om_module
from data import okx
from execution.broker import Broker, is_xperp, xperp_symbol

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    with open(os.path.join(FIX, name)) as f:
        return json.load(f)


def _exchange_with_fixture(**kw):
    ex = ccxt.myokx({"options": {"fetchMarkets": {"types": ["future"]}}, **kw})
    inst = _fixture("okx_eea_futures_instruments.json")
    ex.fetch = lambda url, method="GET", headers=None, body=None: inst       # no network
    ex.load_markets()
    return ex


@pytest.fixture
def broker(tmp_path, monkeypatch):
    monkeypatch.setattr(om_module, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("TRADING_MODE", "paper")
    cfg = {"exchange": "myokx", "market_type": "future", "margin_currency": "USDC",
           "risk": {"leverage": 3}, "futures": {"margin_type": "isolated"},
           "backtest": {"commission": 0.0005, "slippage": 0.0002}}
    return Broker(cfg)


def test_broker_loads_futures_markets_for_okx_xperps(broker):
    assert "future" in broker.exchange.options["fetchMarkets"]["types"]


def test_ccxt_parses_xperps_from_public_instruments():
    ex = _exchange_with_fixture()
    m = ex.market("ETH/USD:USD-310404")
    assert m["id"] == "ETH-USD_UM_XPERP-310404" and m["type"] == "future" and m["linear"]
    assert m["contractSize"] == pytest.approx(0.001) and m["limits"]["amount"]["min"] == 1
    assert is_xperp(m) and not is_xperp(ex.market("BTC/USD:BTC-261127"))


def test_xperp_symbol_resolves_by_coin():
    ex = _exchange_with_fixture()
    assert xperp_symbol(ex.markets, "ETH") == "ETH/USD:USD-310404"
    assert xperp_symbol(ex.markets, "BTC") == "BTC/USD:USD-310404"
    assert xperp_symbol(ex.markets, "DOGE") is None


def test_coins_to_contracts_respects_lot_size():
    ex = _exchange_with_fixture()
    coins = 0.0035                                       # ETH; contract = 0.001 ETH, lot = 1 contract
    contracts = float(ex.amount_to_precision("ETH/USD:USD-310404", coins / ex.market("ETH/USD:USD-310404")["contractSize"]))
    assert contracts == 3.0


def test_order_request_targets_the_xperp_instrument():
    """Builds the request ccxt would send. ccxt's lowest-level HTTP method (fetch) is replaced by a
    capture function, so nothing can leave the process; the dummy credentials are never sent."""
    ex = _exchange_with_fixture(apiKey="k", secret="s", password="p")
    calls = []

    def fake_fetch(url, method="GET", headers=None, body=None):
        calls.append({"url": url, "method": method, "body": body})
        return {"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "", "sCode": "0", "sMsg": ""}]}

    ex.fetch = fake_fetch
    ex.create_order("ETH/USD:USD-310404", "market", "buy", 3, params={"marginMode": "isolated"})
    # ccxt sends a single order via /trade/batch-orders (a list with one entry) or /trade/order
    order = [c for c in calls if c["url"].endswith(("/api/v5/trade/order", "/api/v5/trade/batch-orders"))]
    assert len(order) == 1 and order[0]["method"] == "POST"
    body = json.loads(order[0]["body"])
    body = body[0] if isinstance(body, list) else body
    assert body["instId"] == "ETH-USD_UM_XPERP-310404"
    assert body["sz"] == "3" and body["side"] == "buy" and body["tdMode"] == "isolated"
    assert all(c["url"].startswith("https://eea.okx.com") for c in calls)


def test_parse_public_xperp_data():
    x = okx.parse_xperps(_fixture("okx_eea_futures_instruments.json")["data"])
    assert set(x["base"]) == {"BTC", "ETH"}                       # stock X-Perp and dated future excluded
    c = okx.parse_candles(_fixture("okx_eea_eth_xperp_candles_1Dutc.json")["data"])
    assert (c.index.hour == 0).all() and c.index.is_monotonic_increasing    # UTC days
    f = okx.parse_funding(_fixture("okx_eea_eth_xperp_funding_history.json")["data"])
    assert f.index.is_monotonic_increasing and f.abs().max() < 0.01
    assert (f.index.to_series().diff().dropna() == pd.Timedelta("8h")).all()
