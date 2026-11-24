"""Data fetcher tests. Offline except the `slow` ones (which hit the public Binance archive)."""
import hashlib
import io
import zipfile

import pandas as pd
import pytest

from data.fetcher import DataFetcher, binance_id, perp_symbol

KL_HEADER = b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
KL_ROWS = (b"1609459200000,737.18,740.00,730.00,734.60,98092.526,1609462799999,1,1,1,1,0\n"
           b"1609462800000,734.61,750.40,734.00,749.33,164084.062,1609466399999,1,1,1,1,0\n")
FUND = (b"calc_time,funding_interval_hours,last_funding_rate\n"
        b"1609459200002,8,0.00034491\n1609488000006,8,-0.00061141\n")


def _zip(name, payload):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, payload)
    return buf.getvalue()


class Resp:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content

    @property
    def text(self):
        return self.content.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Serves a dict {url: bytes}; also serves <url>.CHECKSUM automatically (optionally corrupted)."""

    def __init__(self, files, corrupt=False):
        self.files, self.corrupt, self.calls = files, corrupt, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url.endswith(".CHECKSUM"):
            data = self.files.get(url[:-9])
            digest = hashlib.sha256(data).hexdigest() if data is not None else ""
            if self.corrupt:
                digest = "0" * 64
            return Resp(200, f"{digest}  file.zip\n".encode())
        if url in self.files:
            return Resp(200, self.files[url])
        return Resp(404)


class NoRest:
    """Stands in for ccxt.binance when the REST API is geo-blocked."""

    def fetch_ohlcv(self, *a, **k):
        raise RuntimeError("451 Unavailable For Legal Reasons")

    def fetch_funding_rate_history(self, *a, **k):
        raise RuntimeError("451 Unavailable For Legal Reasons")


def test_symbol_helpers():
    assert binance_id("ETH/USDT:USDT") == "ETHUSDT"
    assert perp_symbol("ETH/USDT") == "ETH/USDT:USDT"
    assert perp_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"


@pytest.mark.parametrize("payload", [KL_ROWS, KL_HEADER + KL_ROWS])
def test_parse_klines_with_and_without_header(payload):
    df = DataFetcher.parse_archive_klines(payload)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2021-01-01 00:00", tz="UTC")
    assert df["close"].iloc[1] == pytest.approx(749.33)


def test_parse_klines_microseconds():
    payload = KL_ROWS.replace(b"1609459200000,", b"1609459200000000,").replace(
        b"1609462800000,", b"1609462800000000,")
    df = DataFetcher.parse_archive_klines(payload)
    assert df.index[1] == pd.Timestamp("2021-01-01 01:00", tz="UTC")


def test_parse_funding_floors_to_minute_and_keeps_sign():
    s = DataFetcher.parse_archive_funding(FUND)
    assert s.index[0] == pd.Timestamp("2021-01-01 00:00", tz="UTC")
    assert s.index[1] == pd.Timestamp("2021-01-01 08:00", tz="UTC")
    assert s.iloc[1] == pytest.approx(-0.00061141)


def _archive_session(**kw):
    base = "https://data.binance.vision/data/futures/um/monthly"
    return FakeSession({
        f"{base}/klines/ETHUSDT/1h/ETHUSDT-1h-2021-01.zip": _zip("a.csv", KL_ROWS),
        f"{base}/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2021-01.zip": _zip("f.csv", FUND),
    }, **kw)


def test_falls_back_to_archive_when_rest_is_blocked(tmp_path):
    f = DataFetcher(cache_dir=str(tmp_path), session=_archive_session())
    f._hist = NoRest()
    df = f.ohlcv_range("ETH/USDT:USDT", "1h", "2021-01-01", "2021-01-01 02:00")
    assert len(df) == 2
    fr = f.funding_history("ETH/USDT:USDT", "2021-01-01", "2021-01-02")
    assert len(fr) == 2 and fr.iloc[0] == pytest.approx(0.00034491)


def test_range_is_end_exclusive(tmp_path):
    f = DataFetcher(cache_dir=str(tmp_path), session=_archive_session())
    f._hist = NoRest()
    df = f.ohlcv_range("ETH/USDT:USDT", "1h", "2021-01-01", "2021-01-01 01:00")
    assert list(df.index) == [pd.Timestamp("2021-01-01 00:00", tz="UTC")]


def test_checksum_mismatch_raises(tmp_path):
    f = DataFetcher(cache_dir=str(tmp_path), session=_archive_session(corrupt=True))
    f._hist = NoRest()
    with pytest.raises(ValueError, match="Checksum"):
        f.ohlcv_range("ETH/USDT:USDT", "1h", "2021-01-01", "2021-01-01 02:00")


def test_fixed_past_window_is_cached(tmp_path):
    session = _archive_session()
    f = DataFetcher(cache_dir=str(tmp_path), session=session)
    f._hist = NoRest()
    f.ohlcv_range("ETH/USDT:USDT", "1h", "2021-01-01", "2021-01-01 02:00")
    n = len(session.calls)
    f.ohlcv_range("ETH/USDT:USDT", "1h", "2021-01-01", "2021-01-01 02:00")
    assert len(session.calls) == n


@pytest.mark.slow
def test_real_archive_download(tmp_path):
    f = DataFetcher(cache_dir=str(tmp_path))
    df = f.ohlcv_range("ETH/USDT:USDT", "1h", "2024-01-01", "2024-02-01")
    assert len(df) == 31 * 24
    fr = f.funding_history("ETH/USDT:USDT", "2024-01-01", "2024-02-01")
    assert len(fr) == 31 * 3


def test_cache_written_before_window_end_is_not_immutable(tmp_path):
    """A cache file written while the window was still open holds truncated data; it must be
    refreshed later instead of being treated as an immutable past window."""
    import os
    path = tmp_path / "x.parquet"
    path.write_bytes(b"x")
    end = pd.Timestamp("2021-01-01 02:00", tz="UTC")
    written = (end - pd.Timedelta(days=10)).timestamp()            # written 10 days before `end`
    os.utime(path, (written, written))
    assert not DataFetcher._cache_valid(str(path), end)
    complete = (end + pd.Timedelta(hours=1)).timestamp()            # written after the window closed
    os.utime(path, (complete, complete))
    assert DataFetcher._cache_valid(str(path), end)
