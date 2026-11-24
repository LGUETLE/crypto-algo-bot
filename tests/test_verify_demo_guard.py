"""Guards of scripts/verify_demo_stops.py (offline; nothing is sent)."""
import importlib.util
import os
import types

import pytest

spec = importlib.util.spec_from_file_location("verify_demo_stops", os.path.join("scripts", "verify_demo_stops.py"))
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)


def _broker(mode="demo", sandbox=True, header=True):
    ex = types.SimpleNamespace(isSandboxModeEnabled=sandbox, headers={"x-simulated-trading": "1"} if header else {})
    return types.SimpleNamespace(mode=mode, exchange=ex)


def test_accepts_only_simulated_trading():
    vd.assert_simulated(_broker())
    for b in (_broker(mode="live"), _broker(sandbox=False), _broker(header=False)):
        with pytest.raises(vd.Abort):
            vd.assert_simulated(b)


def test_real_ccxt_sandbox_sets_header():
    import ccxt
    ex = ccxt.myokx()
    ex.set_sandbox_mode(True)
    vd.assert_simulated(types.SimpleNamespace(mode="demo", exchange=ex))


def test_skips_without_credentials(monkeypatch, capsys):
    for k in ("MYOKX_API_KEY", "MYOKX_SECRET", "MYOKX_PASSPHRASE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    assert vd.run() == 0
    assert "SKIP" in capsys.readouterr().out
