"""config/tsmom.yaml must match the preregistered parameters (docs/preregistration.md)."""
import yaml

from backtesting.portfolio import DEFAULTS, VARIANTS


def test_config_matches_preregistration():
    c = yaml.safe_load(open("config/tsmom.yaml"))
    flat = {**c["universe"], **c["signal"], **c["portfolio"], **c["costs"]}
    for k in ("n_enter", "n_stay", "volume_window", "min_history", "lookbacks", "vol_window", "cov_window",
              "vol_target", "cap_asset", "cap_gross", "band", "capital", "commission",
              "slippage_major", "slippage_other", "majors"):
        assert flat[k] == DEFAULTS[k], k
    assert c["data"]["holdout_start"] == "2025-11-01" and c["validation"]["n_trials"] == 7
    assert set(VARIANTS) == {"V1", "V2", "V3", "V4"}


def test_preregistration_is_committed_and_mentions_trial_count():
    text = open("docs/preregistration.md").read()
    assert "Trial count N = 7" in text and "2025-11-01 to 2026-10-31" in text
