"""PSR / DSR against the worked example in Bailey & Lopez de Prado (2014) and basic properties."""
import math

import numpy as np
import pytest

from backtesting.stats import (deflated_sharpe_ratio, expected_max_sharpe,
                               probabilistic_sharpe_ratio)


def test_paper_example():
    """
    Bailey & Lopez de Prado (2014), numerical example: annualised SR 2.5 over T = 1250 daily
    observations (250 days/year), skew -3, kurtosis 10, N = 100 trials whose annualised SRs have
    variance 0.5. The paper reports SR_0 ~ 0.1132 (per day) and DSR ~ 0.90.
    """
    sr    = 2.5 / math.sqrt(250)
    var   = 0.5 / 250
    sr0   = expected_max_sharpe(100, var)
    dsr   = deflated_sharpe_ratio(sr, 1250, -3.0, 10.0, 100, var)
    assert sr0 == pytest.approx(0.1132, abs=1e-4)
    assert dsr == pytest.approx(0.90, abs=0.005)


def test_psr_is_half_at_benchmark():
    assert probabilistic_sharpe_ratio(0.05, 0.05, 1000, 0.3, 6.0) == pytest.approx(0.5)


def test_psr_normal_closed_form():
    sr, n = 0.05, 2000
    expected = 0.5 * (1 + math.erf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr ** 2 / 2) / math.sqrt(2)))
    assert probabilistic_sharpe_ratio(sr, 0.0, n) == pytest.approx(expected)


def test_single_trial_dsr_equals_psr_against_zero():
    assert deflated_sharpe_ratio(0.03, 500, 0.0, 3.0, 1, 0.01) == \
        pytest.approx(probabilistic_sharpe_ratio(0.03, 0.0, 500))


def test_more_trials_deflate_more():
    a = deflated_sharpe_ratio(0.05, 2000, 0.0, 3.0, 10, 1e-3)
    b = deflated_sharpe_ratio(0.05, 2000, 0.0, 3.0, 1000, 1e-3)
    assert b < a


def test_negative_skew_and_fat_tails_lower_psr():
    base = probabilistic_sharpe_ratio(0.08, 0.0, 300, 0.0, 3.0)
    bad  = probabilistic_sharpe_ratio(0.08, 0.0, 300, -2.0, 12.0)
    assert bad < base


def test_monte_carlo_selection_bias_is_removed():
    """Max of N noise strategies looks good by PSR but not by DSR."""
    rng = np.random.default_rng(0)
    n_trials, n_obs = 200, 500
    r   = rng.normal(0, 0.01, (n_trials, n_obs))
    srs = r.mean(axis=1) / r.std(axis=1, ddof=1)
    best = srs.argmax()
    psr = probabilistic_sharpe_ratio(srs[best], 0.0, n_obs)
    dsr = deflated_sharpe_ratio(srs[best], n_obs, 0.0, 3.0, n_trials, srs.var(ddof=1))
    assert psr > 0.95 and dsr < 0.8


def test_sharpe_standard_error_daily_and_years_needed():
    from backtesting.stats import sharpe_standard_error, years_to_confirm
    # one year of daily data: SE of the annualised Sharpe is about 1
    assert sharpe_standard_error(0.0, 365, 365) == pytest.approx(1.0)
    se = sharpe_standard_error(0.84, 1400, 365)
    assert se == pytest.approx(math.sqrt((1 + (0.84 / math.sqrt(365)) ** 2 / 2) / 1400) * math.sqrt(365))
    y = years_to_confirm(0.8)
    assert 0.8 / sharpe_standard_error(0.8, round(y * 365), 365) == pytest.approx(1.645, rel=1e-3)
