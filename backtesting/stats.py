"""
Probabilistic and Deflated Sharpe Ratio
=======================================
Bailey, D. H. and Lopez de Prado, M. (2012), "The Sharpe Ratio Efficient Frontier",
Journal of Risk 15(2)  -> Probabilistic Sharpe Ratio (PSR)
Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management 40(5)

All Sharpe ratios here are NON-annualised (per bar), as in the papers:
  sr        observed Sharpe ratio of the selected strategy (per bar)
  n_obs     number of return observations T
  skew      skewness of the returns (gamma_3)
  kurt      Pearson kurtosis of the returns (gamma_4; 3 for a normal distribution)

  PSR(sr*) = Phi( (sr - sr*) * sqrt(T - 1) / sqrt(1 - gamma_3 * sr + (gamma_4 - 1) / 4 * sr^2) )

  Expected maximum Sharpe ratio of N independent trials with true SR = 0 and cross-trial
  variance V:
  SR_0 = sqrt(V) * ( (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)) ),
  gamma = Euler-Mascheroni constant

  DSR = PSR(SR_0)
"""
import math

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true SR > sr_benchmark) given the observed per-bar SR and the return moments."""
    if n_obs < 2 or not np.isfinite(sr):
        return float("nan")
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, trials_sr_variance: float) -> float:
    """SR_0: expected maximum per-bar SR among n_trials unskilled strategies."""
    if n_trials <= 1 or trials_sr_variance <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trials_sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(sr: float, n_obs: int, skew: float, kurt: float,
                          n_trials: int, trials_sr_variance: float) -> float:
    """PSR against the expected maximum SR of `n_trials` trials (selection-bias corrected)."""
    sr0 = expected_max_sharpe(n_trials, trials_sr_variance)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurt)


def sharpe_standard_error(sr_annual: float, n_obs: int, periods_per_year: float) -> float:
    """
    Standard error of an ANNUALISED Sharpe ratio estimated from n_obs returns (Lo 2002, iid):
    SE(SR_period) = sqrt((1 + SR_period^2 / 2) / T), annualised with sqrt(periods_per_year).
    """
    sr_p = sr_annual / math.sqrt(periods_per_year)
    return math.sqrt((1 + sr_p ** 2 / 2) / n_obs) * math.sqrt(periods_per_year)


def years_to_confirm(sr_annual: float, z: float = 1.645, periods_per_year: float = 365) -> float:
    """Years of data needed before SR_annual / SE(SR) >= z (one-sided 95 % for z = 1.645)."""
    sr_p = sr_annual / math.sqrt(periods_per_year)
    n = (z / sr_p) ** 2 * (1 + sr_p ** 2 / 2)
    return n / periods_per_year
