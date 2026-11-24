"""Statistical power of the TSMOM results (RESEARCH.md, Part 2). python scripts/power_analysis.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.stats import sharpe_standard_error as se, years_to_confirm  # noqa: E402

rows = [("Hold-out (1 year, V1)", -0.09, 365), ("W1 stitched OOS (3.8 years)", 0.84, 1400),
        ("R1 Binance-wide OOS (3.8 years)", 0.66, 1400)]
print(f"{'Sample':<34} {'SR':>6} {'T days':>7} {'SE':>6} {'95% CI (two-sided)':>22}")
for name, sr, t in rows:
    s = se(sr, t, 365)
    print(f"{name:<34} {sr:>6.2f} {t:>7} {s:>6.2f}   [{sr - 1.96 * s:+.2f}, {sr + 1.96 * s:+.2f}]")
print()
for sr in (0.84, 0.8, 0.66, 0.5):
    print(f"Years to confirm SR {sr:.2f} > 0: one-sided 95 % {years_to_confirm(sr):5.1f} y, "
          f"two-sided 95 % {years_to_confirm(sr, 1.96):5.1f} y")
print(f"Years to separate SR 0.84 from vol-matched BTC 0.69 (IR 0.11, one-sided 95 %): "
      f"{years_to_confirm(0.11):.0f} y")
print(f"Smallest SR a 1-year hold-out confirms at one-sided 95 %: {1.645 * se(0, 365, 365):.2f}")
