"""Trailing stop logic shared by the backtest engine and the live bot (guarantees parity)."""
from typing import Optional, Tuple


def update_trailing(
    side: str,                    # "long" or "short"
    entry_price: float,
    sl_price: float,
    active: bool,
    high: float,
    low: float,
    activation: Optional[float],  # e.g. 0.015 -> arms after +1.5 % in favour
    callback: Optional[float],    # e.g. 0.008 -> stop trails 0.8 % behind the extreme
) -> Tuple[float, bool]:
    """Returns the updated (sl_price, trailing_active). The stop only ever moves in favour."""
    if not activation or not callback:
        return sl_price, active

    if side == "long":
        if not active and high >= entry_price * (1 + activation):
            active = True
        if active:
            sl_price = max(sl_price, high * (1 - callback))
    else:
        if not active and low <= entry_price * (1 - activation):
            active = True
        if active:
            sl_price = min(sl_price, low * (1 + callback))
    return sl_price, active
