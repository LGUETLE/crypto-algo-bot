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


def bar_stop_check(
    side: str,                    # "long" or "short"
    entry_price: float,
    sl_price: float,
    active: bool,
    open_: float,
    high: float,
    low: float,
    activation: Optional[float],
    callback: Optional[float],
    tp_price: Optional[float] = None,
) -> Tuple[Optional[float], str, float, bool]:
    """
    Simulates one OHLC bar against an exchange-side stop / trailing stop / take profit.
    Shared by the backtest engine and the paper broker. Returns
    (raw_exit_price or None, reason, sl_price, trailing_active).

    The order of high and low inside a bar is unknown, so the ADVERSE order is assumed:
      1. the stop in force at the start of the bar is checked first; a gap through it fills at
         the open (it would also be hit before any favourable ratchet)
      2. otherwise the trailing stop arms / ratchets with the favourable extreme (update_trailing)
      3. if the adverse extreme also reaches the ratcheted level, the exchange's trailing stop
         fired inside the bar: exit at that level (the alternative path, adverse extreme first,
         would never give a worse result for this bar)
      4. take profit and a stop both reachable in one bar -> the worse fill (the stop)
    """
    is_long = side == "long"

    # 1. stop in force at the open
    if (low <= sl_price) if is_long else (high >= sl_price):
        raw = min(open_, sl_price) if is_long else max(open_, sl_price)
        return raw, ("TrailingSL" if active else "SL"), sl_price, active

    # 2. trailing ratchet with the favourable extreme
    new_sl, new_active = update_trailing(side, entry_price, sl_price, active, high, low, activation, callback)

    # 3. ratcheted stop inside the bar's range -> hit after the favourable extreme
    trail_hit = new_active and new_sl != sl_price and \
        ((low <= new_sl) if is_long else (high >= new_sl))
    tp_hit = tp_price is not None and ((high >= tp_price) if is_long else (low <= tp_price))

    candidates = []
    if trail_hit:
        candidates.append((new_sl, "TrailingSL"))
    if tp_hit:
        candidates.append((max(open_, tp_price) if is_long else min(open_, tp_price), "TP"))
    if candidates:
        # 4. adverse choice: the lower fill for a long, the higher for a short
        raw, reason = (min if is_long else max)(candidates, key=lambda x: x[0])
        return raw, reason, new_sl, new_active
    return None, "", new_sl, new_active
