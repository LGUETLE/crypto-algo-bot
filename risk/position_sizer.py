class PositionSizer:
    """Risk-based sizing: size = (capital * risk %) / stop distance, capped by available leverage."""

    def __init__(self, config: dict):
        risk = config["risk"]
        self.risk_pct = risk["max_risk_per_trade"]
        self.sl_pct   = risk["stop_loss_pct"]             # fallback if ATR is unavailable
        self.atr_mult = risk.get("atr_stop_mult", 2.0)
        self.tp_pct   = risk.get("take_profit_pct")       # None = no fixed take profit
        self.leverage = risk["leverage"]

    def calculate(self, capital: float, entry_price: float, side: str = "long",
                  atr: float = None, risk_pct: float = None) -> dict:
        """
        side:     "long"/"buy" or "short"/"sell"
        atr:      ATR in price units (None -> fixed-percent stop)
        risk_pct: overrides max_risk_per_trade (used to fit the remaining portfolio risk budget)
        """
        is_long = side in ("long", "buy")
        risk_pct = self.risk_pct if risk_pct is None else risk_pct

        # Stop distance in price units: ATR based, with a fixed-percent fallback
        if atr and atr > 0 and self.atr_mult:
            sl_dist = atr * self.atr_mult
        else:
            sl_dist = entry_price * self.sl_pct

        risk_amount = capital * risk_pct
        # size is in coins; leverage only limits the maximum notional
        size   = min(risk_amount / sl_dist, capital * 0.9 * self.leverage / entry_price)
        margin = size * entry_price / self.leverage

        sl = entry_price - sl_dist if is_long else entry_price + sl_dist
        tp = None
        if self.tp_pct:
            tp = entry_price * (1 + self.tp_pct) if is_long else entry_price * (1 - self.tp_pct)

        return {
            "size":     round(size, 6),
            "margin":   round(margin, 2),
            "sl_price": sl,
            "tp_price": tp,
            "sl_dist":  sl_dist,
            "risk_usd": round(risk_amount, 2),
        }
