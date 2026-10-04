import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RiskGuard:
    """
    Entry gate. Exits must never depend on it.

      killed(capital)                 daily loss limit hit? (no NEW entries until the next UTC day)
      can_open(n_open)                is there a free position slot?
      risk_budget(capital, positions) how much more risk (fraction of capital) fits under the
                                      portfolio cap, measured as the sum of all stop-out losses
    """

    def __init__(self, config: dict, initial_capital: float):
        risk = config["risk"]
        self.max_daily_loss = risk["max_daily_loss_pct"]
        self.max_positions  = risk["max_open_positions"]
        self.max_total_risk = risk.get("max_total_risk_pct")    # None = no portfolio cap
        self.day_start      = initial_capital
        self.current_day    = datetime.now(timezone.utc).date()
        self.is_killed      = False
        self.just_killed    = False    # lets the caller send a single notification

    # ── Daily loss kill switch ────────────────────────────────────────────

    def reset_daily(self, capital: float):
        today = datetime.now(timezone.utc).date()
        if today != self.current_day:
            self.day_start   = capital
            self.current_day = today
            self.is_killed   = False
            logger.info(f"Daily reset: {capital:.2f}")

    def killed(self, capital: float) -> bool:
        self.reset_daily(capital)
        if not self.is_killed:
            daily_pnl = (capital - self.day_start) / self.day_start
            if daily_pnl < -self.max_daily_loss:
                logger.error(f"KILL SWITCH: daily PnL {daily_pnl:.1%}")
                self.is_killed = True
                self.just_killed = True
        return self.is_killed

    # ── Position slots ────────────────────────────────────────────────────

    def can_open(self, n_open: int) -> bool:
        if self.is_killed:
            return False
        if n_open >= self.max_positions:
            logger.info(f"Max positions reached: {n_open}/{self.max_positions}")
            return False
        return True

    # ── Portfolio risk cap ────────────────────────────────────────────────

    @staticmethod
    def open_risk(positions: list, capital: float) -> float:
        """Loss if every open position hit its stop, as a fraction of capital.
        A trailing stop that already locks in profit counts as zero risk (never negative)."""
        total = 0.0
        for p in positions:
            d = 1 if p["side"] == "buy" else -1
            total += max(0.0, d * (p["entry_price"] - p["sl_price"])) * p["size"]
        return total / capital

    def risk_budget(self, capital: float, positions: list) -> float:
        """Remaining risk fraction before the portfolio cap is reached (inf if no cap is set)."""
        if self.max_total_risk is None:
            return float("inf")
        return max(0.0, self.max_total_risk - self.open_risk(positions, capital))
