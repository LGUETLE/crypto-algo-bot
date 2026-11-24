"""
Broker (ccxt, exchange-agnostic)
================================
One class for every ccxt venue. The venue is chosen with `exchange:` in config/settings.yaml
(myokx = OKX Europe, okx, bitget, hyperliquid ...).

Modes (TRADING_MODE env var / --mode flag):
  paper  simulated fills and simulated stops, nothing is sent to the exchange
  demo   real order path against the exchange's demo/sandbox account (fake money)
  live   real money

Order path (demo/live):
  - entry with an exchange-side stop loss attached (and optional take profit), computed from the
    last close; after the fill the stop is moved to fill price -/+ stop distance (as in the
    backtest): a reduce-only stop at the fill-based level is placed FIRST, then the estimate-based
    one is cancelled. If either step fails the original stop stays (never unprotected).
  - exchange-side trailing stop order (best effort, the hard SL stays active if it fails)
  - reduce-only market exits, trigger orders cleaned up afterwards
  - sizes are tracked in COINS internally and converted to contracts for the exchange
    (market["contractSize"]), e.g. 1 OKX X-Perp contract = 0.0001 BTC
"""
import os
import logging
from typing import Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv

from execution.order_manager import OrderManager
from risk.stops import bar_stop_check, update_trailing

load_dotenv()
logger = logging.getLogger(__name__)

OKX_IDS = ("okx", "myokx")


class Broker:
    PAPER_START_BALANCE = 1000.0

    def __init__(self, config: dict):
        risk = config["risk"]
        self.config      = config
        self.exchange_id = config["exchange"]
        self.mode        = os.getenv("TRADING_MODE", "paper").lower()    # paper | demo | live
        self.leverage    = risk["leverage"]
        self.margin_type = config["futures"]["margin_type"]
        self.market_type = config.get("market_type", "swap")             # "swap" or "future"
        self.margin_ccy  = config.get("margin_currency", "USDT")
        self.trail_act   = risk.get("trailing_activation")
        self.trail_cb    = risk.get("trailing_callback")
        self.slippage    = config["backtest"]["slippage"]
        self.om          = OrderManager(self.exchange_id, config["backtest"]["commission"])
        self.exchange    = self._build_exchange()

        if self.mode == "paper":
            logger.info("PAPER MODE - simulated, nothing is sent to the exchange")
        elif self.mode == "demo":
            logger.info(f"DEMO MODE - {self.exchange_id} sandbox account")
        else:
            logger.warning(f"LIVE MODE - {self.exchange_id}, real money!")

    def _build_exchange(self):
        params = {"enableRateLimit": True, "options": {"defaultType": self.market_type}}
        prefix = self.exchange_id.upper()

        if self.exchange_id == "hyperliquid":      # wallet based, no API key / passphrase
            params["walletAddress"] = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
            params["privateKey"]    = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        else:
            params["apiKey"] = os.getenv(f"{prefix}_API_KEY", "")
            params["secret"] = os.getenv(f"{prefix}_SECRET", "")
            if os.getenv(f"{prefix}_PASSPHRASE"):
                params["password"] = os.getenv(f"{prefix}_PASSPHRASE")

        exchange = getattr(ccxt, self.exchange_id)(params)
        if self.mode == "demo":
            exchange.set_sandbox_mode(True)
        return exchange

    # ── Balance ───────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Paper: start balance + realized PnL. Demo/live: total equity in the margin currency.
        (Total, not 'free': free balance drops when margin is locked and would falsely
        trigger the daily-loss kill switch.)"""
        if self.mode == "paper":
            return self.PAPER_START_BALANCE + self.om.get_realized_pnl("paper")

        params = {} if self.exchange_id in OKX_IDS else {"type": self.market_type}
        bal = self.exchange.fetch_balance(params)
        entry = bal.get(self.margin_ccy)
        if not entry or entry.get("total") is None:
            raise RuntimeError(
                f"No '{self.margin_ccy}' balance on {self.exchange_id}. "
                f"Available: {list(bal.get('total', {}))} - set margin_currency in config/settings.yaml")
        return float(entry["total"])

    # ── Market data ───────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """The last row is the still-forming candle - callers must drop it with .iloc[:-1]."""
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp").astype(float)

    # ── Entry ─────────────────────────────────────────────────────────────

    def place_order(self, symbol, side, amount, sl_price=None, tp_price=None,
                    entry_price=None, strategy="", margin=0.0, sl_dist=None) -> Optional[dict]:
        """
        Market entry with exchange-side SL (and optional TP). side: 'buy' (long) or 'sell' (short).
        amount is in COINS. Returns the order, or None if nothing was placed.
        sl_dist: stop distance in price units. If given, the stop is set from the ACTUAL fill price
        (fill -/+ sl_dist), like the backtest; sl_price is then only the pre-fill estimate.
        """
        is_long = side == "buy"
        if sl_price is None and sl_dist and entry_price:
            sl_price = entry_price - sl_dist if is_long else entry_price + sl_dist

        if self.mode == "paper":
            price = (entry_price or 0.0) * (1 + self.slippage if is_long else 1 - self.slippage)
            if sl_dist:
                sl_price = price - sl_dist if is_long else price + sl_dist
            order = {"id": "paper", "status": "filled", "amount": amount, "price": price}
            logger.info(f"[PAPER] {side.upper()} {amount:.4f} {symbol} @ {price:.4f} SL={sl_price} TP={tp_price}")
        else:
            result = self._live_entry(symbol, side, amount, sl_price, tp_price, entry_price, sl_dist)
            if result is None:
                return None
            order, price, amount, sl_price = result

        if sl_price:    # long AND short positions are persisted
            self.om.open_position(
                symbol=symbol, side=side, size=amount, entry_price=price,
                sl_price=sl_price, tp_price=tp_price,
                margin=margin or (amount * price / self.leverage),
                leverage=self.leverage, strategy=strategy, order_id=order.get("id", ""),
            )
        return order

    def _live_entry(self, symbol, side, coins, sl_price, tp_price, entry_price, sl_dist=None):
        ex = self.exchange
        if not ex.markets:
            ex.load_markets()
        market        = ex.market(symbol)
        contract_size = market.get("contractSize") or 1.0
        contracts     = float(ex.amount_to_precision(symbol, coins / contract_size))
        min_amount    = ((market.get("limits") or {}).get("amount") or {}).get("min")

        if contracts <= 0 or (min_amount and contracts < min_amount):
            logger.warning(f"{symbol}: {contracts} contracts below exchange minimum {min_amount} - skipped")
            return None

        self._set_leverage(symbol)
        self.cancel_trigger_orders(symbol)         # clear stale orders before a new trade

        params = self._order_params()
        if sl_price:
            params["stopLoss"] = {"triggerPrice": float(ex.price_to_precision(symbol, sl_price)),
                                  "type": "market"}
        if tp_price:
            params["takeProfit"] = {"triggerPrice": float(ex.price_to_precision(symbol, tp_price)),
                                    "type": "market"}

        order = ex.create_market_order(symbol, side, contracts, params=params)
        price = float(order.get("average") or entry_price or 0)
        logger.info(f"[{self.mode.upper()}] {side.upper()} {contracts} contracts {symbol} -> {order['id']} "
                    f"| exchange SL (estimate): {sl_price}")

        if sl_dist and sl_price and price:
            target = price - sl_dist if side == "buy" else price + sl_dist
            sl_price = self._move_stop(symbol, side, contracts, sl_price, target)

        self._place_trailing(symbol, side, contracts, price)
        return order, price, contracts * contract_size, sl_price

    def _move_stop(self, symbol, side, contracts, old_sl, new_sl) -> float:
        """
        Moves the attached (estimate-based) stop to the fill-based level. Places the new
        reduce-only stop FIRST, then cancels the old one, so the position is never unprotected.
        Returns the stop that is in force afterwards. Verify against the demo account first.
        """
        ex = self.exchange
        old_px = float(ex.price_to_precision(symbol, old_sl))
        new_px = float(ex.price_to_precision(symbol, new_sl))
        if new_px == old_px:
            return old_sl
        close_side = "sell" if side == "buy" else "buy"
        try:
            ex.create_order(symbol, "market", close_side, contracts,
                            params={"stopLossPrice": new_px, "reduceOnly": True, **self._order_params()})
        except Exception as e:
            logger.warning(f"{symbol}: could not place the fill-based stop at {new_px} ({e}); "
                           f"keeping the estimate-based stop at {old_px}")
            return old_sl
        try:
            stale = [o for o in ex.fetch_open_orders(symbol, params={"stop": True})
                     if o.get("triggerPrice") is not None and abs(float(o["triggerPrice"]) - old_px) < 1e-12]
            for o in stale:
                ex.cancel_order(o["id"], symbol, params={"stop": True})
            if not stale:
                logger.warning(f"{symbol}: estimate-based stop at {old_px} not found; it may still be "
                               f"active next to the fill-based stop at {new_px}")
        except Exception as e:
            logger.warning(f"{symbol}: could not cancel the estimate-based stop at {old_px} ({e}); "
                           f"both stops are active, the tighter one triggers first")
        logger.info(f"{symbol}: stop moved to fill-based level {new_px} (estimate was {old_px})")
        return new_sl

    def _place_trailing(self, symbol, side, contracts, entry_price):
        """Exchange-side trailing stop (reduce-only). Verify these params in the demo account first."""
        if not (self.trail_act and self.trail_cb):
            return
        is_long = side == "buy"
        trigger = entry_price * (1 + self.trail_act if is_long else 1 - self.trail_act)
        params = {
            "trailingPercent":      self.trail_cb * 100,
            "trailingTriggerPrice": self.exchange.price_to_precision(symbol, trigger),
            "reduceOnly":           True,
            **self._order_params(),
        }
        try:
            self.exchange.create_order(symbol, "market", "sell" if is_long else "buy", contracts, params=params)
            logger.info(f"Trailing stop set: arms at {params['trailingTriggerPrice']}, "
                        f"distance {params['trailingPercent']}%")
        except Exception as e:
            logger.warning(f"Trailing stop order failed (hard SL is still active): {e}")

    # ── Exit ──────────────────────────────────────────────────────────────

    def close_position(self, symbol, current_price, exit_reason="Signal") -> Optional[dict]:
        """Closes the position (reduce-only on the exchange) and removes it from the state."""
        pos = self.om.get_position(symbol)
        if not pos:
            logger.warning(f"No open position for {symbol}")
            return None

        exit_price = current_price
        if self.mode != "paper":
            exit_price = self._live_close(symbol, "sell" if pos["side"] == "buy" else "buy",
                                          pos["size"], current_price)
            if exit_price is None:
                return None
            self.cancel_trigger_orders(symbol)

        return self.om.close_position(symbol, exit_price, exit_reason, self.mode)

    def _live_close(self, symbol, close_side, coins, fallback_price) -> Optional[float]:
        ex = self.exchange
        try:
            if not ex.markets:
                ex.load_markets()
            contract_size = ex.market(symbol).get("contractSize") or 1.0
            contracts     = float(ex.amount_to_precision(symbol, coins / contract_size))
            params        = {"reduceOnly": True, **self._order_params()}
            order         = ex.create_market_order(symbol, close_side, contracts, params=params)
            return float(order.get("average") or fallback_price)
        except Exception as e:
            logger.error(f"Close order failed for {symbol}: {e}")
            return None

    # ── Stop management ───────────────────────────────────────────────────

    def manage_stops(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Call once per new closed candle (df = closed candles only).
        Paper: simulates the stop (same logic as the backtest) and closes if it was hit.
        All modes: ratchets the trailing stop level in the stored state.
        Returns the closed trade dict if the stop was hit, else None.
        """
        pos = self.om.get_position(symbol)
        if not pos:
            return None

        last    = df.iloc[-1]
        is_long = pos["side"] == "buy"

        if self.mode == "paper":
            # same intrabar simulation as the backtest (adverse order, gap fills at the open)
            raw, reason, sl, active = bar_stop_check(
                "long" if is_long else "short", pos["entry_price"], pos["sl_price"],
                bool(pos["trail_active"]), last["open"], last["high"], last["low"],
                self.trail_act, self.trail_cb, pos.get("tp_price"))
            if raw is not None:
                px = raw * (1 - self.slippage if is_long else 1 + self.slippage)
                return self.om.close_position(symbol, px, reason, self.mode)
            if sl != pos["sl_price"] or active != bool(pos["trail_active"]):
                self.om.update_stop(symbol, sl, active)
            return None

        sl, active = update_trailing("long" if is_long else "short", pos["entry_price"],
                                     pos["sl_price"], bool(pos["trail_active"]),
                                     last["high"], last["low"], self.trail_act, self.trail_cb)
        if sl != pos["sl_price"] or active != bool(pos["trail_active"]):
            self.om.update_stop(symbol, sl, active)
        return None

    # ── Exchange helpers (also used by OrderManager.sync_with_exchange) ───

    def _order_params(self) -> dict:
        """OKX takes the margin mode per order; other venues set it per symbol."""
        return {"marginMode": self.margin_type} if self.exchange_id in OKX_IDS else {}

    def _set_leverage(self, symbol: str):
        try:
            if self.exchange_id in OKX_IDS:
                self.exchange.set_leverage(self.leverage, symbol, params={"marginMode": self.margin_type})
            else:
                self.exchange.set_leverage(self.leverage, symbol)
                self.exchange.set_margin_mode(self.margin_type, symbol)
        except Exception as e:
            logger.warning(f"Leverage / margin mode for {symbol}: {e}")

    def cancel_trigger_orders(self, symbol: str):
        """Removes normal, trigger and trailing orders of a symbol (best effort, venue specific)."""
        if self.mode == "paper":
            return
        variants = ({}, {"stop": True}, {"trigger": True}, {"ordType": "move_order_stop"},
                    {"planType": "track_plan"}, {"planType": "moving_plan"})
        for params in variants:
            try:
                self.exchange.cancel_all_orders(symbol, params=params)
                continue
            except Exception:
                pass
            try:    # venues without cancel_all_orders: list and cancel one by one
                ids = [o["id"] for o in self.exchange.fetch_open_orders(symbol, params=params)]
                if ids:
                    self.exchange.cancel_orders(ids, symbol, params=params)
            except Exception:
                pass

    def last_fill_price(self, symbol: str, fallback: float) -> float:
        """Price of the most recent fill (used to book exchange-side stop-outs)."""
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=5)
            return float(trades[-1]["price"]) if trades else fallback
        except Exception:
            return fallback

    # ── State access ──────────────────────────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        return self.om.get_open_positions()

    def has_position(self, symbol: str) -> bool:
        return self.om.get_position(symbol) is not None

    def sync_positions(self) -> list[str]:
        """Demo/live: detect positions closed by an exchange-side SL / trailing stop."""
        if self.mode == "paper":
            return []
        closed = self.om.sync_with_exchange(self)
        if closed:
            logger.info(f"Externally closed positions: {closed}")
        return closed
