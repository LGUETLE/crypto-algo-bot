"""
Order Manager - persistent positions
====================================
Stores the complete position state in SQLite so a restart loses nothing.
Also: PnL bookkeeping, trailing-stop state, sync with the exchange
(detects positions closed by an exchange-side SL / trailing stop, and positions
on the exchange that the bot does not know about).

NOTE: the `positions` table has a `trail_active` column. For an existing paper/demo
DB created by an older version, delete logs/trades.db once so the schema is recreated.
"""
import csv
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrderManager:
    def __init__(self, exchange_id: str = "unknown", fee_rate: float = 0.0006):
        self.exchange_id = exchange_id
        self.fee_rate    = fee_rate        # taker fee per side, used for PnL bookkeeping
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()
        logger.info(f"OrderManager ready | open positions: {len(self.get_open_positions())}")

    # ── DB setup ──────────────────────────────────────────────────────────

    def _init_db(self):
        con = sqlite3.connect(DB_PATH)
        con.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT NOT NULL UNIQUE,      -- one position per symbol
                side         TEXT NOT NULL,             -- buy | sell
                size         REAL NOT NULL,             -- in coins (not contracts)
                entry_price  REAL NOT NULL,
                sl_price     REAL NOT NULL,
                tp_price     REAL,                      -- NULL = no fixed take profit
                margin       REAL NOT NULL,
                leverage     INTEGER DEFAULT 1,
                strategy     TEXT DEFAULT '',
                order_id     TEXT DEFAULT '',
                opened_at    TEXT NOT NULL,
                status       TEXT DEFAULT 'open',
                filled_pct   REAL DEFAULT 1.0,
                trail_active INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                exchange     TEXT NOT NULL,
                symbol       TEXT NOT NULL,
                side         TEXT NOT NULL,
                amount       REAL NOT NULL,
                price        REAL NOT NULL,
                value_usdt   REAL NOT NULL,
                fee          REAL DEFAULT 0,
                fee_currency TEXT DEFAULT 'USDT',
                order_id     TEXT DEFAULT '',
                mode         TEXT DEFAULT 'paper',
                pnl_usdt     REAL,                      -- net of fees
                strategy     TEXT DEFAULT '',
                entry_price  REAL,
                leverage     INTEGER DEFAULT 1,
                margin_type  TEXT DEFAULT 'isolated',
                exit_reason  TEXT DEFAULT '',
                notes        TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS funding (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                symbol        TEXT NOT NULL,
                rate          REAL NOT NULL,
                payment_usdt  REAL NOT NULL,
                position_size REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT NOT NULL UNIQUE,
                balance_usdt   REAL NOT NULL,
                open_positions INTEGER DEFAULT 0,
                daily_pnl      REAL DEFAULT 0,
                cumulative_pnl REAL DEFAULT 0
            );
        """)
        con.commit()
        con.close()

    # ── Positions ─────────────────────────────────────────────────────────

    def open_position(self, symbol, side, size, entry_price, sl_price, tp_price,
                      margin, leverage=1, strategy="", order_id="", filled_pct=1.0):
        """Stores a new open position (long = 'buy', short = 'sell')."""
        con = sqlite3.connect(DB_PATH)
        try:
            con.execute("""
                INSERT OR REPLACE INTO positions
                  (symbol, side, size, entry_price, sl_price, tp_price, margin,
                   leverage, strategy, order_id, opened_at, filled_pct, trail_active)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)
            """, (symbol, side, size, entry_price, sl_price, tp_price, margin,
                  leverage, strategy, order_id, _now(), filled_pct))
            con.commit()
            logger.info(f"Position opened: {side.upper()} {size} {symbol} @ {entry_price}")
        finally:
            con.close()

    def close_position(self, symbol, exit_price, exit_reason="Signal", mode="paper") -> Optional[dict]:
        """Closes a position, logs the trade with net PnL. Returns the trade dict or None."""
        pos = self.get_position(symbol)
        if not pos:
            logger.warning(f"No open position for {symbol}")
            return None

        d       = 1 if pos["side"] == "buy" else -1
        size    = pos["size"]
        gross   = d * (exit_price - pos["entry_price"]) * size   # size in coins: no leverage factor
        fee     = (exit_price + pos["entry_price"]) * size * self.fee_rate   # entry + exit fee
        net_pnl = round(gross - fee, 4)

        con = sqlite3.connect(DB_PATH)
        try:
            con.execute("""
                INSERT INTO trades
                  (timestamp, exchange, symbol, side, amount, price, value_usdt, fee,
                   order_id, mode, pnl_usdt, strategy, entry_price, leverage, exit_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (_now(), self.exchange_id, symbol,
                  "sell" if pos["side"] == "buy" else "buy",
                  size, exit_price, round(size * exit_price, 4), round(fee, 4),
                  pos["order_id"], mode, net_pnl, pos["strategy"],
                  pos["entry_price"], pos["leverage"], exit_reason))
            con.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            con.commit()
        finally:
            con.close()

        logger.info(f"Position closed: {symbol} | PnL: {net_pnl:+.2f} | {exit_reason}")
        return {**pos, "exit_price": exit_price, "pnl_usdt": net_pnl, "exit_reason": exit_reason}

    def get_position(self, symbol: str) -> Optional[dict]:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM positions WHERE symbol=? AND status='open'", (symbol,)).fetchone()
        con.close()
        return dict(row) if row else None

    def get_open_positions(self) -> list[dict]:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM positions WHERE status='open' ORDER BY opened_at").fetchall()
        con.close()
        return [dict(r) for r in rows]

    def update_stop(self, symbol: str, sl_price: float, trail_active: bool):
        """Persists the (trailing) stop level and whether the trailing stop is armed."""
        con = sqlite3.connect(DB_PATH)
        con.execute("UPDATE positions SET sl_price=?, trail_active=? WHERE symbol=?",
                    (sl_price, int(trail_active), symbol))
        con.commit()
        con.close()

    def get_realized_pnl(self, mode: str = "paper") -> float:
        con = sqlite3.connect(DB_PATH)
        total = con.execute("SELECT SUM(pnl_usdt) FROM trades WHERE mode=? AND pnl_usdt IS NOT NULL",
                            (mode,)).fetchone()[0]
        con.close()
        return float(total or 0.0)

    # ── Sync with exchange ────────────────────────────────────────────────

    def sync_with_exchange(self, broker) -> list[str]:
        """
        Checks whether positions we think are open still exist on the exchange.
        If not (exchange-side SL / trailing stop fired): book the close at the real
        last fill price and remove leftover trigger orders. Returns the affected symbols.
        """
        externally_closed = []
        for pos in self.get_open_positions():
            symbol = pos["symbol"]
            try:
                exchange_positions = broker.exchange.fetch_positions([symbol])
                still_open = any(abs(float(p.get("contracts") or 0)) > 0 for p in exchange_positions)
                if still_open:
                    continue

                exit_price = broker.last_fill_price(symbol, fallback=pos["sl_price"])
                broker.cancel_trigger_orders(symbol)
                self.close_position(symbol, exit_price, "Exchange stop/trailing", broker.mode)
                externally_closed.append(symbol)
                logger.info(f"Externally closed position detected: {symbol}")
            except Exception as e:
                logger.error(f"Sync error for {symbol}: {e}")
        return externally_closed

    def find_orphans(self, broker) -> list[str]:
        """Symbols with an open position on the exchange that the bot has no record of
        (e.g. crash between order and DB write, or a manual trade)."""
        try:
            held = {p["symbol"] for p in broker.exchange.fetch_positions()
                    if abs(float(p.get("contracts") or 0)) > 0}
        except Exception as e:
            logger.error(f"Orphan check failed: {e}")
            return []
        known = {p["symbol"] for p in self.get_open_positions()}
        return sorted(held - known)

    # ── Trade log & reporting ─────────────────────────────────────────────

    def get_trades(self, year: Optional[int] = None, mode: str = "live") -> list[dict]:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        if year:
            rows = con.execute("SELECT * FROM trades WHERE mode=? AND strftime('%Y', timestamp)=? ORDER BY timestamp",
                               (mode, str(year))).fetchall()
        else:
            rows = con.execute("SELECT * FROM trades WHERE mode=? ORDER BY timestamp", (mode,)).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def get_summary(self, year: Optional[int] = None, mode: str = "live") -> dict:
        """pnl_usdt is already net of fees, so fees are NOT subtracted again."""
        where, args = "WHERE mode=? AND pnl_usdt IS NOT NULL", [mode]
        if year:
            where += " AND strftime('%Y', timestamp)=?"
            args.append(str(year))

        con = sqlite3.connect(DB_PATH)
        n, wins, pnl, fees = con.execute(f"""
            SELECT COUNT(*), SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END), SUM(pnl_usdt), SUM(fee)
            FROM trades {where}""", args).fetchone()
        fund_q = "SELECT SUM(payment_usdt) FROM funding" + (" WHERE strftime('%Y', timestamp)=?" if year else "")
        funding = con.execute(fund_q, [str(year)] if year else []).fetchone()[0] or 0
        con.close()

        n = n or 0
        return {
            "total_trades":  n,
            "wins":          wins or 0,
            "losses":        n - (wins or 0),
            "total_pnl":     round(pnl or 0, 2),      # net of fees
            "total_fees":    round(fees or 0, 2),
            "total_funding": round(funding, 2),
            "net_result":    round((pnl or 0) + funding, 2),
        }

    def log_snapshot(self, balance, open_pos, daily_pnl, cum_pnl):
        con = sqlite3.connect(DB_PATH)
        con.execute("""INSERT OR REPLACE INTO snapshots
                       (date, balance_usdt, open_positions, daily_pnl, cumulative_pnl)
                       VALUES (date('now'),?,?,?,?)""", (balance, open_pos, daily_pnl, cum_pnl))
        con.commit()
        con.close()

    def log_funding(self, symbol, rate, payment, size=0):
        con = sqlite3.connect(DB_PATH)
        con.execute("INSERT INTO funding (timestamp, symbol, rate, payment_usdt, position_size) VALUES (?,?,?,?,?)",
                    (_now(), symbol, rate, payment, size))
        con.commit()
        con.close()

    def export_tax_csv(self, year: int, path: Optional[str] = None) -> str:
        """CSV export (semicolon separated) for tax tools such as Blockpit / CoinTracking."""
        trades = self.get_trades(year=year, mode="live")
        path = path or os.path.join(os.path.dirname(DB_PATH), f"tax_export_{year}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Date (UTC)", "Exchange", "Symbol", "Type", "Amount", "Price", "Value",
                        "Fee", "PnL", "Strategy", "Order ID", "Exit reason"])
            for t in trades:
                w.writerow([t["timestamp"], t["exchange"], t["symbol"], f"Futures-{t['side'].upper()}",
                            t["amount"], t["price"], t["value_usdt"], t["fee"], t["pnl_usdt"] or "",
                            t["strategy"], t["order_id"], t.get("exit_reason", "")])
        logger.info(f"Tax export: {path} ({len(trades)} trades)")
        return path
