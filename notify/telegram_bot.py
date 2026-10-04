import os, requests, logging
from dotenv import load_dotenv

load_dotenv()
logger  = logging.getLogger(__name__)
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send(msg: str, silent: bool = False) -> bool:
    if not TOKEN or not CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown",
                  "disable_notification": silent},
            timeout=5,
        )
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram: {e}")
        return False


def notify_entry(symbol, side, price, size, sl, tp, mode):
    emoji  = "🟢" if side == "buy" else "🔴"
    tp_txt = f"{tp:.4f}" if tp else "none"
    send(f"{emoji} *{mode.upper()} ENTRY*\n"
         f"Symbol: `{symbol}`\nSide: `{side.upper()}`\n"
         f"Price: `{price:.4f}`\nSize: `{size:.4f}`\n"
         f"SL: `{sl:.4f}` | TP: `{tp_txt}`")


def notify_exit(symbol, pnl, exit_reason, mode):
    emoji = "✅" if pnl >= 0 else "❌"
    send(f"{emoji} *{mode.upper()} EXIT* ({exit_reason})\n"
         f"Symbol: `{symbol}`\nPnL: `{pnl:+.2f}` USDT")


def notify_kill_switch(reason):
    send(f"🛑 *KILL SWITCH*\n{reason}")


def notify_error(error):
    send(f"⚠️ *Error*\n`{error}`")


def notify_daily_summary(balance, daily_pnl, trades, win_rate):
    emoji = "📈" if daily_pnl >= 0 else "📉"
    send(f"{emoji} *Daily summary*\n"
         f"Balance: `${balance:,.2f}`\n"
         f"PnL: `{daily_pnl:+.2f}` USDT\n"
         f"Trades: `{trades}` | Win rate: `{win_rate:.0f}%`", silent=True)
