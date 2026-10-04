"""
crypto-algo-bot - entry point
=============================
python main.py --mode backtest [--symbol ETH]
python main.py --mode optimize [--symbol ETH]
python main.py --mode paper      simulated fills and stops
python main.py --mode demo       exchange demo account (real order path, fake money)
python main.py --mode live       real money
"""
import argparse, logging, os, time, yaml
import pandas as pd
import pandas_ta as ta
import schedule
from dotenv import load_dotenv

from features.macro import add_macro_trend

load_dotenv()
os.makedirs("logs", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/bot.log")],
)
logger = logging.getLogger(__name__)


def load_config(path="config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs(path="config/pairs.yaml") -> list:
    with open(path) as f:
        return [p for p in yaml.safe_load(f)["pairs"] if p["active"]]


def macro_timeframe(config: dict):
    """Returns the guardian timeframe, or None if the filter is disabled."""
    return config.get("macro_timeframe") if config.get("use_macro_filter", True) else None


def get_strategy(config: dict, strategy_name: str = None):
    name = strategy_name or config.get("strategy", "donchian_breakout")
    if name == "donchian_breakout":
        from strategies.donchian_breakout import DonchianBreakoutStrategy
        return DonchianBreakoutStrategy(config)
    if name == "rsi_mean_reversion":
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        return RSIMeanReversionStrategy(config)
    if name == "ema_crossover":
        from strategies.ema_crossover import EmaCrossoverStrategy
        return EmaCrossoverStrategy(config)
    raise ValueError(f"Unknown strategy: {name}")


def latest_atr(df: pd.DataFrame, length: int = 14):
    atr = ta.atr(df["high"], df["low"], df["close"], length=length)
    if atr is None or pd.isna(atr.iloc[-1]):
        return None
    return float(atr.iloc[-1])


# ── Trading cycle ─────────────────────────────────────────────────────────────

def run_cycle(broker, strategies, guard, sizer, pairs, config):
    from notify.telegram_bot import send, notify_entry, notify_exit, notify_error, notify_kill_switch

    # Reconcile with the exchange first (demo/live only)
    closed = broker.sync_positions()
    if closed:
        send(f"Closed by exchange stop/trailing stop: {closed}")
    if broker.mode != "paper":
        orphans = broker.om.find_orphans(broker)
        if orphans:
            logger.error(f"Positions on the exchange unknown to the bot: {orphans}")
            send(f"WARNING: positions on the exchange unknown to the bot: {orphans}")

    capital  = broker.get_balance()
    macro_tf = macro_timeframe(config)
    min_risk = config["risk"].get("min_trade_risk_pct", 0.005)

    # The kill switch only blocks NEW entries; open positions are always managed.
    if guard.killed(capital) and guard.just_killed:
        guard.just_killed = False
        notify_kill_switch("Daily loss limit reached - no new entries until the next UTC day")

    for pair in pairs:
        symbol     = pair["symbol"]                    # symbol on the trading venue
        strat_name = pair.get("strategy", config.get("strategy", "donchian_breakout"))
        strategy   = strategies.get(strat_name)
        if not strategy:
            continue
        try:
            # Closed candles only: the last row from the exchange is still forming
            df = broker.get_ohlcv(symbol, config["timeframe"], limit=300).iloc[:-1]
            if macro_tf:
                macro = broker.get_ohlcv(symbol, macro_tf, limit=60).iloc[:-1]
                df = add_macro_trend(df, macro, macro_tf)
            price = float(df["close"].iloc[-1])

            # 1) Trailing stop update / simulated stop in paper mode
            stopped = broker.manage_stops(symbol, df)
            if stopped:
                notify_exit(symbol, stopped["pnl_usdt"], stopped["exit_reason"], broker.mode)

            # 2) Signal logic
            signal = strategy.get_latest_signal(df)
            pos    = broker.om.get_position(symbol)

            if pos:
                is_long  = pos["side"] == "buy"
                opposite = signal == (-1 if is_long else 1)
                if opposite or strategy.should_exit(df, "long" if is_long else "short"):
                    result = broker.close_position(symbol, price, "Signal")
                    if result:
                        notify_exit(symbol, result["pnl_usdt"], "Signal", broker.mode)

            elif signal in (1, -1):
                open_positions = broker.get_open_positions()
                if not guard.can_open(len(open_positions)):
                    continue

                # Portfolio cap: shrink the trade to the remaining risk budget, or skip it
                budget = guard.risk_budget(capital, open_positions)
                if budget < min_risk:
                    logger.info(f"{symbol}: signal skipped, portfolio risk budget used up ({budget:.2%} left)")
                    continue

                side   = "buy" if signal == 1 else "sell"
                sizing = sizer.calculate(capital, price, side, latest_atr(df),
                                         risk_pct=min(sizer.risk_pct, budget))
                order  = broker.place_order(
                    symbol=symbol, side=side, amount=sizing["size"],
                    sl_price=sizing["sl_price"], tp_price=sizing["tp_price"],
                    entry_price=price, strategy=strat_name, margin=sizing["margin"])
                if order:
                    notify_entry(symbol, side, price, sizing["size"],
                                 sizing["sl_price"], sizing["tp_price"], broker.mode)

        except Exception as e:
            logger.error(f"{symbol}: {e}", exc_info=True)
            notify_error(f"{symbol}: {e}")


def daily_summary(broker):
    from notify.telegram_bot import notify_daily_summary
    capital  = broker.get_balance()
    summary  = broker.om.get_summary(mode=broker.mode)
    win_rate = (summary["wins"] / max(summary["total_trades"], 1)) * 100
    broker.om.log_snapshot(capital, len(broker.get_open_positions()),
                           summary["total_pnl"], summary["net_result"])
    notify_daily_summary(capital, summary["total_pnl"], summary["total_trades"], win_rate)


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(config, pairs, symbol_filter=None):
    from data.fetcher import DataFetcher
    from features.engineer import FeatureEngineer
    from backtesting.engine import BacktestEngine

    fetcher = DataFetcher(config["exchange"])
    engine  = BacktestEngine(config)
    days    = config["backtest"]["lookback_days"]
    target  = [p for p in pairs if symbol_filter is None or symbol_filter.upper() in p["symbol"]]

    for pair in target:
        data_symbol = pair.get("data_symbol", pair["symbol"])    # Binance history as price proxy
        strat_name  = pair.get("strategy", config.get("strategy"))

        print(f"\n{'=' * 60}\n  BACKTEST: {data_symbol} | {strat_name} | {days} days\n{'=' * 60}")

        df = fetcher.historical(data_symbol, config["timeframe"], days=days, macro_tf=macro_timeframe(config))
        df = FeatureEngineer(config).transform(df)
        strategy = get_strategy(config, strat_name)

        engine.run(df, strategy, symbol=data_symbol)
        wf = engine.walk_forward(df, strategy, symbol=data_symbol)
        logger.info(f"OOS consistency: {wf['consistency']:.0%} | avg OOS Sharpe: {wf['avg_oos_sharpe']:.2f}")


# ── Optimizer ─────────────────────────────────────────────────────────────────

def run_optimize(config, pairs, symbol_filter=None):
    from data.fetcher import DataFetcher
    from features.engineer import FeatureEngineer
    from backtesting.engine import BacktestEngine
    from backtesting.optimizer import ParameterOptimizer

    fetcher   = DataFetcher(config["exchange"])
    engine    = BacktestEngine(config)
    optimizer = ParameterOptimizer(config, engine)
    days      = config["backtest"]["lookback_days"]
    target    = [p for p in pairs if symbol_filter is None or symbol_filter.upper() in p["symbol"]]

    for pair in target:
        data_symbol = pair.get("data_symbol", pair["symbol"])
        strat_name  = pair.get("strategy", config.get("strategy"))

        print(f"\n{'=' * 60}\n  OPTIMIZATION: {data_symbol} | {strat_name} | {days} days\n{'=' * 60}")

        df = fetcher.historical(data_symbol, config["timeframe"], days=days, macro_tf=macro_timeframe(config))
        df = FeatureEngineer(config).transform(df)

        if strat_name == "donchian_breakout":
            from strategies.donchian_breakout import DonchianBreakoutStrategy
            strategy_class = DonchianBreakoutStrategy
            param_grid = {
                "dc_enter":      [20, 30, 40, 50],
                "dc_exit":       [5, 10, 15],
                "adx_threshold": [20, 25, 30],
                "atr_min_mult":  [0.3, 0.5, 0.7],
            }
        elif strat_name == "rsi_mean_reversion":
            from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
            strategy_class = RSIMeanReversionStrategy
            param_grid = {
                "rsi_oversold":   [25, 30, 35],
                "rsi_overbought": [65, 70, 75],
                "rsi_exit_long":  [50, 55, 60],
                "rsi_exit_short": [40, 45, 50],
                "adx_max":        [25, 30, 35],
            }
        else:
            logger.error(f"No param_grid for {strat_name}")
            continue

        results = optimizer.grid_search(df, strategy_class, param_grid, min_trades=20, symbol=data_symbol)
        if results.empty:
            logger.error(f"No valid results for {data_symbol}")
            continue

        best = optimizer.best_params(results, metric="sharpe")
        logger.info(f"Best parameters for {data_symbol}: {best}")
        optimizer.validate_best(df, strategy_class, best, symbol=data_symbol)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "demo", "live", "backtest", "optimize"], default="paper")
    parser.add_argument("--symbol", default=None)
    args   = parser.parse_args()
    config = load_config()
    pairs  = load_pairs()

    logger.info("=" * 50)
    logger.info(f"Bot starting | mode: {args.mode.upper()} | exchange: {config['exchange']}")
    logger.info(f"Pairs: {[(p['symbol'], p.get('strategy', '?')) for p in pairs]}")
    logger.info("=" * 50)

    if args.mode == "backtest":
        run_backtest(config, pairs, symbol_filter=args.symbol)
        return
    if args.mode == "optimize":
        run_optimize(config, pairs, symbol_filter=args.symbol)
        return

    from execution.broker import Broker
    from risk.guard import RiskGuard
    from risk.position_sizer import PositionSizer

    os.environ["TRADING_MODE"] = args.mode      # the --mode flag overrides .env

    broker  = Broker(config)
    capital = broker.get_balance()
    guard   = RiskGuard(config, capital)
    sizer   = PositionSizer(config)

    strategies = {}
    for pair in pairs:
        name = pair.get("strategy", config.get("strategy"))
        if name not in strategies:
            strategies[name] = get_strategy(config, name)

    logger.info(f"Start capital: {capital:.2f} {config.get('margin_currency', '')} | "
                f"strategies: {list(strategies.keys())}")

    run_cycle(broker, strategies, guard, sizer, pairs, config)
    schedule.every().hour.at(":01").do(run_cycle, broker, strategies, guard, sizer, pairs, config)
    schedule.every().day.at("00:01").do(daily_summary, broker)

    logger.info("Bot running... Ctrl+C to stop")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
