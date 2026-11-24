"""Strategy name -> class. Shared by the live bot (main.get_strategy) and the research pipeline."""


def strategy_class(name: str):
    if name == "donchian_breakout":
        from strategies.donchian_breakout import DonchianBreakoutStrategy
        return DonchianBreakoutStrategy
    if name == "rsi_mean_reversion":
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        return RSIMeanReversionStrategy
    if name == "ema_crossover":
        from strategies.ema_crossover import EmaCrossoverStrategy
        return EmaCrossoverStrategy
    raise ValueError(f"Unknown strategy: {name}")


def get_strategy(config: dict, name: str = None):
    return strategy_class(name or config.get("strategy", "donchian_breakout"))(config)
