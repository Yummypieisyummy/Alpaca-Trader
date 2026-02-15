"""
ALPACA SHORT-INTERVAL BOT
=========================
Mean-reversion bot for short-interval trades (default: 1Min)
Uses VWAP band + RSI for entries and multi-interval holds

Usage:
    from alpaca_live_runner_short import AlpacaShortIntervalBot
    bot = AlpacaShortIntervalBot(symbol="NVDA", mode="paper")
    bot.run()
"""

import logging
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from alpaca_trader import AlpacaTrader

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_short_interval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlpacaShortIntervalBot:
    """
    Short-interval mean-reversion bot for Alpaca

    Entry:
      - Price below rolling VWAP band
      - RSI below entry threshold

    Exit:
      - Price reverts to VWAP or RSI mean
      - Take profit / stop loss
      - Optional max hold
    """

    def __init__(
        self,
        symbol="NVDA",
        mode="paper",
        timeframe="1Min",
        check_interval=60,
    ):
        self.trader = AlpacaTrader(symbol=symbol, mode=mode)
        self.symbol = symbol
        self.mode = mode
        self.timeframe = timeframe
        self.check_interval = check_interval

        # Strategy params
        self.rsi_period = 14
        self.rsi_entry = 35
        self.rsi_exit = 55
        self.vwap_window = 60  # rolling bars
        self.vwap_band = 0.0025  # 0.25% below VWAP

        # Risk params
        self.position_pct = 0.05  # 5% of equity per trade
        self.take_profit_pct = 0.0035  # 0.35%
        self.stop_loss_pct = 0.0040  # 0.40%
        self.max_hold_minutes = 90
        self.cooldown_minutes = 5

        # State
        self.entry_price = None
        self.entry_time = None
        self.last_exit_time = None

        logger.info(
            f"AlpacaShortIntervalBot initialized for {symbol} in {mode} mode"
        )

    def calculate_rsi(self, series, periods=14):
        """Calculate RSI for a price series"""
        if len(series) < periods + 1:
            return pd.Series(dtype=float)

        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(periods).mean()
        avg_loss = loss.rolling(periods).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_rolling_vwap(self, bars, window=60):
        """Calculate rolling VWAP using typical price"""
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
        pv = typical_price * bars["volume"]
        vwap = pv.rolling(window).sum() / bars["volume"].rolling(window).sum()
        return vwap

    def _in_cooldown(self, now):
        if self.last_exit_time is None:
            return False
        return (now - self.last_exit_time) < timedelta(minutes=self.cooldown_minutes)

    def _set_entry_state_from_position(self, position, now):
        if position and self.entry_price is None:
            self.entry_price = float(position["avg_fill_price"])
            self.entry_time = now

    def _calculate_position_dollars(self, equity, buying_power):
        target = equity * self.position_pct
        cap = buying_power * 0.90
        return max(0.0, min(target, cap))

    def run(self, days_back=5):
        logger.info("=" * 60)
        logger.info("ALPACA SHORT-INTERVAL BOT STARTED")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        logger.info(f"Check Interval: {self.check_interval} seconds")

        iteration = 0

        try:
            while True:
                iteration += 1
                now = datetime.now()

                position = self.trader.get_position()
                self._set_entry_state_from_position(position, now)

                bars = self.trader.get_historical_bars(
                    timeframe=self.timeframe, days_back=days_back
                )

                if bars is None or len(bars) < max(self.vwap_window, self.rsi_period) + 5:
                    logger.warning("Not enough bars yet; waiting for more data")
                    time.sleep(self.check_interval)
                    continue

                price = float(bars["close"].iloc[-1])
                rsi = self.calculate_rsi(bars["close"], self.rsi_period).iloc[-1]
                vwap = self.calculate_rolling_vwap(bars, self.vwap_window).iloc[-1]

                if pd.isna(rsi) or pd.isna(vwap):
                    logger.warning("Indicators not ready; waiting for next bar")
                    time.sleep(self.check_interval)
                    continue

                logger.info(
                    f"\n[Iter {iteration}] {now} | Price: ${price:.2f} | "
                    f"VWAP: ${vwap:.2f} | RSI: {rsi:.1f}"
                )

                entry_signal = (
                    price < vwap * (1 - self.vwap_band)
                    and rsi < self.rsi_entry
                    and not position
                    and not self._in_cooldown(now)
                )

                if entry_signal:
                    account = self.trader.get_account_info()
                    position_dollars = self._calculate_position_dollars(
                        equity=float(account["equity"]),
                        buying_power=float(account["buying_power"]),
                    )

                    if position_dollars <= 0:
                        logger.info("[ENTRY SKIP] Position size too small")
                    else:
                        logger.info(
                            f"[ENTRY] Mean reversion | Size: ${position_dollars:.2f}"
                        )
                        order = self.trader.place_buy_order_dollars(position_dollars)
                        if order:
                            self.entry_price = price
                            self.entry_time = now
                            logger.info("✓ BUY ORDER PLACED")

                elif position:
                    if self.entry_price is None:
                        self.entry_price = float(position["avg_fill_price"])
                        self.entry_time = now

                    pnl_pct = (price - self.entry_price) / self.entry_price
                    hold_minutes = (
                        (now - self.entry_time).total_seconds() / 60.0
                        if self.entry_time else 0
                    )

                    exit_signal = (
                        price >= vwap
                        or rsi >= self.rsi_exit
                        or pnl_pct >= self.take_profit_pct
                        or pnl_pct <= -self.stop_loss_pct
                        or (self.max_hold_minutes and hold_minutes >= self.max_hold_minutes)
                    )

                    if exit_signal:
                        logger.info(
                            f"[EXIT] Reversion/TP/SL | PnL: {pnl_pct:.2%}"
                        )
                        order = self.trader.close_position()
                        if order:
                            self.last_exit_time = now
                            self.entry_price = None
                            self.entry_time = None
                            logger.info("✓ SELL ORDER PLACED")
                else:
                    logger.info("[NO POSITION] Waiting for entry")

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\n[STOP] Bot stopped by user")
            self.trader.export_trade_log("trade_log_short_interval.json")

        except Exception as e:
            logger.error(f"[ERROR] {e}", exc_info=True)
            self.trader.export_trade_log("trade_log_short_interval.json")


if __name__ == "__main__":
    bot = AlpacaShortIntervalBot(symbol="NVDA", mode="paper")
    bot.run(days_back=5)
