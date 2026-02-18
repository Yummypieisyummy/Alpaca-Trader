"""
ALPACA MULTI-SYMBOL 750 EMA BOT (MINIMAL)
Trades NVDA, TSLA, XLK with 750 EMA on 1-min bars from yfinance
"""
import logging
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca_trade_api import REST
import pandas as pd
import numpy as np
import yfinance as yf

# === SETUP ===
load_dotenv("API.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ENDPOINT = os.getenv("ENDPOINT", "https://paper-api.alpaca.markets/v2")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('alpaca_min.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SimpleBot:
    def __init__(self, symbols, mode="paper"):
        self.symbols = symbols
        self.mode = mode
        self.base_url = ENDPOINT.replace("/v2", "")
        self.client = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=self.base_url)
        self.entry_prices = {s: None for s in symbols}
        logger.info(f"Bot initialized: {symbols}")

    def get_bars_yfinance(self, symbol):
        """Fetch 8 days of 1-min bars from yfinance"""
        try:
            logger.info(f"{symbol}: Fetching 8 days of 1-min bars...")
            bars = yf.download(symbol, period="8d", interval="1m", progress=False)
            
            if bars is None or len(bars) == 0:
                logger.warning(f"{symbol}: No bars returned")
                return None
            
            # Handle multi-index columns (single symbol returns Series, multiple return DataFrame)
            if isinstance(bars.columns, pd.MultiIndex):
                bars.columns = bars.columns.get_level_values(0)
            
            # Ensure lowercase column names
            bars.columns = [c.lower() for c in bars.columns]
            
            if len(bars) >= 750:
                logger.info(f"{symbol}: Got {len(bars)} bars with columns: {list(bars.columns)}")
                return bars
            else:
                logger.warning(f"{symbol}: Only got {len(bars)} bars, need 750+")
                return None
        except Exception as e:
            logger.error(f"{symbol} error: {e}")
            return None

    def calculate_slope(self, series, periods=100):
        """Calculate linear regression slope (100-bar default for noise reduction)"""
        if len(series) < periods:
            return 0.0
        recent = series.tail(periods).values
        x = np.arange(len(recent))
        try:
            slope = np.polyfit(x, recent, 1)[0]
            return float(slope)
        except:
            return 0.0

    def run(self):
        logger.info("="*70)
        logger.info("ALPACA MULTI-SYMBOL 750 EMA BOT STARTED")
        logger.info("="*70)
        
        iteration = 0
        
        while True:
            iteration += 1
            timestamp = datetime.now()
            
            # Get account equity
            try:
                account = self.client.get_account()
                capital = float(account.equity)
            except Exception as e:
                logger.error(f"Account error: {e}")
                time.sleep(300)
                continue
            
            logger.info(f"\n[Iteration {iteration}] {timestamp} | Equity: ${capital:,.0f}")
            
            # Process each symbol
            for symbol in self.symbols:
                try:
                    # Get latest price
                    quote = self.client.get_latest_trade(symbol)
                    price = float(quote.price)
                    
                    # Get bars
                    bars = self.get_bars_yfinance(symbol)
                    if bars is None or len(bars) < 750:
                        logger.warning(f"{symbol}: Insufficient data, skipping")
                        continue
                    
                    # Ensure closing price exists
                    if 'close' not in bars.columns:
                        logger.error(f"{symbol}: 'close' column not found. Columns: {list(bars.columns)}")
                        continue
                    
                    # Calculate 750 EMA, short EMA, and slope
                    ema_750 = bars['close'].ewm(span=750).mean().iloc[-1]
                    ema_50 = bars['close'].ewm(span=50).mean().iloc[-1]
                    slope = self.calculate_slope(bars['close'], periods=100)
                    
                    # Regime detection
                    ema_distance = abs(ema_750 - ema_50) / ema_750
                    is_choppy = ema_distance < 0.0015  # Less than 0.15% apart = ranging
                    regime = "CHOPPY" if is_choppy else "TRENDING"
                    
                    logger.info(f"{symbol}: [{regime}] Price ${price:.2f} | EMA750 ${ema_750:.2f} | Slope {slope:.6f}")
                    
                    # Get current position
                    positions = self.client.list_positions()
                    position = next((p for p in positions if p.symbol == symbol), None)
                    
                    # === STOP LOSS CHECK (2.5% hard stop) ===
                    if position and self.entry_prices[symbol]:
                        stop_loss_price = self.entry_prices[symbol] * 0.975
                        if price < stop_loss_price:
                            logger.warning(f"{symbol} HIT STOP LOSS: Entry ${self.entry_prices[symbol]:.2f} -> Current ${price:.2f}")
                            try:
                                self.client.submit_order(
                                    symbol=symbol,
                                    qty=int(position.qty),
                                    side="sell",
                                    type="market",
                                    time_in_force="day"
                                )
                                self.entry_prices[symbol] = None
                                continue
                            except Exception as e:
                                logger.error(f"{symbol} stop loss order failed: {e}")
                    
                    # === REGIME-BASED TRADING ===
                    if is_choppy:
                        # MEAN REVERSION: Buy dips, sell rallies
                        # BUY: when price drops below EMA (support)
                        if price < ema_750 * 0.98 and not position:
                            position_dollars = capital * 0.025  # 2.5% risk
                            shares = int(position_dollars / price)
                            if shares > 0:
                                logger.info(f"{symbol} [BOUNCE ENTRY]: {shares} shares @ ${price:.2f} (reverting to support)")
                                try:
                                    self.client.submit_order(
                                        symbol=symbol,
                                        qty=shares,
                                        side="buy",
                                        type="market",
                                        time_in_force="day"
                                    )
                                    self.entry_prices[symbol] = price
                                except Exception as e:
                                    logger.error(f"{symbol} buy order failed: {e}")
                        
                        # SELL: when price rallies above EMA (resistance)
                        elif price > ema_750 * 1.02 and position:
                            logger.info(f"{symbol} [BOUNCE EXIT]: {position.qty} shares @ ${price:.2f} (sold into strength)")
                            try:
                                self.client.submit_order(
                                    symbol=symbol,
                                    qty=int(position.qty),
                                    side="sell",
                                    type="market",
                                    time_in_force="day"
                                )
                                self.entry_prices[symbol] = None
                            except Exception as e:
                                logger.error(f"{symbol} sell order failed: {e}")
                    
                    else:
                        # TREND FOLLOWING: Buy strength, sell weakness
                        # ENTRY: Strong uptrend
                        if price > ema_750 * 1.02 and slope > 0.05 and not position:
                            position_dollars = capital * 0.025  # 2.5% risk
                            shares = int(position_dollars / price)
                            if shares > 0:
                                logger.info(f"{symbol} [TREND ENTRY]: {shares} shares @ ${price:.2f} (trend up)")
                                try:
                                    self.client.submit_order(
                                        symbol=symbol,
                                        qty=shares,
                                        side="buy",
                                        type="market",
                                        time_in_force="day"
                                    )
                                    self.entry_prices[symbol] = price
                                except Exception as e:
                                    logger.error(f"{symbol} buy order failed: {e}")
                        
                        # EXIT: Strong downtrend
                        elif price < ema_750 * 0.98 and slope < -0.05 and position:
                            logger.info(f"{symbol} [TREND EXIT]: {position.qty} shares @ ${price:.2f} (trend down)")
                            try:
                                self.client.submit_order(
                                    symbol=symbol,
                                    qty=int(position.qty),
                                    side="sell",
                                    type="market",
                                    time_in_force="day"
                                )
                                self.entry_prices[symbol] = None
                            except Exception as e:
                                logger.error(f"{symbol} sell order failed: {e}")
                    
                    # Hold status
                    if position:
                        logger.info(f"{symbol} HOLD: {position.qty} shares, PnL ${float(position.unrealized_pl):.2f}")
                
                except Exception as e:
                    logger.error(f"{symbol} processing error: {e}", exc_info=True)
            
            logger.info(f"Sleeping 60s (next check at {(datetime.now() + timedelta(seconds=60)).strftime('%H:%M:%S')})...")
            time.sleep(60)

if __name__ == "__main__":
    bot = SimpleBot(["NVDA", "TSLA", "XLK"], mode="paper")
    bot.run()
