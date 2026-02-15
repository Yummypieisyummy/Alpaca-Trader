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

    def calculate_slope(self, series, periods=20):
        """Calculate linear regression slope"""
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
                    
                    # Calculate 750 EMA and slope
                    ema_750 = bars['close'].ewm(span=750).mean().iloc[-1]
                    slope = self.calculate_slope(bars['close'], periods=20)
                    
                    logger.info(f"{symbol}: Price ${price:.2f} | EMA750 ${ema_750:.2f} | Slope {slope:.6f}")
                    
                    # Get current position
                    positions = self.client.list_positions()
                    position = next((p for p in positions if p.symbol == symbol), None)
                    
                    # === ENTRY ===
                    if price > ema_750 * 1.005 and slope > 0.005 and not position:
                        position_dollars = capital * 0.025  # 2.5% risk
                        shares = int(position_dollars / price)
                        if shares > 0:
                            logger.info(f"{symbol} ENTRY: {shares} shares @ ${price:.2f}")
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
                    
                    # === EXIT ===
                    elif price < ema_750 * 0.995 and slope < -0.005 and position:
                        logger.info(f"{symbol} EXIT: {position.qty} shares @ ${price:.2f}")
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
                    
                    elif position:
                        logger.info(f"{symbol} HOLD: {position.qty} shares, PnL ${position.unrealized_pl:.2f}")
                
                except Exception as e:
                    logger.error(f"{symbol} processing error: {e}", exc_info=True)
            
            logger.info(f"Sleeping 300s (next check at {(datetime.now() + timedelta(seconds=300)).strftime('%H:%M:%S')})...")
            time.sleep(300)

if __name__ == "__main__":
    bot = SimpleBot(["NVDA", "TSLA", "XLK"], mode="paper")
    bot.run()
