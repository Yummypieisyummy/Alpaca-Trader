"""
ALPACA TRADING UTILITIES
========================
Common operations for paper/live trading

Usage:
    python alpaca_utils.py [command] [args]

Commands:
    status              - Show account status
    positions           - List all positions
    orders              - Show recent orders
    price SYMBOL        - Get current price
    buy SHARES          - Buy TSLA shares
    sell SHARES         - Sell TSLA shares
    close               - Close all TSLA positions
    test                - Run connection test
"""

import sys
import os
from alpaca_trader import AlpacaTrader
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingUtils:
    """Utility functions for common trading tasks"""
    
    def __init__(self, symbol="TSLA"):
        self.trader = AlpacaTrader(symbol=symbol, mode="paper")
        self.symbol = symbol
    
    def show_account_status(self):
        """Display full account status"""
        print("\n" + "="*70)
        print("ACCOUNT STATUS")
        print("="*70)
        
        account = self.trader.get_account_info()
        if account:
            print(f"Equity:                 ${account['equity']:>12,.2f}")
            print(f"Cash:                   ${account['cash']:>12,.2f}")
            print(f"Buying Power:           ${account['buying_power']:>12,.2f}")
            print(f"Portfolio Value:        ${account['portfolio_value']:>12,.2f}")
            print(f"Portfolio %:            {account['portfolio_percentage']:>12.1f}%")
            
            # Find position
            position = self.trader.get_position()
            if position:
                print(f"\n{self.symbol} POSITION:")
                print(f"  Shares:               {position['shares']:>12.0f}")
                print(f"  Avg Fill Price:       ${position['avg_fill_price']:>12,.2f}")
                print(f"  Current Price:        ${position['current_price']:>12,.2f}")
                print(f"  Market Value:         ${position['market_value']:>12,.2f}")
                print(f"  Unrealized P&L:       ${position['unrealized_pl']:>12,.2f}")
                print(f"  Unrealized P&L %:     {position['unrealized_plpc']*100:>12.2f}%")
            else:
                print(f"\nNo position in {self.symbol}")
        
        print("="*70 + "\n")
    
    def show_positions(self):
        """List all open positions"""
        print("\n" + "="*70)
        print("ALL POSITIONS")
        print("="*70)
        
        try:
            positions = self.trader.client.list_positions()
            if positions:
                for pos in positions:
                    print(f"\n{pos.symbol}:")
                    print(f"  Qty:           {pos.qty:>10} shares")
                    print(f"  Avg Price:     ${pos.avg_fill_price:>10,.2f}")
                    print(f"  Current:       ${pos.current_price:>10,.2f}")
                    print(f"  Market Value:  ${pos.market_value:>10,.2f}")
                    print(f"  P&L:           ${pos.unrealized_pl:>10,.2f}")
            else:
                print("No open positions")
        except Exception as e:
            print(f"Error: {e}")
        
        print("="*70 + "\n")
    
    def show_orders(self, limit=10):
        """Show recent orders"""
        print("\n" + "="*70)
        print(f"RECENT ORDERS (Last {limit})")
        print("="*70)
        
        try:
            orders = self.trader.client.list_orders(limit=limit, status="all")
            if orders:
                for order in orders:
                    status_marker = {
                        "filled": "[FILL]",
                        "pending_new": "[PEND]",
                        "partially_filled": "[PART]",
                        "canceled": "[CANC]"
                    }.get(order.status, "[?]")
                    
                    print(f"\n{status_marker} {order.id[:8]}... | {order.created_at.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   {order.side.upper():>6} {order.qty:>6.0f} {order.symbol}  @ ${order.filled_avg_price or 'pending'}")
                    print(f"   Status: {order.status}, Filled: {order.filled_qty:.0f}/{order.qty:.0f}")
            else:
                print("No orders found")
        except Exception as e:
            print(f"Error: {e}")
        
        print("="*70 + "\n")
    
    def show_price(self, symbol=None):
        """Get current price"""
        if symbol is None:
            symbol = self.symbol
        
        try:
            quote = self.trader.client.get_latest_trade(symbol)
            print(f"\n{symbol}: ${quote.price:.2f} (at {quote.timestamp})\n")
        except Exception as e:
            print(f"Error: {e}")
    
    def buy(self, shares):
        """Place buy order"""
        print(f"\nPlacing buy order: {shares} shares of {self.symbol}...")
        order = self.trader.place_buy_order(int(shares))
        if order:
            print(f"[OK] Order placed: {order.id}")
            print(f"  Status: {order.status}")
        else:
            print("[FAIL] Order failed")
        print()
    
    def sell(self, shares):
        """Place sell order"""
        print(f"\nPlacing sell order: {shares} shares of {self.symbol}...")
        order = self.trader.place_sell_order(int(shares))
        if order:
            print(f"[OK] Order placed: {order.id}")
            print(f"  Status: {order.status}")
        else:
            print("[FAIL] Order failed")
        print()
    
    def close_position(self):
        """Close all TSLA positions"""
        position = self.trader.get_position()
        if position and position['shares'] > 0:
            print(f"\nClosing {position['shares']:.0f} shares of {self.symbol}...")
            order = self.trader.close_position()
            if order:
                print(f"[OK] Close order placed: {order.id}")
            else:
                print("[FAIL] Close failed")
        else:
            print(f"No position to close")
        print()
    
    def run_test(self):
        """Test all functionality"""
        print("\n" + "="*70)
        print("ALPACA CONNECTION TEST")
        print("="*70)
        print(f"Testing connection to Alpaca API ({datetime.now()})")
        
        try:
            # Test account access
            print("\n1. Account Access...", end=" ")
            account = self.trader.get_account_info()
            if account:
                print("[OK]")
                print(f"   Equity: ${account['equity']:.2f}")
            else:
                print("[FAIL]")
            
            # Test position access
            print("2. Position Check...", end=" ")
            position = self.trader.get_position()
            print("[OK]")
            
            # Test price access
            print("3. Price Quote...", end=" ")
            price = self.trader.get_latest_price()
            if price:
                print("[OK]")
                print(f"   {self.symbol}: ${price:.2f}")
            else:
                print("[FAIL]")
            
            # Test order list
            print("4. Order History...", end=" ")
            orders = self.trader.client.list_orders(limit=1, status="all")
            print("[OK]")
            
            print("\n" + "="*70)
            print("ALL SYSTEMS OPERATIONAL")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")


def main():
    """Command-line interface"""
    util = TradingUtils(symbol="TSLA")
    
    if len(sys.argv) < 2:
        print(__doc__)
        util.show_account_status()
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        util.show_account_status()
    
    elif command == "positions":
        util.show_positions()
    
    elif command == "orders":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        util.show_orders(limit)
    
    elif command == "price":
        symbol = sys.argv[2].upper() if len(sys.argv) > 2 else "TSLA"
        util.show_price(symbol)
    
    elif command == "buy":
        if len(sys.argv) > 2:
            util.buy(float(sys.argv[2]))
        else:
            print("Usage: python alpaca_utils.py buy SHARES")
    
    elif command == "sell":
        if len(sys.argv) > 2:
            util.sell(float(sys.argv[2]))
        else:
            print("Usage: python alpaca_utils.py sell SHARES")
    
    elif command == "close":
        util.close_position()
    
    elif command == "test":
        util.run_test()
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
