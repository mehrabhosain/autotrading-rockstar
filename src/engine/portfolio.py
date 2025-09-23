"""
Portfolio management system for tracking positions, P&L, and performance metrics
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Position:
    """Represents a single position in a security"""
    
    def __init__(self, symbol: str, quantity: int, entry_price: float, timestamp: datetime):
        self.symbol = symbol
        self.quantity = quantity  # Positive for long, negative for short
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.current_price = entry_price
        self.realized_pnl = 0.0
        
    def update_price(self, price: float):
        """Update current market price"""
        self.current_price = price
    
    def get_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return (self.current_price - self.entry_price) * self.quantity
    
    def get_market_value(self) -> float:
        """Get current market value of position"""
        return abs(self.quantity) * self.current_price
    
    def close_position(self, exit_price: float, quantity: int = None) -> float:
        """Close position (full or partial) and return realized P&L"""
        if quantity is None:
            quantity = self.quantity
        
        # Ensure we don't close more than we have
        quantity = min(abs(quantity), abs(self.quantity)) * (1 if self.quantity > 0 else -1)
        
        # Calculate realized P&L
        pnl = (exit_price - self.entry_price) * quantity
        self.realized_pnl += pnl
        
        # Update position
        self.quantity -= quantity
        
        return pnl

class Portfolio:
    """Portfolio management system"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # Performance tracking
        self.equity_curve = []
        self.trades_history = []
        self.max_portfolio_value = initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def open_position(self, symbol: str, quantity: int, price: float) -> bool:
        """Open a new position or add to existing position"""
        cost = abs(quantity) * price
        
        # Check if we have enough cash
        if cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol}: need ${cost:.2f}, have ${self.cash:.2f}")
            return False
        
        # Deduct cash
        self.cash -= cost
        
        if symbol in self.positions:
            # Add to existing position (average down/up)
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            total_cost = (existing.quantity * existing.entry_price) + (quantity * price)
            
            if total_quantity != 0:
                new_avg_price = total_cost / total_quantity
                existing.quantity = total_quantity
                existing.entry_price = new_avg_price
            else:
                # Position nets to zero, remove it
                del self.positions[symbol]
        else:
            # Create new position
            self.positions[symbol] = Position(symbol, quantity, price, datetime.now())
        
        logger.info(f"Opened position: {quantity} shares of {symbol} at ${price:.2f}")
        return True
    
    def close_position(self, symbol: str, quantity: int = None, price: float = None) -> bool:
        """Close position (full or partial)"""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if price is None:
            price = position.current_price
        
        if quantity is None:
            quantity = position.quantity
        
        # Calculate proceeds
        proceeds = abs(quantity) * price
        self.cash += proceeds
        
        # Close position and get realized P&L
        realized_pnl = position.close_position(price, quantity)
        
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': position.entry_price,
            'exit_price': price,
            'pnl': realized_pnl,
            'timestamp': datetime.now()
        }
        self.trades_history.append(trade)
        
        # Remove position if fully closed
        if position.quantity == 0:
            del self.positions[symbol]
        
        logger.info(f"Closed position: {quantity} shares of {symbol} at ${price:.2f}, P&L: ${realized_pnl:.2f}")
        return True
    
    def update_position_price(self, symbol: str, price: float):
        """Update market price for a position"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get all positions as dictionary"""
        positions_dict = {}
        for symbol, position in self.positions.items():
            positions_dict[symbol] = {
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.get_market_value(),
                'unrealized_pnl': position.get_unrealized_pnl(),
                'timestamp': position.timestamp
            }
        return positions_dict
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(pos.get_market_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        realized_pnl = sum(trade['pnl'] for trade in self.trades_history)
        unrealized_pnl = sum(pos.get_unrealized_pnl() for pos in self.positions.values())
        return realized_pnl + unrealized_pnl
    
    def get_pnl_percent(self) -> float:
        """Get P&L as percentage of initial capital"""
        return self.get_total_pnl() / self.initial_capital * 100
    
    def update_metrics(self):
        """Update performance metrics"""
        current_value = self.get_total_value()
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'value': current_value,
            'pnl': current_value - self.initial_capital
        })
        
        # Update max drawdown
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_max_drawdown(self) -> float:
        """Get maximum drawdown percentage"""
        return self.max_drawdown
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate daily returns
        values = [point['value'] for point in self.equity_curve]
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        excess_return = np.mean(returns) * 252 - risk_free_rate  # 252 trading days
        volatility = np.std(returns) * np.sqrt(252)
        
        return excess_return / volatility if volatility > 0 else 0.0
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        total_value = self.get_total_value()
        total_pnl = self.get_total_pnl()
        
        winning_trades = [t for t in self.trades_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trades_history if t['pnl'] < 0]
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': total_value,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_return_pct': (total_value - self.initial_capital) / self.initial_capital * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'sharpe_ratio': self.get_sharpe_ratio(),
            'total_trades': len(self.trades_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / max(len(self.trades_history), 1) * 100,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        }