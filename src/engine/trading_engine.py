"""
Core trading engine that manages strategy execution, order management, and risk controls
"""
import threading
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..strategies.base_strategy import BaseStrategy
from ..risk.risk_manager import RiskManager
from ..data.data_provider import DataProvider
from .portfolio import Portfolio
from .order_manager import OrderManager

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Main trading engine that coordinates all trading activities
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager(self.portfolio)
        self.order_manager = OrderManager()
        self.data_provider = DataProvider()
        
        self.strategies: Dict[str, BaseStrategy] = {}
        self.running = False
        self.thread = None
        
        # Performance tracking
        self.start_time = None
        self.trades_count = 0
        self.winning_trades = 0
        
        logger.info(f"TradingEngine initialized with ${initial_capital:,.2f}")
    
    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a trading strategy to the engine"""
        self.strategies[name] = strategy
        strategy.set_engine(self)
        logger.info(f"Added strategy: {name}")
    
    def remove_strategy(self, name: str):
        """Remove a trading strategy from the engine"""
        if name in self.strategies:
            del self.strategies[name]
            logger.info(f"Removed strategy: {name}")
    
    def start(self):
        """Start the trading engine"""
        if self.running:
            logger.warning("Trading engine is already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Trading engine started")
    
    def stop(self):
        """Stop the trading engine"""
        if not self.running:
            logger.warning("Trading engine is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Trading engine stopped")
    
    def _run_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Update market data
                self._update_market_data()
                
                # Execute strategies
                for name, strategy in self.strategies.items():
                    if strategy.is_active():
                        signals = strategy.generate_signals()
                        for signal in signals:
                            self._process_signal(signal, name)
                
                # Process pending orders
                self.order_manager.process_orders()
                
                # Update portfolio metrics
                self.portfolio.update_metrics()
                
                # Risk management checks
                self.risk_manager.check_risk_limits()
                
                # Sleep for next iteration
                time.sleep(1)  # 1 second intervals
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _update_market_data(self):
        """Update market data for all tracked symbols"""
        symbols = set()
        for strategy in self.strategies.values():
            symbols.update(strategy.get_symbols())
        
        if symbols:
            self.data_provider.update_data(list(symbols))
    
    def _process_signal(self, signal: Dict, strategy_name: str):
        """Process a trading signal from a strategy"""
        symbol = signal.get('symbol')
        action = signal.get('action')  # 'buy', 'sell', 'close'
        quantity = signal.get('quantity', 0)
        
        if not all([symbol, action]):
            logger.warning(f"Invalid signal from {strategy_name}: {signal}")
            return
        
        # Risk management check
        if not self.risk_manager.validate_trade(symbol, action, quantity):
            logger.warning(f"Trade rejected by risk manager: {signal}")
            return
        
        # Create and submit order
        order = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'strategy': strategy_name,
            'timestamp': datetime.now()
        }
        
        self.order_manager.submit_order(order)
        logger.info(f"Order submitted: {order}")
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return {
            'total_value': self.portfolio.get_total_value(),
            'cash': self.portfolio.cash,
            'positions': self.portfolio.get_positions(),
            'pnl': self.portfolio.get_total_pnl(),
            'pnl_percent': self.portfolio.get_pnl_percent(),
            'trades_count': self.trades_count,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.trades_count, 1),
            'running_time': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        portfolio_value = self.portfolio.get_total_value()
        returns = (portfolio_value - self.initial_capital) / self.initial_capital
        
        return {
            'total_return': returns,
            'portfolio_value': portfolio_value,
            'max_drawdown': self.portfolio.get_max_drawdown(),
            'sharpe_ratio': self.portfolio.get_sharpe_ratio(),
            'total_trades': self.trades_count,
            'win_rate': self.winning_trades / max(self.trades_count, 1)
        }