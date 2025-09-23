"""
Base strategy class that all trading strategies must inherit from
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, name: str, symbols: List[str], parameters: Optional[Dict] = None):
        self.name = name
        self.symbols = symbols
        self.parameters = parameters or {}
        self.active = True
        self.engine = None
        
        # Strategy state
        self.last_signals = {}
        self.position_sizes = {}
        
        logger.info(f"Strategy {name} initialized with symbols: {symbols}")
    
    def set_engine(self, engine):
        """Set reference to the trading engine"""
        self.engine = engine
    
    @abstractmethod
    def generate_signals(self) -> List[Dict]:
        """
        Generate trading signals based on market data and strategy logic
        
        Returns:
            List of signal dictionaries with keys:
            - symbol: str
            - action: str ('buy', 'sell', 'close')
            - quantity: int
            - confidence: float (0.0 to 1.0)
            - reason: str (optional explanation)
        """
        pass
    
    @abstractmethod
    def should_enter(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Determine if strategy should enter a position
        
        Args:
            symbol: Trading symbol
            data: Historical price data
            
        Returns:
            Signal dictionary if should enter, None otherwise
        """
        pass
    
    @abstractmethod
    def should_exit(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Determine if strategy should exit a position
        
        Args:
            symbol: Trading symbol
            data: Historical price data
            
        Returns:
            Signal dictionary if should exit, None otherwise
        """
        pass
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols this strategy trades"""
        return self.symbols
    
    def is_active(self) -> bool:
        """Check if strategy is active"""
        return self.active
    
    def activate(self):
        """Activate the strategy"""
        self.active = True
        logger.info(f"Strategy {self.name} activated")
    
    def deactivate(self):
        """Deactivate the strategy"""
        self.active = False
        logger.info(f"Strategy {self.name} deactivated")
    
    def get_current_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol from the engine"""
        if self.engine and self.engine.portfolio:
            position = self.engine.portfolio.get_position(symbol)
            if position:
                return {
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.get_unrealized_pnl()
                }
        return None
    
    def get_market_data(self, symbol: str, periods: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for a symbol from the engine"""
        if self.engine and self.engine.data_provider:
            return self.engine.data_provider.get_data(symbol, periods)
        return None
    
    def calculate_position_size(self, symbol: str, signal_strength: float = 1.0) -> int:
        """
        Calculate position size based on portfolio value and risk management
        
        Args:
            symbol: Trading symbol
            signal_strength: Confidence in signal (0.0 to 1.0)
            
        Returns:
            Position size in shares
        """
        if not self.engine or not self.engine.portfolio:
            return 0
        
        # Get current portfolio value
        portfolio_value = self.engine.portfolio.get_total_value()
        
        # Default position sizing: 5% of portfolio per position
        base_allocation = portfolio_value * 0.05
        
        # Adjust based on signal strength
        allocation = base_allocation * signal_strength
        
        # Get current price
        data = self.get_market_data(symbol, 1)
        if data is None or len(data) == 0:
            return 0
        
        current_price = data['close'].iloc[-1]
        
        # Calculate shares
        shares = int(allocation / current_price)
        
        return max(shares, 0)
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy-specific performance metrics"""
        return {
            'name': self.name,
            'active': self.active,
            'symbols': self.symbols,
            'parameters': self.parameters,
            'last_signals': self.last_signals
        }