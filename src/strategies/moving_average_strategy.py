"""
Moving Average Crossover Strategy - a simple but effective trend-following strategy
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MovingAverageStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy
    
    Generates buy signals when fast MA crosses above slow MA
    Generates sell signals when fast MA crosses below slow MA
    """
    
    def __init__(self, symbols: List[str], fast_period: int = 10, slow_period: int = 20, **kwargs):
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
        parameters.update(kwargs)
        
        super().__init__("MovingAverage", symbols, parameters)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Track previous crossover states
        self.previous_crossover = {}
    
    def generate_signals(self) -> List[Dict]:
        """Generate trading signals for all symbols"""
        signals = []
        
        for symbol in self.symbols:
            # Check for entry signals
            entry_signal = self.should_enter(symbol, None)
            if entry_signal:
                signals.append(entry_signal)
            
            # Check for exit signals
            exit_signal = self.should_exit(symbol, None)
            if exit_signal:
                signals.append(exit_signal)
        
        return signals
    
    def should_enter(self, symbol: str, data: pd.DataFrame = None) -> Optional[Dict]:
        """Check if we should enter a position"""
        # Skip if we already have a position
        current_position = self.get_current_position(symbol)
        if current_position and current_position['quantity'] != 0:
            return None
        
        # Get market data
        if data is None:
            data = self.get_market_data(symbol, max(self.slow_period + 10, 50))
        
        if data is None or len(data) < self.slow_period + 1:
            return None
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        # Check for crossover
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        # Bullish crossover: fast MA crosses above slow MA
        if (prev_fast <= prev_slow and current_fast > current_slow):
            quantity = self.calculate_position_size(symbol, 0.8)  # 80% confidence
            
            if quantity > 0:
                self.previous_crossover[symbol] = 'bullish'
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'confidence': 0.8,
                    'reason': f'MA crossover: {self.fast_period}MA ({current_fast:.2f}) > {self.slow_period}MA ({current_slow:.2f})'
                }
        
        # Bearish crossover: fast MA crosses below slow MA (for short selling if enabled)
        elif (prev_fast >= prev_slow and current_fast < current_slow):
            # For now, we'll just track this but not short sell
            self.previous_crossover[symbol] = 'bearish'
        
        return None
    
    def should_exit(self, symbol: str, data: pd.DataFrame = None) -> Optional[Dict]:
        """Check if we should exit a position"""
        current_position = self.get_current_position(symbol)
        if not current_position or current_position['quantity'] == 0:
            return None
        
        # Get market data
        if data is None:
            data = self.get_market_data(symbol, max(self.slow_period + 10, 50))
        
        if data is None or len(data) < self.slow_period + 1:
            return None
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        position_quantity = current_position['quantity']
        
        # Exit long position on bearish crossover
        if position_quantity > 0 and (prev_fast >= prev_slow and current_fast < current_slow):
            return {
                'symbol': symbol,
                'action': 'sell',
                'quantity': abs(position_quantity),
                'confidence': 0.9,
                'reason': f'MA crossover exit: {self.fast_period}MA ({current_fast:.2f}) < {self.slow_period}MA ({current_slow:.2f})'
            }
        
        # Stop loss: exit if position is down more than 5%
        unrealized_pnl_pct = (current_position['unrealized_pnl'] / 
                             (abs(position_quantity) * current_position['entry_price'])) * 100
        
        if unrealized_pnl_pct < -5.0:  # 5% stop loss
            return {
                'symbol': symbol,
                'action': 'sell',
                'quantity': abs(position_quantity),
                'confidence': 1.0,
                'reason': f'Stop loss triggered: {unrealized_pnl_pct:.1f}% loss'
            }
        
        return None
    
    def get_technical_indicators(self, symbol: str) -> Optional[Dict]:
        """Get current technical indicators for a symbol"""
        data = self.get_market_data(symbol, max(self.slow_period + 10, 50))
        
        if data is None or len(data) < self.slow_period:
            return None
        
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        return {
            'fast_ma': fast_ma.iloc[-1],
            'slow_ma': slow_ma.iloc[-1],
            'current_price': data['close'].iloc[-1],
            'crossover_state': self.previous_crossover.get(symbol, 'neutral'),
            'trend': 'bullish' if fast_ma.iloc[-1] > slow_ma.iloc[-1] else 'bearish'
        }