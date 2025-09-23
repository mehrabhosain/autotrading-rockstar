"""
Risk management system that enforces trading guardrails and position limits
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management system that enforces various risk controls
    """
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.config = Config()
        
        # Risk limits
        self.max_position_size = self.config.MAX_POSITION_SIZE
        self.max_daily_loss = self.config.MAX_DAILY_LOSS
        self.max_portfolio_drawdown = 0.15  # 15% max drawdown
        
        # Position limits
        self.max_positions = 10  # Maximum number of concurrent positions
        self.max_correlation = 0.7  # Maximum correlation between positions
        
        # Daily tracking
        self.daily_start_value = portfolio.get_total_value()
        self.daily_trades_count = 0
        self.max_daily_trades = 50
        
        # Emergency stop
        self.emergency_stop = False
        
        logger.info("RiskManager initialized with strict guardrails")
    
    def validate_trade(self, symbol: str, action: str, quantity: int) -> bool:
        """
        Validate if a trade is allowed based on risk rules
        
        Args:
            symbol: Trading symbol
            action: Trade action ('buy', 'sell', 'close')
            quantity: Number of shares
            
        Returns:
            True if trade is allowed, False otherwise
        """
        if self.emergency_stop:
            logger.warning("Trade rejected: Emergency stop is active")
            return False
        
        # Check portfolio drawdown
        if not self._check_drawdown_limit():
            logger.warning("Trade rejected: Portfolio drawdown limit exceeded")
            return False
        
        # Check daily loss limit
        if not self._check_daily_loss_limit():
            logger.warning("Trade rejected: Daily loss limit exceeded")
            return False
        
        # Check position size limits
        if action in ['buy'] and not self._check_position_size_limit(symbol, quantity):
            logger.warning("Trade rejected: Position size limit exceeded")
            return False
        
        # Check maximum number of positions
        if action in ['buy'] and not self._check_max_positions():
            logger.warning("Trade rejected: Maximum positions limit exceeded")
            return False
        
        # Check daily trade limit
        if not self._check_daily_trade_limit():
            logger.warning("Trade rejected: Daily trade limit exceeded")
            return False
        
        # Check liquidity (mock implementation)
        if not self._check_liquidity(symbol, quantity):
            logger.warning("Trade rejected: Insufficient liquidity")
            return False
        
        return True
    
    def _check_drawdown_limit(self) -> bool:
        """Check if portfolio drawdown is within limits"""
        current_drawdown = self.portfolio.get_max_drawdown()
        return current_drawdown <= self.max_portfolio_drawdown
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss is within limits"""
        current_value = self.portfolio.get_total_value()
        daily_pnl = (current_value - self.daily_start_value) / self.daily_start_value
        
        return daily_pnl >= -self.max_daily_loss
    
    def _check_position_size_limit(self, symbol: str, quantity: int) -> bool:
        """Check if position size is within limits"""
        current_price = self._get_current_price(symbol)
        if current_price <= 0:
            return False
        
        position_value = quantity * current_price
        portfolio_value = self.portfolio.get_total_value()
        
        position_percentage = position_value / portfolio_value
        
        return position_percentage <= self.max_position_size
    
    def _check_max_positions(self) -> bool:
        """Check if we're under the maximum position limit"""
        current_positions = len(self.portfolio.get_positions())
        return current_positions < self.max_positions
    
    def _check_daily_trade_limit(self) -> bool:
        """Check if we're under the daily trade limit"""
        return self.daily_trades_count < self.max_daily_trades
    
    def _check_liquidity(self, symbol: str, quantity: int) -> bool:
        """Mock liquidity check - in real implementation, this would check order book"""
        # For now, assume all trades are liquid enough
        # In practice, you'd check against average daily volume
        return quantity <= 10000  # Arbitrary limit for demo
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        # In a real implementation, this would fetch from data provider
        # For now, return a mock price
        return 100.0
    
    def check_risk_limits(self):
        """Perform periodic risk checks and take action if needed"""
        # Check for emergency stop conditions
        if self._should_trigger_emergency_stop():
            self.trigger_emergency_stop("Risk limits breached")
        
        # Reset daily counters if new day
        self._reset_daily_counters_if_needed()
    
    def _should_trigger_emergency_stop(self) -> bool:
        """Check if emergency stop should be triggered"""
        # Trigger if portfolio loses more than 20% in a day
        current_value = self.portfolio.get_total_value()
        daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
        
        if daily_loss > 0.20:  # 20% daily loss
            return True
        
        # Trigger if drawdown exceeds 20%
        if self.portfolio.get_max_drawdown() > 0.20:
            return True
        
        return False
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop - halt all trading"""
        self.emergency_stop = True
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # In a real system, you might:
        # 1. Close all positions
        # 2. Cancel all pending orders
        # 3. Send alerts to administrators
        # 4. Log to external monitoring system
    
    def reset_emergency_stop(self):
        """Reset emergency stop (manual intervention required)"""
        self.emergency_stop = False
        logger.info("Emergency stop reset - trading resumed")
    
    def _reset_daily_counters_if_needed(self):
        """Reset daily counters at start of new trading day"""
        # Simple implementation - reset at midnight
        # In practice, you'd use market hours
        current_date = datetime.now().date()
        if not hasattr(self, '_last_reset_date') or self._last_reset_date != current_date:
            self.daily_start_value = self.portfolio.get_total_value()
            self.daily_trades_count = 0
            self._last_reset_date = current_date
            logger.info("Daily risk counters reset")
    
    def record_trade(self, symbol: str, action: str, quantity: int):
        """Record a trade for risk tracking"""
        self.daily_trades_count += 1
        logger.debug(f"Trade recorded: {action} {quantity} {symbol}")
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        current_value = self.portfolio.get_total_value()
        daily_pnl = (current_value - self.daily_start_value) / self.daily_start_value
        
        return {
            'emergency_stop': self.emergency_stop,
            'portfolio_value': current_value,
            'daily_pnl_pct': daily_pnl * 100,
            'max_drawdown_pct': self.portfolio.get_max_drawdown() * 100,
            'current_positions': len(self.portfolio.get_positions()),
            'daily_trades': self.daily_trades_count,
            'risk_limits': {
                'max_position_size_pct': self.max_position_size * 100,
                'max_daily_loss_pct': self.max_daily_loss * 100,
                'max_portfolio_drawdown_pct': self.max_portfolio_drawdown * 100,
                'max_positions': self.max_positions,
                'max_daily_trades': self.max_daily_trades
            }
        }
    
    def calculate_position_risk(self, symbol: str) -> Dict:
        """Calculate risk metrics for a specific position"""
        position = self.portfolio.get_position(symbol)
        if not position:
            return {}
        
        portfolio_value = self.portfolio.get_total_value()
        position_value = abs(position.quantity) * position.current_price
        
        return {
            'symbol': symbol,
            'position_value': position_value,
            'portfolio_percentage': (position_value / portfolio_value) * 100,
            'unrealized_pnl': position.get_unrealized_pnl(),
            'unrealized_pnl_pct': (position.get_unrealized_pnl() / position_value) * 100 if position_value > 0 else 0,
            'days_held': (datetime.now() - position.timestamp).days
        }
    
    def get_portfolio_risk_summary(self) -> Dict:
        """Get comprehensive portfolio risk summary"""
        positions = self.portfolio.get_positions()
        position_risks = [self.calculate_position_risk(symbol) for symbol in positions.keys()]
        
        total_exposure = sum(risk.get('position_value', 0) for risk in position_risks)
        portfolio_value = self.portfolio.get_total_value()
        
        return {
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / portfolio_value if portfolio_value > 0 else 0,
            'number_of_positions': len(positions),
            'largest_position_pct': max([risk.get('portfolio_percentage', 0) for risk in position_risks], default=0),
            'total_unrealized_pnl': sum(risk.get('unrealized_pnl', 0) for risk in position_risks),
            'position_risks': position_risks
        }