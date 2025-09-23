"""
Data provider system for fetching and managing market data
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import time

logger = logging.getLogger(__name__)

class DataProvider:
    """
    Provides real-time and historical market data
    """
    
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        self.current_prices: Dict[str, float] = {}
        
        # Threading for real-time updates
        self.update_thread = None
        self.running = False
        self.update_interval = 60  # Update every 60 seconds
        
        # Supported data sources
        self.data_sources = ['yfinance', 'alpha_vantage', 'polygon']
        self.current_source = 'yfinance'
        
        logger.info("DataProvider initialized")
    
    def start_real_time_updates(self, symbols: List[str]):
        """Start real-time data updates for given symbols"""
        if self.running:
            logger.warning("Real-time updates already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(
            target=self._real_time_update_loop, 
            args=(symbols,), 
            daemon=True
        )
        self.update_thread.start()
        logger.info(f"Started real-time updates for {len(symbols)} symbols")
    
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("Stopped real-time updates")
    
    def _real_time_update_loop(self, symbols: List[str]):
        """Real-time update loop"""
        while self.running:
            try:
                self.update_data(symbols)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in real-time update loop: {e}")
                time.sleep(self.update_interval)
    
    def get_data(self, symbol: str, periods: int = 100, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            periods: Number of periods to fetch
            interval: Data interval ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{periods}_{interval}"
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                last_update = self.last_update.get(cache_key, datetime.min)
                
                # Use cached data if it's recent (within 5 minutes for daily data)
                if interval == '1d' and (datetime.now() - last_update).total_seconds() < 300:
                    return cached_data
            
            # Fetch new data
            if self.current_source == 'yfinance':
                data = self._fetch_yfinance_data(symbol, periods, interval)
            else:
                logger.warning(f"Data source {self.current_source} not implemented, falling back to yfinance")
                data = self._fetch_yfinance_data(symbol, periods, interval)
            
            if data is not None and not data.empty:
                # Cache the data
                self.data_cache[cache_key] = data
                self.last_update[cache_key] = datetime.now()
                
                # Update current price
                self.current_prices[symbol] = data['close'].iloc[-1]
                
                return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
        
        return None
    
    def _fetch_yfinance_data(self, symbol: str, periods: int, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Determine period string for yfinance
            if interval == '1d':
                period_str = f"{periods}d" if periods <= 365 else "max"
            elif interval == '1h':
                period_str = "60d"  # yfinance limit for hourly data
            else:
                period_str = "7d"  # For minute data
            
            data = ticker.history(period=period_str, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}")
                return None
            
            # Get the last N periods
            data = data.tail(periods)
            
            logger.debug(f"Fetched {len(data)} periods for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")
            return None
    
    def update_data(self, symbols: List[str]):
        """Update data for multiple symbols"""
        for symbol in symbols:
            try:
                # Get latest price data
                data = self.get_data(symbol, 1)
                if data is not None and not data.empty:
                    self.current_prices[symbol] = data['close'].iloc[-1]
            except Exception as e:
                logger.error(f"Error updating data for {symbol}: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        # Try to get from cache first
        if symbol in self.current_prices:
            return self.current_prices[symbol]
        
        # Fetch latest data
        data = self.get_data(symbol, 1)
        if data is not None and not data.empty:
            price = data['close'].iloc[-1]
            self.current_prices[symbol] = price
            return price
        
        return None
    
    def get_multiple_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices
    
    def calculate_technical_indicators(self, symbol: str, periods: int = 50) -> Optional[Dict]:
        """Calculate common technical indicators"""
        data = self.get_data(symbol, periods)
        if data is None or len(data) < 20:
            return None
        
        try:
            indicators = {}
            
            # Moving averages
            indicators['sma_10'] = data['close'].rolling(10).mean().iloc[-1]
            indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None
            
            # Exponential moving averages
            indicators['ema_12'] = data['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = (macd_line - signal_line).iloc[-1]
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Bollinger Bands
            sma_20 = data['close'].rolling(20).mean()
            std_20 = data['close'].rolling(20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            
            # Volume indicators
            indicators['volume_sma_10'] = data['volume'].rolling(10).mean().iloc[-1]
            indicators['volume_ratio'] = data['volume'].iloc[-1] / indicators['volume_sma_10']
            
            # Price metrics
            indicators['current_price'] = data['close'].iloc[-1]
            indicators['price_change'] = data['close'].iloc[-1] - data['close'].iloc[-2]
            indicators['price_change_pct'] = (indicators['price_change'] / data['close'].iloc[-2]) * 100
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return None
    
    def get_market_summary(self, symbols: List[str]) -> Dict:
        """Get market summary for a list of symbols"""
        summary = {
            'timestamp': datetime.now(),
            'symbols_count': len(symbols),
            'symbols': {}
        }
        
        for symbol in symbols:
            try:
                current_price = self.get_current_price(symbol)
                indicators = self.calculate_technical_indicators(symbol)
                
                if current_price and indicators:
                    summary['symbols'][symbol] = {
                        'current_price': current_price,
                        'price_change_pct': indicators.get('price_change_pct', 0),
                        'rsi': indicators.get('rsi', 50),
                        'volume_ratio': indicators.get('volume_ratio', 1),
                        'trend': 'bullish' if indicators.get('sma_10', 0) > indicators.get('sma_20', 0) else 'bearish'
                    }
            except Exception as e:
                logger.error(f"Error in market summary for {symbol}: {e}")
        
        return summary
    
    def clear_cache(self):
        """Clear all cached data"""
        self.data_cache.clear()
        self.last_update.clear()
        self.current_prices.clear()
        logger.info("Data cache cleared")