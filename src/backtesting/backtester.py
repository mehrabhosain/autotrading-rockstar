"""
Backtesting engine for testing strategies on historical data
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..strategies.base_strategy import BaseStrategy
from ..engine.portfolio import Portfolio
from ..data.data_provider import DataProvider

logger = logging.getLogger(__name__)

class BacktestResult:
    """Container for backtest results"""
    
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.initial_capital = 0
        self.final_value = 0
        self.total_return = 0
        self.annualized_return = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.sortino_ratio = 0
        self.calmar_ratio = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.avg_trade = 0
        self.avg_win = 0
        self.avg_loss = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # Time series data
        self.equity_curve = pd.DataFrame()
        self.trades = []
        self.daily_returns = pd.Series()
        
        # Benchmark comparison
        self.benchmark_return = 0
        self.alpha = 0
        self.beta = 0
        self.information_ratio = 0

class Backtester:
    """
    Backtesting engine that simulates strategy performance on historical data
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.data_provider = DataProvider()
        
        logger.info(f"Backtester initialized with ${initial_capital:,.2f} capital")
    
    def run_backtest(self, strategy: BaseStrategy, symbols: List[str], 
                    start_date: str, end_date: str, 
                    benchmark: str = 'SPY') -> BacktestResult:
        """
        Run backtest for a strategy
        
        Args:
            strategy: Trading strategy to test
            symbols: List of symbols to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            benchmark: Benchmark symbol for comparison
        
        Returns:
            BacktestResult object with performance metrics
        """
        logger.info(f"Starting backtest: {strategy.name} from {start_date} to {end_date}")
        
        # Initialize components
        portfolio = Portfolio(self.initial_capital)
        result = BacktestResult()
        result.start_date = pd.to_datetime(start_date)
        result.end_date = pd.to_datetime(end_date)
        result.initial_capital = self.initial_capital
        
        # Get historical data for all symbols
        historical_data = {}
        for symbol in symbols:
            data = self._get_historical_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                historical_data[symbol] = data
            else:
                logger.warning(f"No data available for {symbol}")
        
        if not historical_data:
            logger.error("No historical data available for backtesting")
            return result
        
        # Get benchmark data
        benchmark_data = self._get_historical_data(benchmark, start_date, end_date)
        
        # Create date range for simulation
        all_dates = set()
        for data in historical_data.values():
            all_dates.update(data.index)
        
        simulation_dates = sorted(list(all_dates))
        
        # Run simulation
        equity_curve = []
        trades = []
        
        for current_date in simulation_dates:
            try:
                # Update portfolio with current market prices
                self._update_portfolio_prices(portfolio, historical_data, current_date)
                
                # Generate signals for this date
                signals = self._generate_signals_for_date(
                    strategy, historical_data, current_date, portfolio
                )
                
                # Execute trades
                for signal in signals:
                    trade_result = self._execute_trade(
                        portfolio, signal, historical_data, current_date
                    )
                    if trade_result:
                        trades.append(trade_result)
                
                # Record portfolio value
                portfolio_value = portfolio.get_total_value()
                equity_curve.append({
                    'date': current_date,
                    'value': portfolio_value,
                    'cash': portfolio.cash,
                    'positions_value': portfolio_value - portfolio.cash
                })
                
            except Exception as e:
                logger.error(f"Error processing date {current_date}: {e}")
                continue
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics(
            result, equity_curve, trades, benchmark_data
        )
        
        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        return result
    
    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data for backtesting"""
        try:
            # Calculate number of days to fetch (with buffer)
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            days = (end - start).days + 100  # Add buffer for technical indicators
            
            data = self.data_provider.get_data(symbol, periods=days, interval='1d')
            
            if data is not None:
                # Filter to date range
                mask = (data.index >= start) & (data.index <= end)
                return data[mask]
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
        
        return None
    
    def _update_portfolio_prices(self, portfolio: Portfolio, historical_data: Dict, 
                               current_date: datetime):
        """Update portfolio positions with current market prices"""
        for symbol in portfolio.positions.keys():
            if symbol in historical_data:
                data = historical_data[symbol]
                if current_date in data.index:
                    current_price = data.loc[current_date, 'close']
                    portfolio.update_position_price(symbol, current_price)
    
    def _generate_signals_for_date(self, strategy: BaseStrategy, historical_data: Dict,
                                 current_date: datetime, portfolio: Portfolio) -> List[Dict]:
        """Generate trading signals for a specific date"""
        signals = []
        
        for symbol in strategy.get_symbols():
            if symbol not in historical_data:
                continue
            
            data = historical_data[symbol]
            
            # Get data up to current date
            mask = data.index <= current_date
            historical_slice = data[mask]
            
            if len(historical_slice) < 20:  # Need minimum data for indicators
                continue
            
            # Mock the strategy's data access
            original_get_market_data = strategy.get_market_data
            original_get_current_position = strategy.get_current_position
            
            def mock_get_market_data(sym: str, periods: int = 100):
                if sym == symbol:
                    return historical_slice.tail(periods)
                return None
            
            def mock_get_current_position(sym: str):
                position = portfolio.get_position(sym)
                if position:
                    return {
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'current_price': position.current_price,
                        'unrealized_pnl': position.get_unrealized_pnl()
                    }
                return None
            
            # Replace methods temporarily
            strategy.get_market_data = mock_get_market_data
            strategy.get_current_position = mock_get_current_position
            
            try:
                # Check for entry signals
                entry_signal = strategy.should_enter(symbol, historical_slice)
                if entry_signal:
                    entry_signal['date'] = current_date
                    signals.append(entry_signal)
                
                # Check for exit signals
                exit_signal = strategy.should_exit(symbol, historical_slice)
                if exit_signal:
                    exit_signal['date'] = current_date
                    signals.append(exit_signal)
            
            finally:
                # Restore original methods
                strategy.get_market_data = original_get_market_data
                strategy.get_current_position = original_get_current_position
        
        return signals
    
    def _execute_trade(self, portfolio: Portfolio, signal: Dict, 
                      historical_data: Dict, current_date: datetime) -> Optional[Dict]:
        """Execute a trade based on signal"""
        symbol = signal['symbol']
        action = signal['action']
        quantity = signal['quantity']
        
        if symbol not in historical_data or current_date not in historical_data[symbol].index:
            return None
        
        price = historical_data[symbol].loc[current_date, 'close']
        
        # Apply commission
        commission = abs(quantity) * price * self.commission
        
        trade_result = {
            'date': current_date,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'signal': signal
        }
        
        if action == 'buy':
            success = portfolio.open_position(symbol, quantity, price)
            if success:
                portfolio.cash -= commission  # Deduct commission
                return trade_result
        
        elif action in ['sell', 'close']:
            success = portfolio.close_position(symbol, quantity, price)
            if success:
                portfolio.cash -= commission  # Deduct commission
                return trade_result
        
        return None
    
    def _calculate_performance_metrics(self, result: BacktestResult, 
                                     equity_curve: List[Dict], trades: List[Dict],
                                     benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        if not equity_curve:
            return result
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        result.equity_curve = equity_df
        
        # Basic metrics
        result.final_value = equity_df['value'].iloc[-1]
        result.total_return = (result.final_value - result.initial_capital) / result.initial_capital
        
        # Calculate daily returns
        result.daily_returns = equity_df['value'].pct_change().dropna()
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25
        if years > 0:
            result.annualized_return = (1 + result.total_return) ** (1 / years) - 1
        
        # Drawdown metrics
        rolling_max = equity_df['value'].expanding().max()
        drawdown = (equity_df['value'] - rolling_max) / rolling_max
        result.max_drawdown = abs(drawdown.min())
        
        # Risk metrics
        if len(result.daily_returns) > 1:
            excess_returns = result.daily_returns - 0.05/252  # Assuming 5% risk-free rate
            result.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / result.daily_returns.std()
            
            # Sortino ratio (using downside deviation)
            downside_returns = result.daily_returns[result.daily_returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.sqrt(252) * downside_returns.std()
                result.sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
            
            # Calmar ratio
            if result.max_drawdown > 0:
                result.calmar_ratio = result.annualized_return / result.max_drawdown
        
        # Trade analysis
        result.trades = trades
        result.total_trades = len(trades)
        
        if trades:
            # Calculate P&L for each trade (simplified)
            winning_trades = [t for t in trades if t['action'] == 'sell']  # Simplified
            result.winning_trades = len([t for t in winning_trades if t.get('pnl', 0) > 0])
            result.losing_trades = result.total_trades - result.winning_trades
            result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        
        # Benchmark comparison
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_start = benchmark_data['close'].iloc[0]
            benchmark_end = benchmark_data['close'].iloc[-1]
            result.benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
            result.alpha = result.total_return - result.benchmark_return
        
        return result
    
    def plot_results(self, result: BacktestResult, save_path: str = None) -> str:
        """Create comprehensive performance plots"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Equity Curve', 'Drawdown', 'Daily Returns', 
                          'Monthly Returns Heatmap', 'Trade Distribution', 'Rolling Sharpe'],
            specs=[[{"colspan": 2}, None],
                   [{}, {}],
                   [{}, {}]]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=result.equity_curve.index, y=result.equity_curve['value'],
                      name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Drawdown
        rolling_max = result.equity_curve['value'].expanding().max()
        drawdown = (result.equity_curve['value'] - rolling_max) / rolling_max * 100
        fig.add_trace(
            go.Scatter(x=result.equity_curve.index, y=drawdown,
                      name='Drawdown %', fill='tonexty', line=dict(color='red')),
            row=2, col=1
        )
        
        # Daily returns histogram
        fig.add_trace(
            go.Histogram(x=result.daily_returns * 100, name='Daily Returns %',
                        nbinsx=50),
            row=2, col=2
        )
        
        # Rolling Sharpe ratio
        rolling_sharpe = result.daily_returns.rolling(252).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
        )
        fig.add_trace(
            go.Scatter(x=result.equity_curve.index, y=rolling_sharpe,
                      name='Rolling Sharpe (1Y)', line=dict(color='green')),
            row=3, col=2
        )
        
        fig.update_layout(height=800, title_text="Backtest Performance Analysis")
        
        if save_path:
            fig.write_html(save_path)
            return save_path
        
        return fig.to_html()
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate comprehensive backtest report"""
        report = f"""
BACKTEST PERFORMANCE REPORT
==========================

OVERVIEW
--------
Strategy: {result.start_date} to {result.end_date}
Initial Capital: ${result.initial_capital:,.2f}
Final Value: ${result.final_value:,.2f}
Total Return: {result.total_return:.2%}
Annualized Return: {result.annualized_return:.2%}

RISK METRICS
-----------
Maximum Drawdown: {result.max_drawdown:.2%}
Sharpe Ratio: {result.sharpe_ratio:.2f}
Sortino Ratio: {result.sortino_ratio:.2f}
Calmar Ratio: {result.calmar_ratio:.2f}

TRADING STATISTICS
-----------------
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}
Win Rate: {result.win_rate:.2%}

BENCHMARK COMPARISON
-------------------
Benchmark Return: {result.benchmark_return:.2%}
Alpha: {result.alpha:.2%}
        """
        
        return report