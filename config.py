"""
Configuration settings for the autotrading system
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///autotrading.db')
    
    # API keys and secrets
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
    BROKER_API_KEY = os.getenv('BROKER_API_KEY', '')
    BROKER_SECRET_KEY = os.getenv('BROKER_SECRET_KEY', '')
    
    # Trading configuration
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% max position size
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.02'))  # 2% max daily loss
    RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.05'))  # 5% annual risk-free rate
    
    # Server configuration
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Data sources
    DEFAULT_DATA_SOURCE = os.getenv('DEFAULT_DATA_SOURCE', 'yfinance')
    
    # Backtesting
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))  # $100k initial capital
    COMMISSION = float(os.getenv('COMMISSION', '0.001'))  # 0.1% commission