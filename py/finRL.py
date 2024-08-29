import alpaca_trade_api as tradeapi
# from finrl.config import config
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pandas as pd
import gym

# Initialize Alpaca API
ALPACA_API_KEY = "PKVD6WOSPEMKS0UI6A3K"
ALPACA_SECRET_KEY = "BxT64PIQtDBb*tnW"
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

from finrl.config_tickers import DOW_30_TICKER
print(DOW_30_TICKER) 

# Fetch historical data from Alpaca
def fetch_data(symbol, start_date, end_date):
    barset = alpaca.get_barset(symbol, 'day', start=start_date, end=end_date)
    data = barset[symbol]
    df = pd.DataFrame([{
        'timestamp': bar.t.strftime('%Y-%m-%d'),
        'open': bar.o,
        'high': bar.h,
        'low': bar.l,
        'close': bar.c,
        'volume': bar.v
    } for bar in data])
    return df

# Data preprocessing
def preprocess_data(df):
    fe = FeatureEngineer()
    df = fe.preprocess_data(df)
    return df

# Create a trading environment
def create_env(df):
    stock_dim = len(df.tic.unique())
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": stock_dim, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": DOW_30_TICKER, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }
    e_train_gym = StockTradingEnv(df=df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train

# Train model
