import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from stable_baselines3 import PPO  # Replace PPO with your algorithm if different
import os
class AlpacaPaperTrading:
    def __init__(self, ticker_list, time_interval, drl_lib, agent, cwd, net_dim, 
                 state_dim, action_dim, API_KEY, API_SECRET, 
                 API_BASE_URL, tech_indicator_list, turbulence_thresh=30, 
                 max_stock=1e2, latency=None):
        
        self.ticker_list = ticker_list
        self.time_interval = time_interval
        self.drl_lib = drl_lib
        self.agent = agent
        self.cwd = cwd
        self.net_dim = net_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.api_base_url = API_BASE_URL
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock
        self.latency = latency
        self.path = os.path.join(os.getcwd(), "trained_models")

        # Initialize Alpaca API
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.api_base_url, api_version='v2')
        
        # Load the trained model
        self.agent.load(load_dir=self.cwd, net_dim=self.net_dim, state_dim=self.state_dim, action_dim=self.action_dim)
    
    def get_latest_data(self):
        # Retrieve the latest market data
        barset = self.api.get_barset(self.ticker_list, self.time_interval, limit=1)
        data = {ticker: barset[ticker][0].c for ticker in self.ticker_list}
        
        # You may want to calculate technical indicators here if needed
        return pd.DataFrame([data])

    def execute_trade(self, state, actions):
        # Execute the trade based on the model's predicted actions
        for i, ticker in enumerate(self.ticker_list):
            action = actions[i]
            if action > 0:  # Buy signal
                self.api.submit_order(
                    symbol=ticker,
                    qty=abs(action),
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            elif action < 0:  # Sell signal
                self.api.submit_order(
                    symbol=ticker,
                    qty=abs(action),
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
    
    def run(self):
        while True:
            # Get the latest market data
            state = self.get_latest_data()
            
            # Check turbulence condition
            if state['turbulence'] > self.turbulence_thresh:
                print("Turbulence detected, holding positions.")
                time.sleep(60)  # Wait before checking again
                continue
            
            # Get actions from the model
            actions = self.agent.select_action(state)
            
            # Execute the trade
            self.execute_trade(state, actions)
            
            # Sleep for the specified latency
            if self.latency:
                time.sleep(self.latency)

# Example usage
if __name__ == "__main__":
    ticker_list = ["AAPL", "TSLA"]
    time_interval = "1Min"
    drl_lib = "finrl"
    # Instantiate the DRL agent
    # Define the path to your trained model
    # /Users/dovpeles/workspace/jojbot/jojBot/trained_models/PPO_0k_315.zip
    cwd = os.path.join(os.getcwd(),"trained_models") # Path to the trained model directory

    # Define the directory where the model is saved
    
    trained_model_dir = os.path.join(os.getcwd(), "trained_models")  # Replace with the correct directory path
    # print(f"Trained mode path: {trained_model_dir}")
    
    # "/Users/dovpeles/workspace/jojbot/jojBot/trained_models"  # Replace with the correct directory path

    # Define the model file name
    model_file = "PPO_0k_315"  # Replace with the actual filename of the saved model

    # Combine the directory and the file name to form the full path
    model_path = os.path.join(trained_model_dir, model_file)

    # Load the pretrained model from the specified path
    agent = PPO.load(model_path)  # Replace PPO with your algorithm
    print(f"Model loaded from {model_path}")
    # Define the dimensions of the model
    net_dim = 256
    state_dim = 30
    action_dim = len(ticker_list)
    API_KEY = "PKEJH4W0URAU56SHKQW3"
    API_SECRET = "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow"
    API_BASE_URL = 'https://paper-api.alpaca.markets'
    tech_indicator_list = ["macd", "rsi", "cci", "dx"]
    alpaca_trader = AlpacaPaperTrading(
        ticker_list, time_interval, drl_lib, agent, cwd, net_dim, 
        state_dim, action_dim, API_KEY, API_SECRET, 
        API_BASE_URL, tech_indicator_list, turbulence_thresh=30, 
        max_stock=1e2, latency=60
    )
    alpaca_trader.run()