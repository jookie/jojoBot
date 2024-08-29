# Initialize Alpaca API

API_KEY = "PKEJH4W0URAU56SHKQW3"
API_SECRET = "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow"
API_BASE_URL = 'https://paper-api.alpaca.markets'
# Description: This is a simple python script that prints a greeting message along with the current date and time.
from datetime import datetime
import alpaca_trade_api as tradeapi

def main():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Hello from greet.py at {formatted_time}")
    
    api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')
    positions = api.list_positions()
    print(positions)
     
    
if __name__ == "__main__":
    main()     

# 
# from finrl.config import config

# import finrl.config_tickers as config_tickers


# api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')
# barset = api.get_barset(symbol, 'day', start=start_date, end=end_date)

# from finrl.config_tickers import DOW_30_TICKER
# print(config_tickers.DOW_30_TICKER) 


