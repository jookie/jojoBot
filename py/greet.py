# Initialize Alpaca API

API_KEY = "PKEJH4W0URAU56SHKQW3"
API_SECRET = "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow"
API_BASE_URL = 'https://paper-api.alpaca.markets'
# Description: This is a simple python script that prints a greeting message along with the current date and time.
from datetime import datetime
import sys
import os

# import alpaca_trade_api as tradeapi

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, 'lib', 'alpaca_trade_api')
    print(lib_path)
    sys.path.append(lib_path)
    import lib.alpaca_trade_api as tradeapi
       
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Time from greet.py at {formatted_time}")
    
    api = tradeapi.rest.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')
    positions = api.list_positions()
    print(positions)
     
    
if __name__ == "__main__":
    main()     
