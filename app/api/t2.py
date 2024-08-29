# Users/dovpeles/workspace/jojobot/py/greet.py
from datetime import datetime
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"Hello from greet.py at {formatted_time}")
import alpaca_trade_api as tradeapi
ALPACA_API_KEY = "PKEJH4W0URAU56SHKQW3"
ALPACA_API_SECRET = "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow"
ALPACA_API_BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL, api_version='v2')
positions = api.list_positions()
print(positions)

