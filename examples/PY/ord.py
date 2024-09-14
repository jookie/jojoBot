import alpaca_trade_api as tradeapi

api = tradeapi.REST('APCA-API-KEY-ID', 'APCA-API-SECRET-KEY', base_url='https://paper-api.alpaca.markets')

def place_order(action):
    if action == "buy":
        api.submit_order(symbol='AAPL', qty=1, side='buy', type='market', time_in_force='gtc')
    elif action == "sell":
        api.submit_order(symbol='AAPL', qty=1, side='sell', type='market', time_in_force='gtc')
    elif action == "hold":
        print("Hold action. No trade executed.")

def calculate_action():
    # Simplified hedge ratio logic
    hedge_ratio = 0.5  # Placeholder logic
    if hedge_ratio > 0.6:
        return "buy"
    elif hedge_ratio < 0.4:
        return "sell"
    else:
        return "hold"

action = calculate_action()
place_order(action)
