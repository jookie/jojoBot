from flask import Flask, request, jsonify # Import the Flask class from the flask module and other necessary modules    
import time  # Example module to simulate processing time
 
# from flask_cors import CORS # This is the only new import  # Import the Flask class from the flask module and other necessary modules 
from flask_cors import CORS # This is the only new import  # Import the Flask class from the flask module and other necessary modules 

 
app = Flask(__name__) # Create a Flask app
CORS(app) # This will enable CORS for all routes

# Define a route 
@app.route('/your-endpoint')  
def your_function():
    data = {"message": "Hello, world! from your-endpoint"} 
    return jsonify(data)

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json.get('data')
    # Simulate running a Python script (replace this with your actual logic)
    time.sleep(2)  # Simulating a delay
    result = f'Processed data: {data}'  # Example response

    return jsonify({'message': result})


@app.route('/')
def home():
    from datetime import datetime
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    aa = f"Hello from greet.py at {formatted_time}"
    # from alpaca_trade_api import tradeapi
    # ALPACA_API_KEY = "PKEJH4W0URAU56SHKQW3"
    # ALPACA_API_SECRET = "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow"
    # ALPACA_API_BASE_URL = 'https://paper-api.alpaca.markets'
    # bb = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL, api_version='v2')
    return 'Home ' + aa 
@app.route('/about')
def about():
    return 'About'

if __name__ == '__main__':
    app.run(debug=True)

