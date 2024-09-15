from flask import Flask, request, jsonify # Import the Flask class from the flask module and other necessary modules    
import time  # Example module to simulate processing time
from flask_cors import CORS # This is the only new import  # Import the Flask class from the flask module and other necessary modules 

 
app = Flask(__name__) # Create a Flask app
CORS(app) # This will enable CORS for all routes

# Define a route 
@app.route('/your-endpoint')  
def your_function():
    data = {"message": "Hello, world! ferom your-endpoint"} 
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
    return 'Hello, From The Home route World!'

@app.route('/about')
def about():
    return 'About'

# Users/dovpeles/workspace/jojobot/py/greet.py

# from finrl.config_tickers import DOW_30_TICKER
# print(DOW_30_TICKER) 
from datetime import datetime
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"Hello from greet.py at {formatted_time}")

if __name__ == '__main__':
    app.run(debug=True)
