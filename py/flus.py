from flask import Flask, request, jsonify
import time  # Example module to simulate processing time

app = Flask(__name__)

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json.get('data')
    # Simulate running a Python script (replace this with your actual logic)
    time.sleep(2)  # Simulating a delay
    result = f'Processed data: {data}'  # Example response

    return jsonify({'message': result})

if __name__ == '__main__':
    app.run(debug=True)
