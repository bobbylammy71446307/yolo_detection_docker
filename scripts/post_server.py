from flask import Flask, request, jsonify

app = Flask(__name__)

# Create a global list in memory to store our detections
# In a real application, use a database like SQLite, PostgreSQL, etc.
detections_history = []

# Define a route to handle POST requests at the /api/detections endpoint
@app.route('/api/detections', methods=['POST'])
def handle_detections():
    try:
        content_type = request.headers.get('Content-Type')

        if content_type == 'application/json':
            data = request.get_json()
            print("Received JSON data:", data)
            
            # 1. STORE THE DATA: Append the new detection to our history list
            detections_history.append(data)
            print(f"Total detections stored: {len(detections_history)}")
            
            return jsonify({"status": "success", "message": "Data received"}), 200

        else:
            return jsonify({"error": "Unsupported Content-Type"}), 415

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Invalid data"}), 400

# 2. CREATE A NEW GET ENDPOINT TO READ THE DATA
@app.route('/api/detections', methods=['GET'])
def get_detections():
    """This endpoint allows other machines to retrieve all stored detections."""
    try:
        # Simply return the entire history list as JSON
        return jsonify({
            "status": "success",
            "count": len(detections_history),
            "detections": detections_history
        }), 200
    except Exception as e:
        return jsonify({"error": "Failed to retrieve data"}), 500

# 3. (Optional) Root endpoint for easy testing
@app.route('/')
def index():
    return "Detection API is running. POST JSON to /api/detections or GET from it."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)