from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Load the trained model and preprocessors
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')

# Flask app setup
app = Flask(__name__)

# Store requests temporarily
active_requests = {}

@app.route("/")
def index():
    return "ESP32 and Flask server are running!"

# Initialize request from ESP32
@app.route("/initialize-request", methods=["POST"])
def initialize_request():
    data = request.json
    request_id = data.get("request_id")
    district = data.get("district")
    building_type = data.get("building_type")
    
    # Store request details
    active_requests[request_id] = {"district": district, "building_type": building_type}
    return jsonify({"status": "success", "message": "Request initialized", "request_id": request_id})

# Render the web form for user input
@app.route("/input/<request_id>")
def user_input(request_id):
    if request_id not in active_requests:
        return "Invalid request ID"
    
    details = active_requests[request_id]
    district = details["district"]
    building_type = details["building_type"]
    city = "Dubai"  # Or whatever logic you need

    
    # Render the HTML form
    return render_template(
        "input_form.html",
        request_id=request_id,
        district=district,
        building_type=building_type,
        city=city
    )

# Handle form submission and make predictions
@app.route("/predict", methods=["POST"])
def predict():
    request_id = request.form.get("request_id")
    bedrooms = int(request.form.get("bedrooms"))
    bathrooms = int(request.form.get("bathrooms"))
    area = float(request.form.get("area"))

    # Retrieve the pre-stored details
    if request_id not in active_requests:
        return "Invalid request ID"

    details = active_requests[request_id]
    district = details["district"]
    building_type = details["building_type"]
    city = "Dubai"  # Always Dubai

    # Prepare the input array for prediction
    x = np.zeros(len(model.feature_names_in_))

    # Assign numerical values
    x[np.where(model.feature_names_in_ == "Area")[0][0]] = area
    x[np.where(model.feature_names_in_ == "Bedrooms")[0][0]] = bedrooms
    x[np.where(model.feature_names_in_ == "Bathrooms")[0][0]] = bathrooms

    # Assign one-hot encoded values
    district_col = f"District_{district}"
    if district_col in model.feature_names_in_:
        x[np.where(model.feature_names_in_ == district_col)[0][0]] = 1

    building_col = f"B_type_{building_type}"
    if building_col in model.feature_names_in_:
        x[np.where(model.feature_names_in_ == building_col)[0][0]] = 1

    # City is not needed if hardcoded as Dubai, but handle it if necessary

    # Scale the numerical features
    x[:3] = scaler.transform([x[:3]])

    # Predict price
    predicted_price = model.predict([x])[0]

    # Display the result
    return render_template("result.html", price=predicted_price)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)