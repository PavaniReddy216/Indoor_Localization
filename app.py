from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__, template_folder="templates")  # Ensure 'templates' folder is created

# Load trained model and encoders
model = tf.keras.models.load_model(r"C:\MY PROJECTS\pro\indoor_localization_model.h5")
scaler = joblib.load(r"C:\MY PROJECTS\pro\scaler.pkl")
encoder_building = joblib.load(r"C:\MY PROJECTS\pro\building_encoder.pkl")
encoder_floor = joblib.load(r"C:\MY PROJECTS\pro\floor_encoder.pkl")
encoder_room = joblib.load(r"C:\MY PROJECTS\pro\room_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")  # Serve HTML page
@app.route('/description')
def description():
    return render_template("description.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('rssi_values', [])

        if not isinstance(data, list) or len(data) != 5:
            return jsonify({"error": "Expected exactly 5 RSSI values as a list"}), 400

        X = np.array(data).reshape(1, -1)
        X_expanded = np.tile(X, (1, 104))
        X_scaled = scaler.transform(X_expanded).reshape(1, 1, 520)

        preds = model.predict(X_scaled)
        building_pred = np.argmax(preds[0])
        floor_pred = np.argmax(preds[1])
        room_pred = np.argmax(preds[2])

        building = encoder_building.inverse_transform([building_pred])[0]
        floor = encoder_floor.inverse_transform([floor_pred])[0]
        room = encoder_room.inverse_transform([room_pred])[0]

        return jsonify({
            "Building": int(building),
            "Floor": int(floor),
            "Room": int(room)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
