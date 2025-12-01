import os
from flask import Flask, request, jsonify

from src.utils.helpers import get_logger

# Create project-wide logger
logger = get_logger(__name__)

app = Flask(__name__)

# Example: Load your trained model here
# from src.utils.model_utils import load_model
# model = load_model("models/model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Cluster Driver prediction API is running",
        "endpoints": ["/v1/predict (POST)"]
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")

        # TODO: transform data into correct format for your model
        # features = preprocess(data)
        # prediction = model.predict(features)
        prediction = 1  # dummy for now

        logger.info(f"Returning prediction: {prediction}")
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting prediction API service on 0.0.0.0:8000")
    app.run(host="0.0.0.0", port=5000)

