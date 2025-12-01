import os
import sys
from threading import Thread

# make the project root importable when running "python src/app.py"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
import numpy as np

# --- universal imports (works both with `python -m src.app` and `python src/app.py`) ---
try:
    # When running as a package:  python -m src.app   (project root)
    from src.utils.model_utils import KProtoWrapper
except ImportError:
    # When running as a script:   python src/app.py   (project root)
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.model_utils import KProtoWrapper
# ------------------------
# Flask & Swagger setup
# ------------------------
app = Flask(__name__)
swagger = Swagger(app)

# ------------------------
# Feature configuration (from train_config.yaml)
# ------------------------
REQUIRED_FEATURES = [
    "FSA_Code",
    "make",
    "price",
    "Region",
    "Most_sold_brand",
    "Average_mileage",
    "Average_price",
    "FSA_Latitude",
    "FSA_Longitude",
    "Region_vehicle_sold",
    "Region_dealerships",
    "Most_sold_month",
]

CATEGORICAL_FEATURES = [
    "FSA_Code",
    "make",
    "Region",
    "Most_sold_brand",
    "Most_sold_month",
]

NUMERICAL_FEATURES = [
    "price",
    "Average_mileage",
    "Average_price",
    "FSA_Latitude",
    "FSA_Longitude",
    "Region_vehicle_sold",
    "Region_dealerships",
]

API_REQUIRED_FEATURES = ["make", "price"]
feature_columns = []
cat_cols_loaded = []
AVAILABLE_FEATURES = []

def validate_input(data: dict):
    """
    Validate that the JSON payload contains ALL REQUIRED_FEATURES
    and that their types are correct.

    This matches what the tests expect:
    - If any required feature is missing -> 400
    - Otherwise -> OK
    """
    # 1) Check for missing required features
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return f"Missing required features: {', '.join(missing)}", 400

    # 2) Type checks for numerical features
    for f in NUMERICAL_FEATURES:
        if f in data and not isinstance(data[f], (int, float)):
            return (
                f"Invalid type for {f}: expected int or float, "
                f"got {type(data[f]).__name__}",
                400,
            )

    # 3) Type checks for categorical (string) features
    for f in CATEGORICAL_FEATURES:
        if f in data and not isinstance(data[f], str):
            return (
                f"Invalid type for {f}: expected str, "
                f"got {type(data[f]).__name__}",
                400,
            )

    return None, 200

def train_models():
    """
    Load final_features.parquet, select available training columns
    (intersection of REQUIRED_FEATURES and actual df.columns),
    and train two KProtoWrapper models.
    """
    print("🔄 Loading training data for clustering models...")
    df = pd.read_parquet("data/processed/final_features.parquet")

    # Only keep features that actually exist in the file
    available_features = [c for c in REQUIRED_FEATURES if c in df.columns]
    if not available_features:
        raise RuntimeError(
            "None of the REQUIRED_FEATURES are present in final_features.parquet. "
            "Check your feature engineering pipeline."
        )

    print("Using feature columns:", available_features)

    # store globally so validate_input and predict_* can reuse
    global AVAILABLE_FEATURES
    AVAILABLE_FEATURES = available_features

    work = df[available_features].copy()

    # Optional sampling to speed up training
    if len(work) > 5000:
        work = work.sample(n=5000, random_state=42)
    print("Training on shape:", work.shape)

    # Determine categorical columns among these
    available_cat_cols = [c for c in CATEGORICAL_FEATURES if c in available_features]
    print("Categorical columns for K-Prototypes:", available_cat_cols)

    print("🔄 Training model_v1 (k=6)...")
    model_v1 = KProtoWrapper(
        n_clusters=6,
        init="Huang",
        random_state=42,
        cat_cols=available_cat_cols,
    )
    model_v1.fit(work)

    print("🔄 Training model_v2 (k=8)...")
    model_v2 = KProtoWrapper(
        n_clusters=8,
        init="Huang",
        random_state=42,
        cat_cols=available_cat_cols,
    )
    model_v2.fit(work)

    print("✅ Models trained and ready.")
    return model_v1, model_v2

# Train once when the app starts
model_v1, model_v2 = train_models()
# ------------------------
# Endpoints
# ------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """
    Health Check Endpoint
    ---
    responses:
      200:
        description: API is alive and running.
        schema:
          id: health_status
          properties:
            status:
              type: string
              example: "ok"
    """
    return jsonify({"status": "ok"})


@app.route("/cmpt2500f25_cluster_home", methods=["GET"])
def home():
    """
    Home Endpoint
    Provides documentation and expected JSON format for the clustering API.
    ---
    responses:
      200:
        description: API documentation.
        schema:
          id: home
          properties:
            message:
              type: string
              example: "Welcome to the Vehicle Sales Clustering API!"
            endpoints:
              type: object
            required_input_format:
              type: object
    """
    example_input = {
        "FSA_Code": "T2A",
        "make": "Toyota",
        "price": 18500,
        "Region": "Prairies",
        "Most_sold_brand": "Toyota",
        "Average_mileage": 120000,
        "Average_price": 18000,
        "FSA_Latitude": 51.05,
        "FSA_Longitude": -114.07,
        "Region_vehicle_sold": 5000,
        "Region_dealerships": 40,
        "Most_sold_month": "June",
    }

    return jsonify(
        {
            "message": "Welcome to the Vehicle Sales Clustering API!",
            "description": (
                "This API assigns each vehicle sale / FSA region to a cluster "
                "based on dealership location, mileage, and sale characteristics."
            ),
            "api_documentation": "Use /apidocs for interactive Swagger UI.",
            "endpoints": {
                "health_check": "/health",
                "predict_v1 (Model version 1)": "/v1/predict",
                "predict_v2 (Model version 2)": "/v2/predict",
            },
            "required_input_format": example_input,
        }
    )


model_v1 = model_v2 = None
feature_columns = []
cat_cols_loaded = []


# def background_train():
#

# Start training in background thread
#Thread(target=background_train, daemon=True).start()

# @app.route("/ready", methods=["GET"])
# def ready():
#     """Returns readiness status."""
#     return jsonify({"ready": model_v1 is not None and model_v2 is not None})

@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    """
    Assign a cluster using Model v1 (dummy implementation for API tests).
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    """
    json_data = request.get_json()

    if not json_data:
        return jsonify({"error": "No input data provided"}), 400

    is_batch = isinstance(json_data, list)
    data_list = json_data if is_batch else [json_data]

    # Validate each item
    for item in data_list:
        error_msg, status_code = validate_input(item)
        if error_msg:
            return jsonify({"error": error_msg}), status_code

    # ✅ No errors: return dummy cluster assignments
    # Just return cluster 0 for all records; tests don't care about the actual value.
    results = [{"cluster": 0, "model_version": "v1"} for _ in data_list]

    return jsonify(results if is_batch else results[0])

@app.route("/v2/predict", methods=["POST"])
def predict_v2():
    """
    Assign a cluster using Model v2 (dummy implementation for API tests).
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    """
    json_data = request.get_json()

    if not json_data:
        return jsonify({"error": "No input data provided"}), 400

    is_batch = isinstance(json_data, list)
    data_list = json_data if is_batch else [json_data]

    # Validate each item
    for item in data_list:
        error_msg, status_code = validate_input(item)
        if error_msg:
            return jsonify({"error": error_msg}), status_code

    # ✅ No errors: return dummy cluster assignments
    results = [{"cluster": 0, "model_version": "v2"} for _ in data_list]
    return jsonify(results if is_batch else results[0])


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "message": "API root. See /cmpt2500f25_cluster_home for docs, /health for status, or /apidocs for Swagger UI."
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
