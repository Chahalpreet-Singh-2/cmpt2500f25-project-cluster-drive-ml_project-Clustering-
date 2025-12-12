import os
import time

from flask import Flask, request, jsonify

from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import psutil

from src.utils.helpers import get_logger

MODEL_VERSION = "v1"

# Logger and app
logger = get_logger(__name__)
app = Flask(__name__)

# -----------------------------
# Prometheus base metrics
# -----------------------------
metrics = PrometheusMetrics(app)
metrics.info(
    "app_info",
    "ML API Information",
    version="1.0.0",
    app_name="cluster-driver-api",
)

# -----------------------------
# Custom ML metrics
# -----------------------------
prediction_counter = Counter(
    "ml_predictions_total",
    "Total number of predictions made",
    ["model_version", "prediction_result", "status"],
)

prediction_latency = Histogram(
    "ml_prediction_duration_seconds",
    "Time spent processing prediction requests",
    ["model_version"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

memory_usage_gauge = Gauge(
    "app_memory_usage_bytes",
    "Memory usage of the application",
)

cpu_usage_gauge = Gauge(
    "app_cpu_usage_percent",
    "CPU usage percentage",
)

model_loaded_gauge = Gauge(
    "model_loaded",
    "Whether models are loaded",
    ["model_version"],
)

# Track current process
process = psutil.Process(os.getpid())

# Mark model as "loaded"
model_loaded_gauge.labels(model_version=MODEL_VERSION).set(1)


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "Cluster Driver prediction API is running",
            "endpoints": ["/v1/predict (POST)"],
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    """Single prediction endpoint for model v1 with Prometheus instrumentation."""
    start_time = time.time()

    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")

                # Handle missing / invalid input
        if not data:
            prediction_counter.labels(
                model_version=model_version,
                prediction_result='no_input',
                status='error'
            ).inc()
            return jsonify({"error": "No input data provided"}), 400

        required_keys = [
            "FSA_Code", "make", "price", "Region",
            "Most_sold_brand", "Average_mileage", "Average_price",
            "FSA_Latitude", "FSA_Longitude",
            "Region_vehicle_sold", "Region_dealerships",
            "Most_sold_month"
        ]
        if any(k not in data for k in required_keys):
            # This will be counted as an error
            raise ValueError("Missing required feature(s)")


        # TODO: transform data into correct format for your model
        # features = preprocess(data)
        # prediction = model.predict(features)
        prediction = 1  # dummy for now

        logger.info(f"Returning prediction: {prediction}")

        # Record successful prediction
        prediction_counter.labels(
            model_version=MODEL_VERSION,
            prediction_result="prediction",
            status="success",
        ).inc()

        # Record latency
        duration = time.time() - start_time
        prediction_latency.labels(model_version=MODEL_VERSION).observe(duration)

        # Update gauges for current resource usage
        memory_usage_gauge.set(process.memory_info().rss)
        cpu_usage_gauge.set(psutil.cpu_percent(interval=None))

        return jsonify({"prediction": prediction}), 200

    except Exception as exc:
        # Record exceptions as errors in metrics
        prediction_counter.labels(
            model_version=MODEL_VERSION,
            prediction_result="error",
            status="error",
        ).inc()

        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    logger.info("Starting prediction API service on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
