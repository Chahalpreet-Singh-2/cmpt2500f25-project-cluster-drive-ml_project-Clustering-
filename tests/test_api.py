# tests/test_api.py
import json
import pytest

# IMPORTANT: import the Flask app object from your code
from src.app import app

# -------- Fixture: Flask test client (per lab instructions) --------
@pytest.fixture(scope="module")
def client():
    app.testing = True
    with app.test_client() as c:
        yield c


# -------- Basic endpoints --------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body and body.get("status") == "ok"


def test_home_ok(client):
    r = client.get("/cmpt2500f25_cluster_home")
    assert r.status_code == 200
    body = r.get_json()
    # sanity checks for keys you return from home()
    assert "message" in body
    assert "endpoints" in body
    assert "required_input_format" in body


def test_apidocs_ok(client):
    # Flasgger serves Swagger UI here
    r = client.get("/apidocs/")
    # Swagger UI returns HTML; 200 indicates docs are mounted
    assert r.status_code == 200


# -------- Predict v1 (single) --------
def test_predict_v1_single_ok(client):
    payload = {
        # Use the SAME keys your model expects (align with your config/bundle)
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
        "Most_sold_month": "June"
    }
    r = client.post(
        "/v1/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    # Your API returns e.g. {"cluster": int, "model_version": "v1"}
    assert isinstance(body, dict)
    assert body.get("model_version") == "v1"
    assert isinstance(body.get("cluster"), int)


# -------- Predict v2 (batch) --------
def test_predict_v2_batch_ok(client):
    payload = [
        {
            "FSA_Code": "T2A",
            "make": "Toyota",
            "price": 22000,
            "Region": "Prairies",
            "Most_sold_brand": "Toyota",
            "Average_mileage": 100000,
            "Average_price": 21000,
            "FSA_Latitude": 51.05,
            "FSA_Longitude": -114.07,
            "Region_vehicle_sold": 5200,
            "Region_dealerships": 42,
            "Most_sold_month": "May"
        },
        {
            "FSA_Code": "V5K",
            "make": "Honda",
            "price": 16000,
            "Region": "Pacific",
            "Most_sold_brand": "Honda",
            "Average_mileage": 140000,
            "Average_price": 16500,
            "FSA_Latitude": 49.28,
            "FSA_Longitude": -123.12,
            "Region_vehicle_sold": 6100,
            "Region_dealerships": 55,
            "Most_sold_month": "June"
        }
    ]
    r = client.post(
        "/v2/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    # Your API returns a list for batch
    assert isinstance(body, list) and len(body) == 2
    for item in body:
        assert item.get("model_version") == "v2"
        assert isinstance(item.get("cluster"), int)


# -------- Input validation (missing feature) --------
def test_predict_v1_missing_feature_400(client):
    bad_payload = {
        # Intentionally missing "FSA_Code" (or any required column)
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
        "Most_sold_month": "June"
    }
    r = client.post(
        "/v1/predict",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    assert r.status_code == 400
    body = r.get_json()
    assert "error" in body
    # optionally assert the error mentions missing keys
