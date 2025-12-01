"""
Pytest configuration for Vehicle Sales Clustering project.
Ensures src/ package is discoverable and provides reusable fixtures.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 🔧 Fix import path so "from src.app import app" works ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# (Optional) keep the old fixtures — they won’t interfere
@pytest.fixture
def sample_data():
    """Small DataFrame for generic tests."""
    data = {
        "FSA_Code": ["T2A", "V5K", "M4B"],
        "make": ["Toyota", "Honda", "Ford"],
        "price": [18500, 16000, 21000],
        "Region": ["Prairies", "Pacific", "Central"],
        "Most_sold_brand": ["Toyota", "Honda", "Ford"],
        "Average_mileage": [120000, 140000, 100000],
        "Average_price": [18000, 16500, 20000],
        "FSA_Latitude": [51.05, 49.28, 43.65],
        "FSA_Longitude": [-114.07, -123.12, -79.38],
        "Region_vehicle_sold": [5000, 6100, 7300],
        "Region_dealerships": [40, 55, 70],
        "Most_sold_month": ["June", "May", "April"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def processed_data():
    np.random.seed(42)
    X_train = np.random.randn(80, 10)
    X_test = np.random.randn(20, 10)
    y_train = np.random.randint(0, 5, 80)
    y_test = np.random.randint(0, 5, 20)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture
def trained_model(processed_data):
    """Simple placeholder ML model (not used for clustering tests)."""
    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model
