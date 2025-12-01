import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.utils.helpers import get_project_root, get_logger

log = get_logger()


# ---------- New functions used in tests ----------

def calculate_accuracy(y, yhat):
    """
    Simple accuracy: fraction of correct predictions.
    y and yhat can be lists or numpy arrays.
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)

    if y.size == 0:
        raise ValueError("y is empty.")
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same shape.")

    return float(accuracy_score(y, yhat))


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    # Use macro so it works for multiclass, and avoid divide-by-zero warnings
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,  
    }

def evaluate_model(model, X_test, y_test):
    """
    Run model.predict on X_test and compute evaluation metrics.
    Returns a dict with metrics + confusion matrix + classification report.
    """
    if X_test is None or len(X_test) == 0:
        raise ValueError("X_test is empty.")

    yhat = model.predict(X_test)

    metrics = calculate_metrics(y_test, yhat)
    cm = confusion_matrix(y_test, yhat)
    cr = classification_report(y_test, yhat)

    metrics["confusion_matrix"] = cm
    metrics["classification_report"] = cr

    return metrics


# ---------- Your original KMeans / silhouette logic ----------

def main():
    """
    Original clustering evaluation logic: compute silhouette scores for
    numeric features in final_features.parquet.
    """
    root = get_project_root()
    df = pd.read_parquet(root / "data" / "processed" / "final_features.parquet")

    num = df.select_dtypes(include=np.number).fillna(0)
    X = StandardScaler().fit_transform(num)

    ks = range(2, 15)
    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
        sils.append(silhouette_score(X, km.labels_))

    best_k = ks[int(np.argmax(sils))]
    log.info(
        f"Best silhouette (numeric proxy) ~ k={best_k}, score={max(sils):.3f}"
    )


if __name__ == "__main__":
    main()
