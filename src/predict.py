import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.helpers import get_project_root, get_logger
from src.utils.model_utils import KProtoWrapper

log = get_logger()


class ModelPredictor:
    """
    Simple wrapper that loads a saved KProtoWrapper model and exposes predict().
    """

    def __init__(self, model_path: str | Path):
        # Use the project root so tests can monkeypatch get_project_root()
        root = get_project_root()
        model_path = Path(root) / model_path
        self.model = KProtoWrapper.load(model_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Delegate to the underlying model (tests assert this behaviour)
        return self.model.predict(df)


def load_model(path: str | Path):
    """
    Helper used by tests: load a joblib model from disk.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


def predict(model, X):
    """
    Thin helper so tests (and other code) can call predict(model, X).

    - Accepts pandas DataFrame/Series or numpy arrays.
    - Raises ValueError on empty input.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        data = X
        n = len(X)
    else:
        data = np.asarray(X)
        if data.ndim == 0:
            n = 0
        else:
            n = data.shape[0]

    if n == 0:
        raise ValueError("empty input")

    return model.predict(data)


def parse_args() -> argparse.Namespace:
    """
    CLI arguments for `python -m src.predict` (or src/predict.py).
    Tests typically monkeypatch sys.argv to just ["predict"], so defaults matter.
    """
    parser = argparse.ArgumentParser(
        description="Run trained clustering model to produce predictions."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/kprototypes_model.joblib",
        help="Path to the trained model, relative to the project root.",
    )

    parser.add_argument(
        "--in-parquet",
        type=str,
        default="data/processed/final_features.parquet",
        help="Input features parquet, relative to the project root.",
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        default="data/processed/predictions.csv",
        help="Where to write predictions CSV, relative to the project root.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint.

    Behaviour:
      - Uses get_project_root() so tests can monkeypatch it to a temp dir.
      - Loads a parquet from `--in-parquet` (default data/processed/final_features.parquet).
      - Loads a model via ModelPredictor.
      - Writes predictions to `--out-csv` (default data/processed/predictions.csv).
    """
    root = get_project_root()
    args = parse_args()

    in_parquet = Path(root) / args.in_parquet
    model_rel = args.model_path
    out_csv = Path(root) / args.out_csv

    if not in_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_parquet}")

    log.info(f"Loading features from: {in_parquet}")
    df = pd.read_parquet(in_parquet)

    log.info(f"Loading model from: {Path(root) / model_rel}")
    predictor = ModelPredictor(model_rel)
    preds = predictor.predict(df)

    out_df = df.copy()
    out_df["Cluster"] = preds
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    log.info(f"Saved predictions -> {out_csv}")


if __name__ == "__main__":
    main()
