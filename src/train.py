import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# MLflow is optional: in production it should be installed,
# but tests shouldn't crash if it's missing.
try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # pragma: no cover
    mlflow = None

import src.utils.helpers as helpers
from src.utils.helpers import get_logger, load_config
from src.utils.model_utils import KProtoWrapper

log = get_logger()


def get_project_root() -> Path:
    """
    Wrapper so tests can monkeypatch src.train.get_project_root.

    In normal use, this just delegates to src.utils.helpers.get_project_root().
    """
    return helpers.get_project_root()

def setup_mlflow():
    """
    - In Docker (MLFLOW_TRACKING_URI set), use the MLflow server, e.g. http://mlflow:5001
    - Locally (no env var), use a file-based store under mlruns_lab2
    """
    if mlflow is None:
        return None

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        # Docker will pass MLFLOW_TRACKING_URI=http://mlflow:5001
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Local dev: use project folder
        mlruns_dir = helpers.resolve_under_root("mlruns_lab2")
        mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    mlflow.set_experiment("Clustering")
    return mlflow


# # Use env variable if set (from docker-compose), otherwise fallback
# mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
# mlflow.set_tracking_uri(mlflow_tracking_uri)

# mlflow.set_experiment("my-ml-project")  # change name if you want

# def train_model():
#     with mlflow.start_run():
#         # log parameters
#         mlflow.log_param("model_type", "RandomForest")

#         # ... your training code here ...
#         # model = ...
#         # accuracy = ...

#         # log metrics
#         # mlflow.log_metric("accuracy", accuracy)

#         # log model
#         # mlflow.sklearn.log_model(model, artifact_path="model")

#         print("Training complete")



def resolve_under_root(path: str | Path) -> Path:
    """
    Wrapper so tests can monkeypatch src.train.resolve_under_root.

    Delegates to src.utils.helpers.resolve_under_root() by default.
    """
    return helpers.resolve_under_root(path)



def load_data(path: str | Path) -> np.ndarray:
    """
    Generic loader that returns a NumPy array from parquet/csv/npy.
    Used by tests in tests/test_train.py.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p).to_numpy()
    elif p.suffix == ".csv":
        return pd.read_csv(p).to_numpy()
    elif p.suffix == ".npy":
        return np.load(p)
    else:
        raise ValueError(f"Unsupported data format: {p.suffix}")


def with_overrides(path: str | Path, overrides: dict | None = None) -> dict:
    """
    Load a YAML config and optionally override specific keys.

    overrides should look like:
        {"train": {"n_clusters": 5}}
    """
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for section, values in overrides.items():
            cfg.setdefault(section, {}).update(values)

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train K-Prototypes clustering model on processed features."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the training config YAML "
             "(default: configs/train_config.yaml)",
    )

    parser.add_argument(
        "-k",
        "--n-clusters",
        type=int,
        dest="k",
        help="Override the number of clusters (n_clusters) from config.",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="If set, save the trained model to the models/ directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    root = get_project_root()
    # --------- config ---------
    # Use helpers.resolve_under_root so tests can monkeypatch project root.
    config_path = resolve_under_root(args.config)
    overrides = {"train": {"n_clusters": args.k}} if args.k else None
    cfg = with_overrides(config_path, overrides)
    train_cfg = cfg.get("train", {})

    # --------- input data path ---------
    in_parquet = train_cfg.get(
        "in_parquet",
        "data/processed/final_features.parquet",
    )
    in_parquet = resolve_under_root(in_parquet)

    log.info(f"Loading processed features from: {in_parquet}")
    df = pd.read_parquet(in_parquet)

    # --------- feature / categorical columns ----------
    feature_cols = train_cfg.get("feature_columns", [])
    keep = [c for c in feature_cols if c in df.columns]

    if not keep:
        raise ValueError(
            "No valid feature columns found in dataframe. "
            "Check 'feature_columns' in your config."
        )

    work = df[keep].copy()

    # OPTIONAL: speed up local experiments by sampling
    sample_frac = train_cfg.get("sample_frac")
    if sample_frac is not None:
        # work = work.sample(frac=sample_frac, random_state=random_state)
        log.info(f"Subsampled data to {len(work)} rows (sample_frac={sample_frac}).")


    cat_cols_cfg = train_cfg.get("categorical_columns", [])
    cat_cols = [c for c in cat_cols_cfg if c in work.columns]

    log.info(f"Using {len(keep)} feature columns.")
    log.info(f"Categorical columns: {cat_cols}")

    # --------- K-Prototypes hyperparameters ----------
    n_clusters = train_cfg.get("n_clusters", 3)
    init = train_cfg.get("init", "Cao")
    random_state = train_cfg.get("random_state", 42)

    # HARD-CODED tiny sample for debug
    # sample_frac = float(train_cfg.get("sample_frac", 0.001))  # 0.1% of rows
    # work = work.sample(frac=sample_frac, random_state=random_state)
    # log.info(f"Subsampled data to {len(work)} rows (sample_frac={sample_frac}).")

    log.info(
        f"Training KProtoWrapper with n_clusters={n_clusters}, "
        f"init={init}, random_state={random_state}"
    )

    # OPTIONAL sampling to speed up training
    sample_frac = train_cfg.get("sample_frac")
    if sample_frac is not None:
       work = work.sample(frac=sample_frac, random_state=random_state)
       log.info(f"Subsampled data to {len(work)} rows (sample_frac={sample_frac}).")

    model = KProtoWrapper(
        n_clusters=n_clusters,
        init=init,
        random_state=random_state,
        cat_cols=cat_cols,
    )

    # --------- optional MLflow logging ----------
    ml_client = setup_mlflow()

    if ml_client is not None:
        with ml_client.start_run(run_name="kprototypes"):
            ml_client.log_param("n_clusters", n_clusters)
            ml_client.log_param("init", init)
            ml_client.log_param("random_state", random_state)
            ml_client.log_param("feature_columns", ",".join(keep))
            ml_client.log_param("categorical_columns", ",".join(cat_cols))

            labels = model.fit(work)

            #  Compute silhouette score
            from sklearn.metrics import silhouette_score
            num_data = work.select_dtypes(include=[np.number])

            if not num_data.empty:
                try:
                    sil = silhouette_score(num_data, labels)
                    ml_client.log_metric("silhouette", sil)
                    log.info(f"Silhouette score logged: {sil:.4f}")
                except Exception as e:
                    log.warning(f"Could not compute silhouette score: {e}")
    else:
        # No MLflow: just train the model
        labels = model.fit(work)

    work["Cluster"] = labels

    # --------- clustered CSV path (CONFIG-DRIVEN) ----------
    out_clustered = train_cfg.get(
        "out_clustered_csv",
        "data/processed/clustered.csv",
    )
    out_clustered = resolve_under_root(out_clustered)
    out_clustered.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(out_clustered, index=False)
    log.info(f"Saved clustered data -> {out_clustered}")

    # --------- model save path (CONFIG-DRIVEN) ----------
    save_model_flag = args.save_model or train_cfg.get("save_model", False)
    
    model_path = None 
    if save_model_flag:
        model_path = train_cfg.get(
            "model_path",
            "models/kprototypes_model.joblib",
        )
        model_path = resolve_under_root(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        log.info(f"Saved model -> {model_path}")
    else:
        log.info(
            "Model not saved (use --save-model flag or set train.save_model: true in config)."
        )
    
    if mlflow is not None and save_model_flag and model_path is not None:
        try:
            tracking_uri = mlflow.get_tracking_uri()
            # Only log artifact if using a local file-based store
            if tracking_uri.startswith("file:"):
                with mlflow.start_run(run_name="kprototypes_model_artifact", nested=True):
                    mlflow.log_artifact(str(model_path), artifact_path="models")
                    log.info(f"Logged model artifact to MLflow: {model_path}")
            else:
                log.warning(
                    f"Skipping MLflow artifact logging for tracking URI '{tracking_uri}' "
                    f"(no permission to write to its artifact root)."
                )
        except Exception as e:
            log.warning(f"Could not log model artifact to MLflow: {e}")


if __name__ == "__main__":
    main()
