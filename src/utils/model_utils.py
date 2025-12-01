import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
from kmodes.kprototypes import KPrototypes
# src/utils/model_utils.py
class KProtoWrapper:
    def __init__(self, n_clusters, init="Huang", random_state=42, cat_cols=None):
        self.n_clusters = n_clusters
        self.init = init
        self.random_state = random_state
        self.cat_cols = cat_cols or []
        self.model = None
        self.labels_ = None

        # tests expect this to exist for save/load
        self.params = {
            "n_clusters": n_clusters,
            "init": init,
            "random_state": random_state,
            "cat_cols": self.cat_cols,
        }

    def fit(self, df):
        """
        Fit K-Prototypes on the given DataFrame and return cluster labels.
        """
        kproto = KPrototypes(
            n_clusters=self.n_clusters,
            init=self.init,
            random_state=self.random_state,
        )

        # indices of categorical columns
        cat_idx = [df.columns.get_loc(c) for c in self.cat_cols]
        data = df.to_numpy()

        labels = kproto.fit_predict(data, categorical=cat_idx)

        self.model = kproto
        self.labels_ = np.array(labels)
        return self.labels_  # numpy array, as tests expect

    def predict(self, df):
        """
        Predict clusters for new data.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        cat_idx = [df.columns.get_loc(c) for c in self.cat_cols]
        data = df.to_numpy()
        labels = self.model.predict(data, categorical=cat_idx)
        return np.array(labels)

    def save(self, filepath):
        """
        Save model + metadata to disk.
        """
        joblib.dump(
            {
                "model": self.model,
                "cat_cols": self.cat_cols,
                "params": self.params,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath):
        """
        Load model + metadata from disk and return a KProtoWrapper.
        """
        obj = joblib.load(filepath)
        params = obj["params"]

        wrapper = cls(
            n_clusters=params["n_clusters"],
            init=params["init"],
            random_state=params["random_state"],
            cat_cols=params["cat_cols"],
        )
        wrapper.model = obj["model"]
        return wrapper

    # ---------- helper ----------
    def _cat_indices(self, df: pd.DataFrame):
        cols = list(df.columns)
        return [cols.index(c) for c in self.cat_cols if c in cols]

    
