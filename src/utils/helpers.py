# src/utils/helpers.py

from __future__ import annotations

from pathlib import Path
import logging
import yaml


def get_project_root() -> Path:
    """
    Return the repository root (parent of src/).

    Example:
        /workspaces/cmpt2500f25-project-cluster-driver-Lab2
    """
    # This assumes this file lives in src/utils/helpers.py
    return Path(__file__).resolve().parents[2]


def resolve_under_root(path: str | Path) -> Path:
    """
    Resolve a path relative to the project root, unless it's already absolute.

    - If `path` is absolute, return it as-is.
    - If `path` is relative, prepend get_project_root().

    This MUST call get_project_root() so that tests can monkeypatch it.
    """
    p = Path(path)
    if p.is_absolute():
        return p

    root = get_project_root()
    return (root / p).resolve()


def load_config(path: str | Path) -> dict:
    """
    Load a YAML config file and return it as a dict.

    Accepts either a string path or Path object.
    Uses resolve_under_root so relative paths are interpreted
    relative to the project root.
    """
    config_path = resolve_under_root(path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # In case the YAML file is empty, return an empty dict instead of None
    return cfg or {}


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a module-level logger.

    If no name is provided, use a default project-wide logger name.
    Ensures a handler is attached only once.
    """
    logger_name = name or "cluster_driver"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger
