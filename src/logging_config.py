import logging
import os
from logging.handlers import RotatingFileHandler

def configure_logging():
    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers (important when re-importing)
    logger.handlers = []

    # Console handler (Docker captures this)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(ch)

    # Rotating file handler for logs persisted on volume
    fh = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(fh)

    return logger
