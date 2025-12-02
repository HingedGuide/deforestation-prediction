import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: str | Path = None, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger that writes to console and optionally to a file.

    Args:
        name (str): The name of the logger (usually __name__).
        log_file (str | Path, optional): Path to the log file.
        level: Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Optional)
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger