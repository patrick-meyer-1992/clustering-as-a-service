# logger.py
import logging
import sys


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger with standardized formatting and console output.

    This function initializes a logger with a given name and log level.
    If the logger has not been previously configured, it sets up a stream
    handler that logs to stdout with a consistent timestamped format.

    Parameters:
        name (str): The name of the logger (usually __name__ of the calling module).
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG). Default is INFO.

    Returns:
        logging.Logger: A configured logger instance ready for use.

    Notes:
        - Prevents duplicate handlers by checking if handlers are already present.
        - Forces DEBUG level for both logger and handler regardless of initial `level` (may be adjusted).
        - Disables propagation to avoid duplicate logs from parent loggers.
    """
        
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Log format
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevents Logger-Hirarchie
    logger.propagate = False

    return logger
