import sys
from loguru import logger

def configure_logger(config):
    logger.remove()
    fmt = "<level>{message}</level>"
    if hasattr(config, 'json_logs') and config.json_logs:
        logger.add(sys.stderr, serialize=True)
    else:
        logger.add(sys.stderr, format=fmt, level="INFO")
