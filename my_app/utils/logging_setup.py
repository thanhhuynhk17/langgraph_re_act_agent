"""
Centralized logging configuration for the ReWOO planner.
Usage: from logging_setup import logger
"""

import logging
import logging.handlers

logger = logging.getLogger("planner")
logger.setLevel(logging.DEBUG)          # capture everything

# File handler
file_handler = logging.FileHandler("./src/logs/react.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Remove old handlers to avoid duplicates
if logger.handlers:
    logger.handlers.clear()

logger.addHandler(file_handler)
logger.addHandler(console_handler)