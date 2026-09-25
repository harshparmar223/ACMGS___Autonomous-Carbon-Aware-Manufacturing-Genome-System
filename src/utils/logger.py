"""
Centralized structured logger for ACMGS.
Provides clean ANSI color-coded console logs and structured formatting.
"""

import logging
import sys
from typing import Optional

# ANSI Color codes for Windows & POSIX terminals
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_BOLD = "\033[1m"
COLOR_MAGENTA = "\033[35m"

class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: COLOR_CYAN,
        logging.INFO: COLOR_GREEN,
        logging.WARNING: COLOR_YELLOW,
        logging.ERROR: COLOR_RED,
        logging.CRITICAL: COLOR_RED + COLOR_BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, COLOR_RESET)
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        prefix = f"{COLOR_CYAN}[ACMGS]{COLOR_RESET} {time_str} | {color}{record.levelname:<8}{COLOR_RESET} | {COLOR_MAGENTA}{record.name}{COLOR_RESET} - "
        return f"{prefix}{record.getMessage()}"


def get_logger(name: str = "ACMGS", level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
        log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
        logger.setLevel(log_level)
    return logger
