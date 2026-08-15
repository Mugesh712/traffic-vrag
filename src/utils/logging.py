"""Shared logging setup for all pipeline stages."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.config import get_settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once: console + a rotating-free file in log_dir."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_level = level or settings.logging.level
    log_dir = settings.resolve_path(settings.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
