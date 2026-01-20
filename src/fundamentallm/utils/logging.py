"""Logging utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert log record to JSON."""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "module_data"):
            log_data.update(record.module_data)
        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name."""
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
) -> None:
    """Configure root logging with format and optional file output.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file: Optional path to write logs to file.
        json_format: If True, use JSON structured logging format.
    
    Example:
        >>> setup_logging("DEBUG", log_file=Path("training.log"))
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_handler(logger_inst: logging.Logger, handler_type: type) -> Optional[logging.Handler]:
    """Get a handler of specific type from a logger.
    
    Args:
        logger_inst: Logger instance to search.
        handler_type: Handler class to find (e.g., logging.FileHandler).
    
    Returns:
        First handler of matching type, or None.
    """
    for handler in logger_inst.handlers:
        if isinstance(handler, handler_type):
            return handler
    return None


def log_metrics(logger_inst: logging.Logger, metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics in a structured way.
    
    Args:
        logger_inst: Logger instance.
        metrics: Dictionary of metric names to values.
        step: Optional step/epoch number.
    """
    if not metrics:
        return
    
    parts = []
    if step is not None:
        parts.append(f"Step {step}:")
    
    metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                   for k, v in metrics.items()]
    parts.append(" | ".join(metric_strs))
    
    logger_inst.info(" ".join(parts))
