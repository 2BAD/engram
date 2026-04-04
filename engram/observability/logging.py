"""Structured JSON logging for non-TTY environments."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON Lines formatter for structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as a JSON line."""
        log_data: dict[str, Any] = {
            'timestamp': datetime.now(UTC).isoformat(),
            'level': record.levelname,
            'event': getattr(record, 'event', record.msg),
        }
        data = getattr(record, 'data', None)
        if data and isinstance(data, dict):
            log_data.update(data)
        return json.dumps(log_data, default=str)


def configure_logging(json_format: bool = False) -> logging.Logger:
    """Configure the engram logger for TTY or JSON output."""
    logger = logging.getLogger('engram')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(handler)
    return logger


def log_event(event: str, level: str = 'INFO', **data: Any) -> None:
    """Emit a structured log event."""
    logger = logging.getLogger('engram')
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(event, extra={'event': event, 'data': data})
