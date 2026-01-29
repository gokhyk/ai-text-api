import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Annotated
from datetime import datetime, timezone
import traceback
import json
import uuid


def _new_request_id() -> str:
    return uuid.uuid4().hex

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _text_preview(text: str, limit: int = 100) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated, len={len(text)}]"

def _setup_jsonl_logger(name: str, path: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    #logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger    



def _exception_to_dict(e: Exception) -> dict[str, Any]:
    te = traceback.TracebackException.from_exception(e)
    return {
        "type": type(e).__name__,
        "message": str(e),
        "stack": [
            {
                "file": f.filename,
                "line": f.lineno,
                "func": f.name,
                "code": (f.line or "").strip(),
            }
            for f in te.stack
        ],
    }

def _log_exception_json(logger: logging.Logger, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["traceback"] = traceback.format_exc()
    logger.error(json.dumps(payload, ensure_ascii=False))

def _log_json(logger: logging.Logger, payload: dict[str, Any], level: int = logging.INFO) -> None:
    logger.log(level, json.dumps(payload, ensure_ascii=False))


def setup_logging():
    logging.basicConfig(level=logging.INFO)

    request_handler = RotatingFileHandler(
        "requests.log",
        maxBytes=5_000_000,
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    request_handler.setFormatter(formatter)

    logger = logging.getLogger("request")
    logger.addHandler(request_handler)
    logger.propagate = False
    