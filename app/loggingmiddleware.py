#!/usr/bin/env python3

from fastapi import Request
import time
import uuid
import logging

logger = logging.getLogger("request")

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """
    Global timing + request logging middleware.

    This runs for EVERY HTTP request.
    It wraps the entire request lifecycle:
    - middleware
    - dependencies
    - endpoint
    - response
    """

    request_id = uuid.uuid4().hex
    start = time.perf_counter()

    try:
        # Continue processing the request
        response = await call_next(request)

        return response

    finally:
        # This ALWAYS runs — success or exception
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info({
            "event": "http.request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": getattr(
                locals().get("response", None), "status_code", None
            ),
            "latency_ms": latency_ms,
            "client": request.client.host if request.client else None,
        })
