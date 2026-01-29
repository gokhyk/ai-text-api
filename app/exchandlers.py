from fastapi import Request
from fastapi.responses import JSONResponse
from openai import OpenAIError
import fastapihelpers

from core.fastapi_client import app

@app.exception_handler(OpenAIError)
async def openai_error_handler(request: Request, exc: OpenAIError):
    request_id = getattr(request.state, "request_id", None)
    print("openai_error_handler")
    fastapihelpers._log_exception_json(error_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "error.openai",
        "request_id": request_id,
        "error_type": "openai",
        "error": str(exc),
        "method": request.method,
        "path": request.url.path,
    })

    return JSONResponse(
        status_code=502,
        content={"detail": "Upstream model request failed", "request_id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)

    fastapihelpers._log_exception_json(error_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "error.unhandled",
        "request_id": request_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "method": request.method,
        "path": request.url.path,
    })

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )
