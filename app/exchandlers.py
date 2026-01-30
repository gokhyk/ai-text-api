# app/exchandlers.py
import logging
import openai
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import fastapihelpers

def register_exception_handlers(app) -> None:
    @app.exception_handler(openai.OpenAIError)
    async def openai_error_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        error_logger = request.app.state.error_logger

        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "error.openai",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })

        return JSONResponse(
            status_code=502,
            content={"detail": f"Upstream model request failed: {str(exc)}", "request_id": request_id},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", None)
        error_logger = request.app.state.error_logger

        fastapihelpers._log_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "http.exception",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
            "detail": exc.detail,
        }, level=logging.ERROR)

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        error_logger = request.app.state.error_logger

        fastapihelpers._log_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "error.unhandled",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, level=logging.ERROR)

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )




# from fastapi import Request
# from fastapi.responses import JSONResponse
# import openai
# import fastapihelpers

# from core.fastapi_client import app

# @app.exception_handler(openai.OpenAIError)
# async def openai_error_handler(request: Request, exc: Exception):
#     request_id = getattr(request.state, "request_id", None)
#     print("openai_error_handler")
#     fastapihelpers._log_exception_json(fastapihelpers.error_logger, {
#         "ts": fastapihelpers._now_iso(),
#         "event": "error.openai",
#         "request_id": request_id,
#         "error_type": type(exc).__name__,
#         "error": str(exc),
#         "method": request.method,
#         "path": request.url.path,
#     })

#     return JSONResponse(
#         status_code=502,
#         content={"detail": "Upstream model request failed", "request_id": request_id},
#     )


# @app.exception_handler(Exception)
# async def unhandled_error_handler(request: Request, exc: Exception):
#     request_id = getattr(request.state, "request_id", None)

#     fastapihelpers._log_exception_json(fastapihelpers.error_logger, {
#         "ts": fastapihelpers._now_iso(),
#         "event": "error.unhandled",
#         "request_id": request_id,
#         "error_type": type(exc).__name__,
#         "error": str(exc),
#         "method": request.method,
#         "path": request.url.path,
#     })

#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error", "request_id": request_id},
#     )

# from fastapi import HTTPException

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     request_id = getattr(request.state, "request_id", None)

#     fastapihelpers._log_json(fastapihelpers.error_logger, {
#         "ts": fastapihelpers._now_iso(),
#         "event": "http.exception",
#         "request_id": request_id,
#         "path": request.url.path,
#         "method": request.method,
#         "status_code": exc.status_code,
#         "detail": exc.detail,
#     }, level=logging.ERROR)

#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"detail": exc.detail, "request_id": request_id},
#     )
