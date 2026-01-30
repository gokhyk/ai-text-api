#main.py
from __future__ import annotations

import json
import os
import time
import uuid
import logging
import traceback

from datetime import datetime, timezone
from typing import Any, Annotated
from openai import OpenAIError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field


import fastapihelpers
from callmodeljson import _call_model_json
from fastapihelpers import NonBlankStr



try:
    #pydantic v2
    from pydantic import ConfigDict
except Exception:   #pragma: no cover
    ConfigDict = None #type: ignore[assignment]


try:
    from pydantic import ValidationError
except Exception:
    ValidationError = Exception

from core.openai_client import client
from core.fastapi_client import app

# ----------------------------------
# Logging 
# ----------------------------------
assert hasattr(app.state, "reqres_logger"), "reqres_logger not initialized on app.state"
assert hasattr(app.state, "error_logger"), "error_logger not initialized on app.state"

# ----------------------------
# Schemas
# ----------------------------
NonEmptyStr = Annotated[str, Field(min_length=1)]

class SummarizeOutput(BaseModel):
    """Structured output expected from the model."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")

    summary: NonEmptyStr = Field(description="A concise summary of the text.")
    key_points: list[NonEmptyStr] = Field(
        default_factory=list,
        description="A bulleted list of the most important points.",
    )

    if ConfigDict is None:  # Pydantic v1 fallback
        class Config:
            extra = "forbid"

class AnalyzeOutput(BaseModel):
    """Structured output expected from the model."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")

    sentiment: list[NonEmptyStr] = Field(default_factory=list, description="Sentiment of the text.")
    topics: list[NonEmptyStr] = Field(default_factory=list, description="Topics in the text.")
    confidence: float = Field(ge=0, le=10, description="confidence level of the system on the output")
    

    if ConfigDict is None:  # Pydantic v1 fallback
        class Config:
            extra = "forbid"

class SummarizeRequest(BaseModel):
    text: NonBlankStr

class AnalyzeRequest(BaseModel):
    text: NonBlankStr

@app.post("/summarize", response_model=SummarizeOutput)
def summarize(payload: SummarizeRequest, request: Request) -> SummarizeOutput:

    model="gpt-5-nano"    
    request_id = get_request_id(request)

    reqres_logger = request.app.state.reqres_logger
    error_logger = request.app.state.error_logger

    text = payload.text

    fastapihelpers._log_json(reqres_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "summarize.start",
        "request_id": request_id,
        "text_len": len(text),
        "text_preview": fastapihelpers._text_preview(text),
        "model": model,
    }, logging.INFO)

    system_prompt = "Summarize the following text clearly and concisely and provide key points."

    validated_response = _call_model_json(
            client=client,
            ModelClass=SummarizeOutput, 
            schema_name="summary_schema",
            schema_dict=None,   #SUMMARY_SCHEMA,
            system_prompt=system_prompt, 
            text=text, 
            model=model, 
            error_logger=error_logger
        )


    fastapihelpers._log_json(reqres_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "summarize.success",
        "request_id": request_id,
        "model": model,
        "result" : {
            "summary_len": len(validated_response.summary),
            "key_points_count": len(validated_response.key_points)
        
        }
    }, logging.INFO)

    return SummarizeOutput(summary=validated_response.summary, key_points=validated_response.key_points)

@app.post("/analyze", response_model=AnalyzeOutput)
def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeOutput:
 
    model="gpt-5-nano"    
    request_id = get_request_id(request)

    text = payload.text

    fastapihelpers._log_json(fastapihelpers.reqres_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "analyze.start",
        "request_id": request_id,
        "text_len": len(text),
        "text_preview": fastapihelpers._text_preview(text),
        "model": model,
    }, logging.INFO)

    system_prompt = """Analyze the text.Return JSON with: 
                    - sentiment: array of strings (e.g. [\"positive\", \"cheerful\", \"happy\"])
                    - topics: array of strings
                    -confidence: number from 0 to 10
                    Return only valid JSON
                    """

    validated_response = _call_model_json(
        client=client,
        ModelClass=AnalyzeOutput, 
        schema_name="analyze_schema",
        schema_dict=None,        #ANALYZE_SCHEMA,
        system_prompt=system_prompt, 
        text=text, 
        model=model, 
        error_logger=fastapihelpers.error_logger
    )



    fastapihelpers._log_json(fastapihelpers.reqres_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "analyze.success",
            "request_id": request_id,
            "model": model,
            "result" : {
                "sentiment_count": len(validated_response.sentiment),
                "topics_count": len(validated_response.topics),
                "confidence": validated_response.confidence
            
            }
        },logging.INFO)  

    return AnalyzeOutput(sentiment=validated_response.sentiment, 
                              topics=validated_response.topics, 
                              confidence=validated_response.confidence)

@app.get("/")
def root():
    return {
        "status": "root ok",
        "message": "API is running. Go to /docs to use it."
    }

@app.get("/index", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head><title>main.py</title></head>
        <body>
            <h1>This response is produced by FastAPI</h1>
            <p>FastAPI and uvicorn is working fine</p>
        </body>
    </html>
    """

@app.get("/health")
def health():
    return {
        "status": "health ok",
        "message": "API is running. Go to /docs to use it."        
    }

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/log-test")
def log_test():
    fastapihelpers._log_json(
        fastapihelpers.reqres_logger,
        {"ts": fastapihelpers._now_iso(), "event": "log_test", "ok": True}
    )
    return {"status": "logged"}

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
    reqres_logger = request.app.state.reqres_logger
    error_logger = request.app.state.error_logger

    start = time.perf_counter()
    request.state.request_id = uuid.uuid4().hex

    status_code = None
    try:
        # Continue processing the request
        response: Response = await call_next(request)
        status_code = response.status_code
        return response
    
    except HTTPException as e:
        status_code = e.status_code
        raise  # let your HTTPException handler run (or FastAPI default)

    except Exception:
        status_code = 500
        raise  # let your Exception handler run

    finally:
        # This ALWAYS runs — success or exception
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_json(reqres_logger, {
            "ts": fastapihelpers._now_iso(),
            "called": " by middleware",
            "event": "http.request",
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "client": request.client.host if request.client else None,
        }, logging.INFO)

def get_request_id(request: Request) -> str:
    return request.state.request_id
