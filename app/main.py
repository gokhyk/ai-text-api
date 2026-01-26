#main.py
from __future__ import annotations

import json

import os
import time


from datetime import datetime, timezone

from typing import Any, Annotated

from openai import OpenAIError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field

import traceback
import logging

import fastapihelpers
from callmodeljson import _call_model_json


try:
    #pydantic v2
    from pydantic import ConfigDict
except Exception:   #pragma: no cover
    ConfigDict = None #type: ignore[assignment]


try:
    from pydantic import ValidationError
except Exception:
    ValidationError = Exception

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

app = FastAPI()
client = OpenAI()  # <-- CORRECT

# ----------------------------------
# Logging 
# ----------------------------------
log_file = "open_api_log_file.txt"
error_file = "open_api_error_file.txt"

reqres_logger = fastapihelpers._setup_jsonl_logger("openai.reqres", log_file, logging.INFO)
error_logger = fastapihelpers._setup_jsonl_logger("openai.error", error_file, logging.ERROR)



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
    text: NonEmptyStr


class AnalyzeRequest(BaseModel):
    text: NonEmptyStr


SUMMARY_SCHEMA =  {
                "type": "object",
                    "properties": {
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "key_points"],
                "additionalProperties": False,
            }


@app.post("/summarize", response_model=SummarizeOutput)
def summarize(request: SummarizeRequest) -> SummarizeOutput:
    fastapihelpers._log_json(reqres_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "summarize.called",
    })
    start = time.perf_counter()   
    model="gpt-5-nano"    
    request_id = fastapihelpers._new_request_id()

    text = (request.text or "").strip()
    if not text:
        fastapihelpers._log_json(reqres_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": 400,
            "error_type": "validation",
            "error": "Text cannot be empty",
        })
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    fastapihelpers._log_json(reqres_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "summarize.start",
        "request_id": request_id,
        "text_len": len(text),
        "text_preview": fastapihelpers._text_preview(text),
        "model": model,
    })

    system_prompt = "Summarize the following text clearly and concisely and provide key points."
    try:
        validated_response = _call_model_json(client=client,
                           ModelClass=SummarizeOutput, 
                           schema_name="summary_schema",
                           schema_dict=SUMMARY_SCHEMA,
                           system_prompt=system_prompt, 
                           text=text, 
                           model=model, 
                           error_logger=error_logger)

        openai_response_id = getattr(validated_response, "id", None)
        openai_request_id = getattr(validated_response, "_request_id", None)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        fastapihelpers._log_json(reqres_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "summarize.success",
            "request_id": request_id,
            "status_code": 200,
            "latency_ms": latency_ms,
            "openai": {
                "response_id": openai_response_id,
                "request_id": openai_request_id,
                "model": model,
            },
            "result" : {
                "summary_len": len(validated_response.summary),
                "key_points_count": len(validated_response.key_points)
            
            }
        })

        return SummarizeOutput(summary=validated_response.summary, key_points=validated_response.key_points)

    except HTTPException as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": e.status_code,
            "latency_ms": latency_ms,
            "error_type": "http_exception",
            "detail": e.detail,
        })
        raise

    except OpenAIError as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": 502,
            "latency_ms": latency_ms,
            "error_type": "openai",
            "error": str(e),
            "text_len": len(text),
            "text_preview": fastapihelpers._text_preview(text),
        })
        raise HTTPException(status_code=502, detail="Upstream model request failed")

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": 500,
            "latency_ms": latency_ms,
            "error_type": type(e).__name__,
            "error": str(e),
            "text_len": len(text),
            "text_preview": fastapihelpers._text_preview(text),
        })
        raise HTTPException(status_code=500, detail="Internal server error")

ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 10},
    },
    "required": ["sentiment", "topics", "confidence"],
    "additionalProperties": False,
}

@app.post("/analyze", response_model=AnalyzeOutput)
def analyze(request: AnalyzeRequest) -> AnalyzeOutput:
    fastapihelpers._log_json(reqres_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "analyze.called",
    })
    start = time.perf_counter()   
    model="gpt-5-nano"    
    request_id = fastapihelpers._new_request_id()

    text = (request.text or "").strip()
    if not text:
        fastapihelpers._log_json(reqres_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": 400,
            "error_type": "validation",
            "error": "Text cannot be empty",
        })
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    fastapihelpers._log_json(reqres_logger, {
        "ts": fastapihelpers._now_iso(),
        "event": "analyze.start",
        "request_id": request_id,
        "text_len": len(text),
        "text_preview": fastapihelpers._text_preview(text),
        "model": model,
    })

    system_prompt = """Analyze the text.Return JSON with: 
                    - sentiment: array of strings (e.g. [\"positive\", \"cheerful\", \"happy\"])
                    - topics: array of strings
                    -confidence: number from 0 to 10
                    Return only valid JSON
                    """
    try:
        validated_response = _call_model_json(client=client,
                    ModelClass=AnalyzeOutput, 
                    schema_name="analyze_schema",
                    schema_dict=ANALYZE_SCHEMA,
                    system_prompt=system_prompt, 
                    text=text, 
                    model=model, 
                    error_logger=error_logger)

        openai_response_id = getattr(validated_response, "id", None)
        openai_request_id = getattr(validated_response, "_request_id", None)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)


        fastapihelpers._log_json(reqres_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "analyze.success",
            "request_id": request_id,
            "status_code": 200,
            "latency_ms": latency_ms,
            "openai": {
                "response_id": openai_response_id,
                "request_id": openai_request_id,
                "model": model,
            },
            "result" : {
                "sentiment_count": len(validated_response.sentiment),
                "topics_count": len(validated_response.topics),
                "confidence": validated_response.confidence
            
            }
        })  

        return AnalyzeOutput(sentiment=validated_response.sentiment, 
                              topics=validated_response.topics, 
                              confidence=validated_response.confidence)


    except HTTPException as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": e.status_code,
            "latency_ms": latency_ms,
            "error_type": "unexpected",
            "detail": str(e),
        })
        raise

    except OpenAIError as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": 502,
            "latency_ms": latency_ms,
            "error_type": "openai",
            "error": str(e),
            "text_len": len(text),
            "text_preview": fastapihelpers._text_preview(text),
        })
        raise HTTPException(status_code=502, detail="Upstream model request failed")


    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": 500,
            "latency_ms": latency_ms,
            "error_type": type(e).__name__,
            "error": str(e),
            "text_len": len(text),
            "text_preview": fastapihelpers._text_preview(text),
        })
        raise HTTPException(status_code=500, detail="Internal server error")

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
        reqres_logger,
        {"ts": fastapihelpers._now_iso(), "event": "log_test", "ok": True}
    )
    return {"status": "logged"}
