#main.py
from __future__ import annotations

import json
import logging
import os
import time
import uuid

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Annotated

import openai

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field

import traceback

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

def _new_request_id() -> str:
    return uuid.uuid4().hex

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _text_preview(text: str, limit: int = 1000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated, len={len(text)}]"

def _setup_jsonl_logger(name: str, path: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger    

reqres_logger = _setup_jsonl_logger("openai.reqres", log_file, logging.INFO)
error_logger = _setup_jsonl_logger("openai.error", error_file, logging.ERROR)

def _log_json(logger: logging.Logger, payload: dict[str, Any], level: int = logging.INFO) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False))

def _traceback_frames() -> list[dict[str, Any]]:
    tb = traceback.TracebackException.from_exception(Exception()).stack
    return [{"file": f.filename, "line": f.lineno, "func": f.name} for f in tb]

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

class AnalysisOutput(BaseModel):
    """Structured output expected from the model."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")

    sentiment: list[NonEmptyStr] = Field(default_factory=list, description="Sentiment of the text.")
    topics: list[NonEmptyStr] = Field(default_factory=list, description="Topics in the text.")
    confidence: float = Field(description="confidence level of the system on the output")
    

    if ConfigDict is None:  # Pydantic v1 fallback
        class Config:
            extra = "forbid"

# ---- Request / Response Schemas ----

# 1. Define the desired output structure using Pydantic
#class AnalysisOutput(BaseModel):
#    summary: str = Field(description="A concise summary of the text.")
#    key_points: list[str] = Field(description="A bulleted list of the most important points.")

class SummarizeRequest(BaseModel):
    text: NonEmptyStr


class SummarizeResponse(BaseModel):
    summary: NonEmptyStr
    key_points: list[NonEmptyStr] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    text: NonEmptyStr

class AnalyzeResponse(BaseModel):
    sentiment: list[NonEmptyStr] = Field(default_factory=list)
    topics: list[NonEmptyStr] = Field(default_factory=list)
    confidence: float
# ---- Endpoint ----

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    return summarize_v1(request)


@app.post("/v1/summarize", response_model=SummarizeResponse)
def summarize_v1(request: SummarizeRequest) -> SummarizeResponse:

    request_id = _new_request_id()
    start = time.perf_counter()

    text = (request.text or "").strip()
    if not text:
        _log_json(reqres_logger, {
            "ts": _now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": 400,
            "error_type": "validation",
            "error": "Text cannot be empty",
        })
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    _log_json(reqres_logger, {
        "ts": _now_iso(),
        "event": "summarize.called",
        "text_len": len(text),  })
        
    _log_json(reqres_logger, {
        "ts": _now_iso(),
        "event": "summarize.start",
        "request_id": request_id,
        "text_len": len(text),
        "text_preview": _text_preview(text),
        "model": "gpt-5-nano",
    })

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            #model="this-model-does-not-exist",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following text clearly and concisely and provide key points."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format= {
                "type": "json_schema",
                "json_schema": {
                    "name": "summary_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                        "summary": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "key_points"],
                    "additionalProperties": False,
                    },
                },
            },
        )

        openai_response_id = getattr(response, "id", None)
        openai_request_id = getattr(response, "_request_id", None)

        content = response.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="Model return no content")
        

        #Parse JSON (fallback if needed)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"summary": content.strip(), "key_points": []}


        try:
            if hasattr(SummarizeOutput, "model_validate"):   # Pydantic v2
                validated = SummarizeOutput.model_validate(parsed)
            else:  # Pydantic v1
                validated = SummarizeOutput.parse_obj(parsed)
        except ValidationError as e:
            error_logger.exception(json.dumps({
                "ts": _now_iso(),
                "event": "schema_validation_failed",
                "error": str(e),
                "raw_content_preview": _text_preview(content, 500),
            }, ensure_ascii=False))
            raise HTTPException(status_code=502, detail="Model returned invalid structured output")

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        _log_json(reqres_logger, {
            "ts": _now_iso(),
            "event": "summarize.success",
            "request_id": request_id,
            "status_code": 200,
            "latency_ms": latency_ms,
            "openai": {
                "response_id": openai_response_id,
                "request_id": openai_request_id,
                "model": "gpt-5-nano",
            },
            "result" : {
                "summary_len": len(validated.summary),
                "key_points_count": len(validated.key_points)
            
            }
        })

        return SummarizeResponse(summary=validated.summary, key_points=validated.key_points)


    except HTTPException as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_exception_json(error_logger, {
            "ts": _now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": e.status_code,
            "latency_ms": latency_ms,
            "error_type": "http_exception",
            "detail": e.detail,
        })
        raise

    except openai.OpenAIError as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_exception_json(error_logger, {
            "ts": _now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": 502,
            "latency_ms": latency_ms,
            "error_type": "openai",
            "error": str(e),
            "text_len": len(text),
            "text_preview": _text_preview(text),
        })
        raise HTTPException(status_code=502, detail="Upstream model request failed")


    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_exception_json(error_logger, {
            "ts": _now_iso(),
            "event": "summarize.fail",
            "request_id": request_id,
            "status_code": 500,
            "latency_ms": latency_ms,
            "error_type": type(e).__name__,
            "error": str(e),
            "text_len": len(text),
            "text_preview": _text_preview(text),
        })
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:

    request_id = _new_request_id()
    start = time.perf_counter()

    text = (request.text or "").strip()
    if not text:
        _log_json(reqres_logger, {
            "ts": _now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": 400,
            "error_type": "validation",
            "error": "Text cannot be empty",
        })
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    _log_json(reqres_logger, {
        "ts": _now_iso(),
        "event": "analyze.called",
        "text_len": len(text),  })
        
    _log_json(reqres_logger, {
        "ts": _now_iso(),
        "event": "analyze.start",
        "request_id": request_id,
        "text_len": len(text),
        "text_preview": _text_preview(text),
        "model": "gpt-5-nano",
    })

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            #model="this-model-does-not-exist",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the text.Return JSON with: \n"
                    ' - sentiment: array of strings (e.g. ["positive", "cheerful", "happy"])\n'
                    "- topics: array of strings\n"
                    "-confidence: number from 0 to 10\n"
                    "Return only valid JSON"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format= {
                "type": "json_schema",
                "json_schema": {
                    "name": "analyze_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                        "sentiment": {"type": "array", "items": {"type": "string"}},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 10}
                    },
                    "required": ["sentiment", "topics", "confidence"],
                    "additionalProperties": False,
                    },
                },
            },
        )

        openai_response_id = getattr(response, "id", None)
        openai_request_id = getattr(response, "_request_id", None)

        content = response.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="Model return no content")
        

        #Parse JSON (fallback if needed)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"sentiment": ["unknown"], "topics": [], "confidence": 0}


        try:
            if hasattr(AnalysisOutput, "model_validate"):   # Pydantic v2
                validated = AnalysisOutput.model_validate(parsed)
            else:  # Pydantic v1
                validated = AnalysisOutput.parse_obj(parsed)
        except ValidationError as e:
            error_logger.exception(json.dumps({
                "ts": _now_iso(),
                "event": "schema_validation_failed",
                "error": str(e),
                "raw_content_preview": _text_preview(content, 500),
            }, ensure_ascii=False))
            raise HTTPException(status_code=502, detail="Model returned invalid structured output")

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        _log_json(reqres_logger, {
            "ts": _now_iso(),
            "event": "analyze.success",
            "request_id": request_id,
            "status_code": 200,
            "latency_ms": latency_ms,
            "openai": {
                "response_id": openai_response_id,
                "request_id": openai_request_id,
                "model": "gpt-5-nano",
            },
            "result" : {
                "sentiment_count": len(validated.sentiment),
                "topics_count": len(validated.topics),
                "confidence": validated.confidence
            
            }
        })  

        return AnalyzeResponse(sentiment=validated.sentiment, topics=validated.topics, confidence=validated.confidence)


    except HTTPException as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_exception_json(error_logger, {
            "ts": _now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": e.status_code,
            "latency_ms": latency_ms,
            "error_type": "unexpected",
            "detail": str(e),
        })
        raise

    except openai.OpenAIError as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_exception_json(error_logger, {
            "ts": _now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": 502,
            "latency_ms": latency_ms,
            "error_type": "openai",
            "error": str(e),
            "text_len": len(text),
            "text_preview": _text_preview(text),
        })
        raise HTTPException(status_code=502, detail="Upstream model request failed")


    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_exception_json(error_logger, {
            "ts": _now_iso(),
            "event": "analyze.fail",
            "request_id": request_id,
            "status_code": 500,
            "latency_ms": latency_ms,
            "error_type": type(e).__name__,
            "error": str(e),
            "text_len": len(text),
            "text_preview": _text_preview(text),
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
    _log_json(
        reqres_logger,
        {"ts": _now_iso(), "event": "log_test", "ok": True}
    )
    return {"status": "logged"}
