#main.py
from __future__ import annotations

import json
import logging
import os
import time

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Annotated

import openai

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field


try:
    #pydantic v2
    from pydantic import ConfigDict
except Exception:   #pragma: no cover
    ConfigDict = None #type: ignore[assignment]

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

app = FastAPI()
client = OpenAI()  # <-- CORRECT

# ----------------------------------
# Logging 
# ----------------------------------
log_file = "open_api_log_file.txt"
error_file = "open_api_error_file.txt"

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

def _log_json(logger: logging.Logger, payload: dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False))

# ----------------------------
# Schemas
# ----------------------------
NonEmptyStr = Annotated[str, Field(min_length=1)]

class AnalysisOutput(BaseModel):
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


# ---- Request / Response Schemas ----

# 1. Define the desired output structure using Pydantic
#class AnalysisOutput(BaseModel):
#    summary: str = Field(description="A concise summary of the text.")
#    key_points: list[str] = Field(description="A bulleted list of the most important points.")

class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str
    keypoints: list[str]


# ---- Endpoint ----

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:

    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", 
                 "content": "Summarize the following text clearly and concisely and provide key points"},
                {"role": "user", 
                 "content": text}
            ],
            # Specify the structured output format
            response_format={"type": "json_schema",
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
                "additionalProperties": False,}}}
        )
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response_id = getattr(response, "id", None)
        request_id = getattr(response, "_request_id", None)

        content = response.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="Model returned no content")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"summary": content.strip(), "key_points": []}

        validated = AnalysisOutput.model_validate(parsed)

        _log_json(
            reqres_logger,
            {
                "ts": _now_iso(),
                "latency": duration_ms,
                "event": "openai.chat.completions",
                "request": {
                    "model": "gpt-5-nano",
                    "text_len": len(text),
                    "text_preview": _text_preview(text),
                },
                "response": {
                    "response_id": response_id,
                    "request_id": request_id,
                    "parsed": {"summary": validated.summary, "key_points": validated.key_points},
                },
            },
        )

        return SummarizeResponse(summary=validated.summary, keypoints=validated.key_points)

    except openai.OpenAIError as e:
        error_logger.exception(
            json.dumps(
                {
                    "ts": _now_iso(),
                    "event": "openai.error",
                    "error": str(e),
                    "text_len": len(text),
                    "text_preview": _text_preview(text),
                },
                ensure_ascii=False,
            )
        )
        raise HTTPException(status_code=502, detail=str(e))

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
