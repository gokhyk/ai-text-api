#main.py
from __future__ import annotations

import json

import os
import time
import uuid

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

class AnalysisOutput(BaseModel):
    """Structured output expected from the model."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")

    sentiment: list[NonEmptyStr] = Field(default_factory=list, description="Sentiment of the text.")
    topics: list[NonEmptyStr] = Field(default_factory=list, description="Topics in the text.")
    confidence: float = Field(ge=0, le=10, description="confidence level of the system on the output")
    

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


class AnalyzeRequest(BaseModel):
    text: NonEmptyStr

# ---- Endpoint ----
from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)

def model_call_helper(ModelClass: Type[T], response_format: dict,
                      system_prompt: dict, text: str, model: str) -> T:
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[system_prompt, {"role": "user", "content": text}],
            response_format=response_format,
        )

        content = response.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="Model returned no content")
                                
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=502, detail="Model returned non-JSON output") from e
        
        try:
            if hasattr(ModelClass, "model validate"):
                validated = ModelClass.model_validate(parsed)
            else: #Pydantic v1
                validated=ModelClass.parse_obj(parsed)
        except ValidationError as e:
            error_logger.exception(json.dumps({
                "event": "schema_validation_failed",
                "error": str(e),
                "raw_content_preview": _text_preview(content, 500),
            }, ensure_ascii=False))
            raise HTTPException(status_code=502, detail="Model returned invalid structured output")
    
        return validated
    
    except HTTPException:
        #IMPORTANT: don't swallow your oHTTP errors
        raise

    except OpenAIError as e:
        _log_exception_json(error_logger, {
            "event": "model_call.fail",
            "status_code": 502,
            "error_type": "openai",
            "error": str(e),
            "text_len": len(text),
            "text_preview": _text_preview(text),
        })
        raise HTPPException(status_code=502, detail="Upstream model request failed")

    #         model = model,
    #         messages = [system_prompt, {"role": "user", "content": text}],
    #         response_format = schema_dict
    #     )

    #     openai_response_id = getattr(response, "id", None)
    #     openai_request_id = getattr(response, "_request_id", None)

    #     content = response.choices[0].message.content
    #     if not content:
    #         raise HTTPException(status_code=502, detail="Model return no content")
        

    #     #Parse JSON (fallback if needed)
    #     try:
    #         parsed = json.loads(content)
    #     except json.JSONDecodeError as e:
    #         raise HTTPException(status_code=502, detail="Model returned non-JSON output") from e

    #     try:
    #         if hasattr(schema_name, "model_validate"):   # Pydantic v2
    #             validated = schema_name.model_validate(parsed)
    #         else:  # Pydantic v1
    #             validated = schema_name.parse_obj(parsed)
    #     except ValidationError as e:
    #         error_logger.exception(json.dumps({
    #             #"ts": _now_iso(),
    #             "event": "schema_validation_failed",
    #             "error": str(e),
    #             "raw_content_preview": _text_preview(content, 500),
    #         }, ensure_ascii=False))
    #         raise HTTPException(status_code=502, detail="Model returned invalid structured output")
        
    # except Exception as e:
    #     #latency_ms = round((time.perf_counter() - start) * 1000, 2)
    #     _log_exception_json(error_logger, {
    #         #"ts": _now_iso(),
    #         "event": "summarize.fail",
    #         #"request_id": request_id,
    #         "status_code": 500,
    #         #"latency_ms": latency_ms,
    #         "error_type": type(e).__name__,
    #         "error": str(e),
    #         "text_len": len(text),
    #         "text_preview": _text_preview(text),
    #     })
    #     raise HTTPException(status_code=500, detail="Internal server error")
    
    # return SummarizeOutput(summary=validated.summary, key_points=validated.key_points)

SUMMARY_SCHEMA =  {
                "type": "object",
                    "properties": {
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "key_points"],
                "additionalProperties": False,
            }

@app.post("/summy", response_model=SummarizeOutput)
def summy(request: SummarizeRequest) -> SummarizeOutput:
    text = (request.text or "").strip()
    #schema_name = SummarizeOutput

    # system_prompt = {
    #                 "role": "system",
    #                 "content": "Summarize the following text clearly and concisely and provide key points."
    #             }
    system_prompt = "Summarize the following text clearly and concisely and provide key points."
    model="gpt-5-nano"
    so = _call_model_json(client=client,
                          ModelClass=SummarizeOutput, 
                          schema_name="summary_schema",
                          schema_dict=SUMMARY_SCHEMA,
                          system_prompt=system_prompt, 
                          text=text, 
                          model=model, 
                          error_logger=error_logger)

    return so

@app.post("/summarize", response_model=SummarizeOutput)
def summarize(request: SummarizeRequest) -> SummarizeOutput:

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
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=502, detail="Model returned non-JSON output") from e


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

        return SummarizeOutput(summary=validated.summary, key_points=validated.key_points)


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

    except OpenAIError as e:
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


@app.post("/analyze", response_model=AnalysisOutput)
def analyze(request: AnalyzeRequest) -> AnalysisOutput:

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
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=502, detail="Model returned non-JSON output") from e

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

        return AnalysisOutput(sentiment=validated.sentiment, topics=validated.topics, confidence=validated.confidence)


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

    except OpenAIError as e:
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
