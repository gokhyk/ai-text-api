from typing import Any, Dict, Type, TypeVar
from pydantic import BaseModel
from fastapi import HTTPException
from openai import OpenAIError
from logging import Logger
import openai

import json
import time

import fastapihelpers

T = TypeVar("T", bound=BaseModel)

def _call_model_json(
        *,
        client: openai.OpenAI,
        ModelClass: Type[T],
        schema_name: str,
        schema_dict: Dict[str, Any],
        system_prompt: str,
        text: str,
        model: str = "gtp-5-nano",
        request_id: str | None = None,
        event_prefix: str = "model",
        error_logger: Logger,
) -> T:
    """
    Docstring for _call_model_json
    
    :param ModelClass: Description
    :type ModelClass: Type[T]
    :param schema_name: Description
    :type schema_name: str
    :param schema_dict: Description
    :type schema_dict: Dict[str, Any]
    :param system_prompt: Description
    :type system_prompt: str
    :param text: Description
    :type text: str
    :param model: Description
    :type model: str
    :param request_id: Description
    :type request_id: str | None
    :param event_prefix: Description
    :type event_prefix: str
    :return: Description
    :rtype: T
    """
    """
    Call OpenAI chat.completions with JSON Schema response_format and validate result
    into ModelClass. Raises HTTPException on failure with consistent status codes.
    """
    #print(type(client))
    start = time.perf_counter()

    if text is None or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Build response_format expected by OpenAI SDK
    response_format = {
        "type": "json_schema", "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema_dict,
        },
    }

    # Build system message

    system_message = {"role": "system", "content": system_prompt}

    try:
        response = client.chat.completions.create(
            model = model,
            messages = [system_message, {"role": "user", "content": text}],
            response_format = response_format,
        )

        content = response.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="Model returned no content")
        
        #parse JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=502, detail="Model returned non-JSON output") from e
        
        #Validate against Pydanticv model
        try:
            if hasattr(ModelClass, "model validate"):  # pydantic V2
                validated = ModelClass.model_validate(parsed)
            else: #Pydantic v1
                validated=ModelClass.parse_obj(parsed)
        except Exception as e:
            #Log schema mismatch, then return 502 (upstream output invalid)
            fastapihelpers._log_exception_json(error_logger, {
                "ts": fastapihelpers._now_iso(),
                "event": f"{event_prefix}.schema_validation_failed",
                "request_id": request_id,
                "model": model,
                "schema_name": schema_name,
                "raw_content_preview": _text_preview(content, 500),
                "error_type": type(e).__name__,
                "error": str(e),
            })
            raise HTTPException(status_code=502, detail="Model returned invalid structured output")
    
      # Success log (optional but helpful)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_json(reqres_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": f"{event_prefix}.success",
            "request_id": request_id,
            "status_code": 200,
            "latency_ms": latency_ms,
            "model": model,
            "schema_name": schema_name,
        })

        return validated

    except HTTPException:
        # Don't mask your intended HTTP errors
        raise

    except OpenAIError as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": f"{event_prefix}.openai_fail",
            "request_id": request_id,
            "status_code": 502,
            "latency_ms": latency_ms,
            "model": model,
            "schema_name": schema_name,
            "error_type": "openai",
            "error": str(e),
            "text_len": len(text),
        })
        raise HTTPException(status_code=502, detail="Upstream model request failed")

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        fastapihelpers._log_exception_json(error_logger, {
            "ts": fastapihelpers._now_iso(),
            "event": f"{event_prefix}.internal_fail",
            "request_id": request_id,
            "status_code": 500,
            "latency_ms": latency_ms,
            "model": model,
            "schema_name": schema_name,
            "error_type": type(e).__name__,
            "error": str(e),
            "text_len": len(text),
        })
        raise HTTPException(status_code=500, detail="Internal server error")