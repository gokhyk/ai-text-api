from typing import Any, Dict, Type, TypeVar, Optional
from pydantic import BaseModel
from fastapi import HTTPException
from openai import OpenAIError
from logging import Logger, INFO, ERROR
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
        schema_dict: Optional[Dict[str, Any]] = None,
        system_prompt: str,
        text: str,
        model: str = "gpt-5-nano",
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

    if schema_dict is None:
        schema_dict = _openai_strictify_json_schema(pydantic_to_json_schema(ModelClass))
        #schema_dict = pydantic_to_json_schema(ModelClass)
    start = time.perf_counter()

    if text is None or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
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
            if hasattr(ModelClass, "model_validate"):  # pydantic V2
                validated = ModelClass.model_validate(parsed)
            else: #Pydantic v1
                validated=ModelClass.parse_obj(parsed)
        except Exception as e:
            #Log schema mismatch, then return 502 (upstream output invalid)
            fastapihelpers._log_json(error_logger, {
                "ts": fastapihelpers._now_iso(),
                "called": " by _call_model_json Exception",
                "event": f"{event_prefix}.schema_validation_failed",
                "request_id": request_id,
                "model": model,
                "schema_name": schema_name,
                "raw_content_preview": fastapihelpers._text_preview(content, 500),
                "error_type": type(e).__name__,
                "error": str(e),
            }ERROR)
            raise HTTPException(status_code=502, detail="Model returned invalid structured output")

        return validated

    except HTTPException:
        # Don't mask your intended HTTP errors
        raise

    except OpenAIError as e:
        raise
        # fastapihelpers._log_exception_json(error_logger, {
        #     "ts": fastapihelpers._now_iso(),
        #     "event": f"{event_prefix}.openai_error",
        #     "request_id": request_id,
        #     "model": model,
        #     "schema_name": schema_name,
        #     "error_type": type(e).__name__,
        #     "error": str(e),
        # })
        # raise HTTPException(status_code=502, detail=f"Upstream model request failed: {str(e)}")


    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    


# schema_utils.py (or inside callmodeljson.py / )
def pydantic_to_json_schema(ModelClass: Type[BaseModel]) -> Dict[str, Any]:
    """
    Return a JSON Schema dict from a Pydantic model v1 or v2
    """

    #Pydantic v2
    if hasattr(ModelClass, "model_json_schema"):
        return ModelClass.model_json_schema()
    #Pydantic v1
    return ModelClass.schema()


def _openai_strictify_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI strict json_schema expects that for any object with properties,
    'required' exists and includes every property key.
    This walks the schema and enforces that rule.
    """
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            # If it's an object schema with properties, enforce required
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                props = node["properties"]
                node["required"] = list(props.keys())
                # OpenAI also likes this strictness
                node.setdefault("additionalProperties", False)

            # Recurse into dict values
            for v in node.values():
                walk(v)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    schema = dict(schema)  # shallow copy
    walk(schema)
    return schema
