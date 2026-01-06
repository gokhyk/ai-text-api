from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI
import openai

from dotenv import load_dotenv
import os
import json

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

app = FastAPI()
client = OpenAI()  # <-- CORRECT


log_file = "open_api_log_file.txt"
error_file = "open_api_error_file.txt"

# ---- Request / Response Schemas ----

# 1. Define the desired output structure using Pydantic
class AnalysisOutput(BaseModel):
    summary: str = Field(description="A concise summary of the text.")
    key_points: list[str] = Field(description="A bulleted list of the most important points.")

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str


# ---- Endpoint ----

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "Summarize the following text clearly and concisely and provide key points"},
                {"role": "user", "content": request.text}
            ],
            # Specify the structured output format
            response_format={"type": "json_schema",
                             "json_schema": {"name": "summary_schema",
                                "schema": {
                                 "type": "object", 
                                    "properties": {                                
                                    "summary": {"type": "string"},
                                    "key_points": {"type": "array", "items": {"type": "string"}},
                }, 
                "required": ["summary", "key_points"], 
                "additionalProperties": False,}}}
        )

        content = response.choices[0].message.content
        if content is None:
            raise HTTPException(status_code=502, detail="Model returned no content")
        summary = content.strip()
        with open(log_file, 'a') as lf:
            lf.write(request.text)
            lf.write(json.dumps(summary))

        return {"summary": summary}
    
    except openai.OpenAIError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )

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