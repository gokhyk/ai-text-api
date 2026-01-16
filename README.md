# AI Text Analysis API

A small FastAPI service that uses OpenAI models to **summarize text** and **analyze sentiment/topics** with **strict structured JSON output**.

This project demonstrates:
- Building a production-style REST API
- Forcing structured LLM output with JSON Schema
- Validating model responses with Pydantic
- Request/response logging and error handling

---

## Features

- `/summarize` 
  Returns a concise summary and key points

- `/analyze` 
  Returns sentiment, topics, and a confidence score (0–10)

- Strict JSON schema enforcement
- Input validation (non-empty text)
- Structured logging (JSONL)
- Health and readiness endpoints

---

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic (v1/v2 compatible)
- OpenAI API
- Uvicorn

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone <your-repo-url>
cd ai-text-api
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

pip install fastapi uvicorn python-dotenv openai

### 3. Set environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here

### Running the API
uvicorn main:app --reload


API will be available at:

http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs

### API Usage
#### Summarize

POST /summarize

{
  "text": "FastAPI is a modern web framework for building APIs with Python."
}


#### Response

{
  "summary": "FastAPI is a modern Python framework for building APIs.",
  "key_points": [
    "Designed for speed and ease of use",
    "Uses Python type hints",
    "Includes automatic API documentation"
  ]
}

### Analyze

POST /analyze

{
  "text": "I really enjoy working with FastAPI. It is fast and easy to use."
}


#### Response

{
  "sentiment": ["positive"],
  "topics": ["FastAPI", "developer experience"],
  "confidence": 9.2
}

#### Logging

Request/response logs: open_api_log_file.txt

Error logs: open_api_error_file.txt

Logs are written in JSON format and include:

request_id

latency

model used

error details (when applicable)

#### Health Checks

/health – API health

/log-test – verifies logging pipeline

### Project Goal

This project is part of a structured learning plan to build production-ready AI services with:

deterministic outputs

strong validation

debuggable logs

It serves as a foundation for more advanced AI APIs.
