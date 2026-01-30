#!/usr/bin/env python3
#app/core/openapi_client.py

from openai import OpenAI
import os
from dotenv import load_dotenv

def get_openai_client() -> OpenAI:
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    return OpenAI()

client = get_openai_client()
