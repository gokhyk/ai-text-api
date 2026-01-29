#!/usr/bin/env python3

from openai import OpenAI
import os
from dotenv import load_dotenv

def get_openai_client() -> OpenAI:
    # Central place for configuration
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    #return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return OpenAI()

client = get_openai_client()
