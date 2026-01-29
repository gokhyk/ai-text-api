#!/usr/bin/env python3

from fastapi import FastAPI

def get_fastapi_client() -> FastAPI:
    return FastAPI()

app = get_fastapi_client()