# app/core/fastapi_client.py
from fastapi import FastAPI
from core.logging import init_loggers
from exchandlers import register_exception_handlers

def get_fastapi_client() -> FastAPI:
    app = FastAPI()

    # init loggers + attach to app.state
    init_loggers(app)

    # register exception handlers on THIS app
    register_exception_handlers(app)

    return app

app = get_fastapi_client()




# #!/usr/bin/env python3
# #app/core/fastapi_client.py

# from fastapi import FastAPI

# def get_fastapi_client() -> FastAPI:
#     return FastAPI()

# app = get_fastapi_client()