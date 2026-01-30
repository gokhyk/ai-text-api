# app/core/logging.py
import logging
import fastapihelpers

LOG_FILE = "open_api_log_file.txt"
ERROR_FILE = "open_api_error_file.txt"

def init_loggers(app) -> None:
    # JSONL loggers (each line is JSON)
    reqres_logger = fastapihelpers._setup_jsonl_logger("openai.reqres", LOG_FILE, logging.INFO)
    error_logger = fastapihelpers._setup_jsonl_logger("openai.error", ERROR_FILE, logging.ERROR)

    # Store on app.state so everyone can access without globals
    app.state.reqres_logger = reqres_logger
    app.state.error_logger = error_logger
