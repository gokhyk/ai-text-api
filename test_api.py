#!/usr/bin/env python3
#test_api.py

import sys
print(sys.executable)

from dotenv import load_dotenv
import os

load_dotenv()
print("KEY:", os.getenv("OPENAI_API_KEY"))

from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

print("KEY:", os.getenv("OPENAI_API_KEY"))
