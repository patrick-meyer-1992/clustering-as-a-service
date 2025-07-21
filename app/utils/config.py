import os

import pytz

FASTAPI_HOST = os.getenv("FASTAPI_HOST", "caas-fastapi")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
FASTAPI_PROTOCOL = os.getenv("FASTAPI_PROTOCOL", "http")

TIMEZONE = pytz.timezone("UTC")
