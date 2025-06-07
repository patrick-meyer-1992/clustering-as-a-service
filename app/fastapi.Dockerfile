FROM python:3.12-slim

WORKDIR /app

COPY ./fastapi-requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./clustering /app/clustering
COPY ./workers/ /app/workers
COPY ./api /app/api

CMD ["fastapi", "run", "main.py", "--port", "8000"]