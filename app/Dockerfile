# This Dockerfile is used for FastAPI and for the clustering worker.
FROM python:3.12-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# COPY ./code /code/ # Excluded while hot reloading is enabled in docker-compose

CMD ["fastapi", "run", "main.py", "--port", "8000"]