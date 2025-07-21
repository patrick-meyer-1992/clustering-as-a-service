FROM python:3.12-slim

WORKDIR /app

COPY ./clustering_worker-requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./clustering /app/clustering
COPY ./workers/ /app/workers
COPY ./utils /app/utils

CMD ["celery", "-A", "workers.worker_tasks", "worker", "--loglevel=info"]