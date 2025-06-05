FROM python:3.12-slim

WORKDIR /app

COPY ./clustering_worker-requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./clustering /app/clustering
COPY ./clustering_worker/ /app/clustering_worker

CMD ["celery", "-A", "clustering_worker.tasks", "worker", "--loglevel=info"]