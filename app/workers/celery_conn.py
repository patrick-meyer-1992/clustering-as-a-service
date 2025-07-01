import os

from celery import Celery

RMQ_USER = "guest"
RMQ_PASSWD = "guest"
RMQ_HOST = os.getenv("RABBITMQ_HOST")
RMQ_PORT = os.getenv("RABBITMQ_PORT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")

celery = Celery(
    "tasks",
    broker=f"pyamqp://{RMQ_USER}:{RMQ_PASSWD}@{RMQ_HOST}:{RMQ_PORT}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)

celery.conf.task_routes = {
    "tasks.*": {"queue": "default"},
    "automl_worker.run_autocluster": {"queue": "automl"},
}

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
