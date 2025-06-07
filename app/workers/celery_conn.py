from celery import Celery
import os

rmq_user = os.getenv('RABBITMQ_DEFAULT_USER')
rmq_passwd = os.getenv('RABBITMQ_DEFAULT_PASS')
rmq_host = os.getenv('RABBITMQ_HOST')
rmq_port = os.getenv('RABBITMQ_PORT')
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')

celery = Celery(
    'tasks',
    broker=f'pyamqp://{rmq_user}:{rmq_passwd}@{rmq_host}:{rmq_port}',
    backend=f'redis://{redis_host}:{redis_port}/0',
)

celery.conf.task_routes = {
    'tasks.*': {'queue': 'default'},
    'automl_worker.run_autocluster': {'queue': 'automl'},
}

celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)