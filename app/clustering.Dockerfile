FROM python:3.12-slim

WORKDIR /code

COPY ./clustering-requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./code/clustering /code/
COPY ./code/celery_conn.py /code/
COPY ./code/tasks.py /code/

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]