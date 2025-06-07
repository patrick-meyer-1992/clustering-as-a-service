FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from central requirements file
COPY ./automl-requirements.txt /code/automl-requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/automl-requirements.txt

COPY ./code/auto_ml/autocluster /code/autocluster

RUN pip install /code/autocluster

COPY ./code/celery_conn.py /code/
COPY ./code/tasks.py /code/
COPY ./code/automl_worker.py /code/


CMD ["celery", "-A", "automl_worker", "worker", "-Q", "automl", "--loglevel=info"]