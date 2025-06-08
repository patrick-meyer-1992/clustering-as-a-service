FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from central requirements file
COPY ./automl-requirements.txt /app/automl-requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/automl-requirements.txt

COPY ./workers /app/workers

CMD ["celery", "-A", "workers.automl_worker", "worker", "-Q", "automl", "--loglevel=info"]