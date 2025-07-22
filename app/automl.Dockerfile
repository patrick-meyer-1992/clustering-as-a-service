FROM python:3.10-slim

WORKDIR /app

RUN mkdir -p /app/plots

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    graphviz \
    liblapack-dev \
    gfortran \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./automl-requirements.txt /app/automl-requirements.txt
RUN grep -v '^autocluster' /app/automl-requirements.txt > /tmp/requirements.txt && \
    pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

RUN pip install pytest pytest-cov

RUN pip install git+https://github.com/wywongbd/autocluster.git@master && \
    pip show autocluster

RUN python -c "import autocluster; print('AutoCluster import successful')"

COPY ./workers /app/workers
COPY ./utils /app/utils

ENV PYTHONPATH=/app

CMD ["celery", "-A", "workers.automl.automl_tasks", "worker", "-Q", "automl", "--loglevel=info", "--pool=solo", "--concurrency=2"]
