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

RUN pip install git+https://github.com/wywongbd/autocluster.git@master && \
    pip show autocluster6

RUN python -c "import autocluster; print('AutoCluster import successful')"

COPY ./workers /app/workers

CMD ["celery", "-A", "workers.automl_worker", "worker", "-Q", "automl", "--loglevel=info", "--pool=threads", "--concurrency=1"]
