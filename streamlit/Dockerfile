# This Dockerfile is used for FastAPI and for the clustering worker.
FROM python:3.12-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/

CMD ["streamlit", "run", "/code/app.py", "--server.port=8501", "--server.enableCORS=false"]