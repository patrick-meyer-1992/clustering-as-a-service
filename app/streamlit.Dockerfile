# This Dockerfile is used for FastAPI and for the clustering worker.
FROM python:3.12-slim

WORKDIR /code

COPY ./streamlit-requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./code/clustering /code/
COPY ./code/streamlit_app.py /code/

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]