FROM python:3.12-slim

WORKDIR /app

COPY ./streamlit-requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./clustering /app/clustering
COPY ./frontend /app/frontend
COPY ./utils /app/utils

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.enableCORS=false"]