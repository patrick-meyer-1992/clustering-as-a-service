FROM python:3.12-slim

WORKDIR /code

COPY ./fastapi-requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./code/clustering /code/
COPY ./code/celery_conn.py /code/
COPY ./code/tasks.py /code/
COPY ./code/fastapi_main.py /code/

CMD ["fastapi", "run", "fastapi_main.py", "--port", "8000"]