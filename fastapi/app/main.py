from typing import Union

from fastapi import FastAPI
import pika
import json
import os

app = FastAPI()

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_QUEUE = "caas"
RABBITMQ_PORT = "5672"
RABBITMQ_CREDENTIALS = pika.PlainCredentials(os.getenv("RABBITMQ_DEFAULT_USER"), os.getenv("RABBITMQ_DEFAULT_PASS"))


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/send_message")
def send_message():
    # Arguments: dataset location, algorithm type, parameters, ggf. ID
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=RABBITMQ_CREDENTIALS))
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE)
    channel.basic_publish(exchange='', routing_key=RABBITMQ_QUEUE, body=json.dumps({"file_url": "https://storage.googleapis.com/fernunihagen-ss2025-caas/data/iris_without_class.csv"}))
    connection.close()
    return {"message": "Message sent to RabbitMQ queue"}
