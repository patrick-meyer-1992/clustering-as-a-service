import pika
import json
import os
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_QUEUE = "caas"
RABBITMQ_PORT = "5672"
RABBITMQ_CREDENTIALS = pika.PlainCredentials(os.getenv("RABBITMQ_DEFAULT_USER"), os.getenv("RABBITMQ_DEFAULT_PASS"))

def load_file(file_url):
    try:
        df = pd.read_csv(file_url)
        return df
    except Exception as e:
        logging.error(f"Failed to read CSV file: {file_url}. Error: {e}")
        return None    

def cluster(message):
    # Perform computations with the message
    logging.debug(f"Processing message: {message}")
    # Example computation
    df = load_file(message["file_url"])
    if df is None:
        logging.error("Failed to load data from cloud storage.")
        return
    else:
        model = KMeans(n_clusters=3)
        model.fit(df.to_numpy())
        labels = model.labels_.tolist()
        centers = model.cluster_centers_.tolist()
        logging.info(f"Labels: {labels}\n Centers: {centers}")

def callback(ch, method, properties, body):
    logging.debug("Received message from RabbitMQ")
    try:
        message = json.loads(body)
        cluster(message)
    except json.JSONDecodeError:
        logging.error("Failed to decode message. Ensure it is in JSON format.")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():

    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=RABBITMQ_CREDENTIALS))
    channel = connection.channel()

    channel.queue_declare(queue=RABBITMQ_QUEUE)
    logging.info(f"Waiting for messages in queue: {RABBITMQ_QUEUE}")

    channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.info("Stopping consumer...")
        channel.stop_consuming()
    finally:
        connection.close()

if __name__ == "__main__":
    main()
