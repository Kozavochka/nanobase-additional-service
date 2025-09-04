import pika
import json
import sys
from dotenv import load_dotenv
import os

# Загрузить переменные из .env
load_dotenv()

host=os.getenv("RABBIT_MQ_HOST")
port=os.getenv("RABBIT_MQ_PORT")
vhost=os.getenv("RABBIT_MQ_VHOST")
user=os.getenv("RABBIT_MQ_USER")
password=os.getenv("RABBIT_MQ_PASSWORD")

def send_message(queue_name: str, message: str):

    credentials = pika.PlainCredentials(user, password)
    connection_params = pika.ConnectionParameters(
        host=host,
        port=port,
        virtual_host=vhost,
        credentials=credentials
    )

    connection = pika.BlockingConnection(connection_params)
    channel     = connection.channel()

    """
    Отправляет сообщение в указанную очередь RabbitMQ.

    :param queue_name: Имя очереди
    :param message: Строка сообщения
    """
    # Убедимся, что очередь существует
    channel.queue_declare(queue=queue_name, durable=True)
    
    # Отправляем сообщение
    channel.basic_publish(
        exchange="",
        routing_key=queue_name,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2  # сделать сообщение устойчивым
        )
    )
