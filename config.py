import os
from dotenv import load_dotenv

load_dotenv()  # подтягиваем .env

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # может быть None
