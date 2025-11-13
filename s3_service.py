import boto3
from botocore.client import Config
from io import BytesIO
import config

class S3Service:
    def __init__(self):
        self.bucket = config.AWS_BUCKET
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_DEFAULT_REGION,
            endpoint_url=config.AWS_ENDPOINT_URL,
            config=Config(signature_version="s3v4")
        )

    def download_file(self, key: str) -> bytes:
        buf = BytesIO()
        self.s3.download_fileobj(self.bucket, key, buf)
        buf.seek(0)
        return buf.read()

    def upload_file(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=content, ContentType=content_type)
        return key

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in
        )
