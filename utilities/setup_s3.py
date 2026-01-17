from prefect_aws import S3Bucket, AwsCredentials
from dotenv import load_dotenv
import os

load_dotenv()


def create_s3_block():
    # This securely stores your AWS keys
    aws_creds = AwsCredentials(
        aws_access_key_id=os.environ.get("ACCESS_KEY"),  # Change this
        aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"),  # Change this
    )
    # Optional: Save creds separately if you want to reuse them for other tasks
    # aws_creds.save("my-aws-creds", overwrite=True)

    # 2. Create the S3 Bucket Block using those credentials
    s3_block = S3Bucket(
        bucket_name="nys-optimisation-project",  # Change this (Just the name, no 's3://')
        credentials=aws_creds,
        basepath="prefect_cache",  # Optional subfolder in your bucket
    )

    # 3. Save the S3 Bucket Block
    s3_block.save("s3-cache", overwrite=True)
    print("✓ Success! Block 's3-bucket/s3-cache' created.")


if __name__ == "__main__":
    create_s3_block()
