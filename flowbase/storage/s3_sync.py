"""S3 synchronization utilities for artifacts."""

import logging
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


logger = logging.getLogger(__name__)


class S3Sync:
    """Handles syncing artifacts to/from S3."""

    def __init__(self, bucket: str, prefix: str = ""):
        """Initialize S3 sync client.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix for all S3 keys (e.g., "flowbase-greyhounds/")
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 syncing. Install with: pip install boto3"
            )

        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.s3_client = boto3.client("s3")

        logger.info(f"Initialized S3 sync: s3://{bucket}/{prefix}")

    def upload_file(self, local_path: str | Path, s3_key: str) -> bool:
        """Upload a file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 key (will be prefixed with self.prefix)

        Returns:
            True if successful, False otherwise
        """
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"Local file not found: {local_path}")
            return False

        # Add prefix to s3_key
        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            logger.info(f"Uploading {local_path} to s3://{self.bucket}/{full_key}")
            self.s3_client.upload_file(str(local_path), self.bucket, full_key)
            logger.info(f"Successfully uploaded to s3://{self.bucket}/{full_key}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            return False

    def upload_directory(self, local_dir: str | Path, s3_prefix: str) -> bool:
        """Upload a directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 key prefix (will be prefixed with self.prefix)

        Returns:
            True if all files uploaded successfully, False otherwise
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.error(f"Local directory not found: {local_dir}")
            return False

        success = True
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Calculate relative path
                rel_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{rel_path}"

                if not self.upload_file(file_path, s3_key):
                    success = False

        return success

    def download_file(self, s3_key: str, local_path: str | Path) -> bool:
        """Download a file from S3.

        Args:
            s3_key: S3 key (will be prefixed with self.prefix)
            local_path: Local file path

        Returns:
            True if successful, False otherwise
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Add prefix to s3_key
        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            logger.info(f"Downloading s3://{self.bucket}/{full_key} to {local_path}")
            self.s3_client.download_file(self.bucket, full_key, str(local_path))
            logger.info(f"Successfully downloaded to {local_path}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    def download_directory(self, s3_prefix: str, local_dir: str | Path) -> bool:
        """Download a directory from S3.

        Args:
            s3_prefix: S3 key prefix (will be prefixed with self.prefix)
            local_dir: Local directory path

        Returns:
            True if all files downloaded successfully, False otherwise
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Add prefix to s3_prefix
        full_prefix = f"{self.prefix}/{s3_prefix}" if self.prefix else s3_prefix

        try:
            # List all objects with the prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=full_prefix)

            success = True
            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    # Remove the full prefix to get relative path
                    rel_key = s3_key[len(full_prefix):].lstrip("/")
                    if not rel_key:  # Skip the prefix itself
                        continue

                    local_file = local_dir / rel_key
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        self.s3_client.download_file(self.bucket, s3_key, str(local_file))
                        logger.info(f"Downloaded {s3_key} to {local_file}")
                    except (ClientError, NoCredentialsError) as e:
                        logger.error(f"Failed to download {s3_key}: {e}")
                        success = False

            return success
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to list/download directory from S3: {e}")
            return False

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3.

        Args:
            s3_key: S3 key (will be prefixed with self.prefix)

        Returns:
            True if file exists, False otherwise
        """
        # Add prefix to s3_key
        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError:
            return False

    def sync_artifact(
        self,
        local_path: str | Path,
        artifact_type: str,
        artifact_name: str
    ) -> Optional[str]:
        """Sync an artifact (file or directory) to S3.

        Args:
            local_path: Local file or directory path
            artifact_type: Type of artifact (e.g., "models", "features", "datasets")
            artifact_name: Name of the artifact

        Returns:
            S3 URL if successful, None otherwise
        """
        local_path = Path(local_path)

        if local_path.is_file():
            s3_key = f"{artifact_type}/{artifact_name}"
            if self.upload_file(local_path, s3_key):
                return f"s3://{self.bucket}/{self.prefix}/{s3_key}".replace("//", "/")
        elif local_path.is_dir():
            s3_prefix = f"{artifact_type}/{artifact_name}"
            if self.upload_directory(local_path, s3_prefix):
                return f"s3://{self.bucket}/{self.prefix}/{s3_prefix}".replace("//", "/")

        return None


__all__ = ["S3Sync"]
