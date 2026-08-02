"""Object storage abstraction (P2) with local-disk fallback.

Two backends:

- ``local`` (default): files stay under ``uploads/`` and are served by the
  existing ``/uploads`` static mount. ``upload_file`` is a no-op that just maps
  the local path to its ``/uploads/...`` URL — identical to the original
  behaviour, so nothing breaks without object storage.
- ``s3`` (MinIO / AWS S3): uploads the file to a bucket and returns its URL.

Any S3 problem (missing ``boto3``, unreachable endpoint, upload error) degrades
to returning the local ``/uploads`` URL, keeping the app functional.
"""

import os
import logging

logger = logging.getLogger(__name__)


def local_public_url(fs_path: str):
    """Map a filesystem path under ``uploads/`` to its ``/uploads/...`` URL."""
    if not fs_path:
        return None
    normalized = fs_path.replace("\\", "/").lstrip("./")
    marker = "uploads/"
    idx = normalized.find(marker)
    if idx == -1:
        return None
    return "/" + normalized[idx:]


class ObjectStorage:
    def __init__(self, config):
        cfg = getattr(config, "object_storage", None)
        self.backend = (getattr(cfg, "backend", "local") or "local").strip().lower()
        self.endpoint_url = getattr(cfg, "endpoint_url", "") or ""
        self.bucket = getattr(cfg, "bucket", "") or "medical-assistant"
        self.access_key = getattr(cfg, "access_key", "") or ""
        self.secret_key = getattr(cfg, "secret_key", "") or ""
        self.region = getattr(cfg, "region", "") or ""
        self.public_base_url = getattr(cfg, "public_base_url", "") or ""
        self._client = None

        if self.backend == "s3":
            self._client = self._build_s3_client()
            if self._client is None:
                logger.warning("Object storage 's3' requested but unavailable; using local disk.")

    def _build_s3_client(self):
        try:
            import boto3  # optional dependency

            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url or None,
                aws_access_key_id=self.access_key or None,
                aws_secret_access_key=self.secret_key or None,
                region_name=self.region or None,
            )
            # Best-effort bucket existence check / creation.
            try:
                client.head_bucket(Bucket=self.bucket)
            except Exception:
                try:
                    client.create_bucket(Bucket=self.bucket)
                    logger.info("Created object storage bucket '%s'.", self.bucket)
                except Exception as e:
                    logger.warning("Could not ensure bucket '%s' (%s).", self.bucket, e)
            return client
        except Exception as e:
            logger.warning("S3/MinIO client init failed (%s); falling back to local storage.", e)
            return None

    @property
    def active_backend(self) -> str:
        """Effective backend after considering client availability."""
        return "s3" if (self.backend == "s3" and self._client is not None) else "local"

    def _s3_public_url(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url.rstrip('/')}/{key}"
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        return f"/{self.bucket}/{key}"

    def upload_file(self, local_path: str, key: str = None):
        """Upload a local file and return a public URL.

        For the local backend (or on any S3 failure) returns the ``/uploads/...``
        URL of the existing on-disk file.
        """
        if self.active_backend == "s3":
            key = key or os.path.basename(local_path)
            try:
                self._client.upload_file(local_path, self.bucket, key)
                return self._s3_public_url(key)
            except Exception as e:
                logger.warning("S3 upload failed for %s (%s); using local URL.", local_path, e)
        return local_public_url(local_path)
