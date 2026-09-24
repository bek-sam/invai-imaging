"""Object storage behind a small interface: S3 (MinIO locally) or a local directory.

Every endpoint reads its inputs from and writes its outputs to storage by key; images never
travel over HTTP.
"""

import mimetypes
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import Settings, settings


class NotFoundError(Exception):
    pass


def content_type_for(key: str) -> str:
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


class Storage(Protocol):
    def download(self, key: str, dest: Path) -> None: ...
    def upload(self, src: Path, key: str) -> None: ...
    def put_bytes(self, data: bytes, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class LocalStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key.lstrip("/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"invalid key: {key}")
        return path

    def download(self, key: str, dest: Path) -> None:
        path = self._path(key)
        if not path.is_file():
            raise NotFoundError(key)
        shutil.copyfile(path, dest)

    def upload(self, src: Path, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)

    def put_bytes(self, data: bytes, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


class S3Storage:
    def __init__(self, cfg: Settings):
        import boto3
        from botocore.config import Config

        self.bucket = cfg.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.s3_endpoint or None,
            region_name=cfg.s3_region,
            aws_access_key_id=cfg.s3_access_key_id or None,
            aws_secret_access_key=cfg.s3_secret_access_key or None,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def _missing(self, err: Exception) -> bool:
        code = getattr(err, "response", {}).get("Error", {}).get("Code", "")
        return code in ("404", "NoSuchKey", "NotFound")

    def download(self, key: str, dest: Path) -> None:
        from botocore.exceptions import ClientError

        try:
            self.client.download_file(self.bucket, key, str(dest))
        except ClientError as e:
            if self._missing(e):
                raise NotFoundError(key) from e
            raise

    def upload(self, src: Path, key: str) -> None:
        self.client.upload_file(
            str(src), self.bucket, key, ExtraArgs={"ContentType": content_type_for(key)}
        )

    def put_bytes(self, data: bytes, key: str) -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type_for(key)
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if self._missing(e):
                return False
            raise


@lru_cache
def get_storage() -> Storage:
    if settings.storage == "local":
        return LocalStorage(settings.local_storage_dir)
    return S3Storage(settings)
