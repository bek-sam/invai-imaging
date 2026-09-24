from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "s3" talks to S3/MinIO; "local" maps keys to files under local_storage_dir (tests, offline dev).
    storage: Literal["s3", "local"] = "s3"
    local_storage_dir: str = "./.storage"

    s3_bucket: str = "invai-local"
    s3_endpoint: str | None = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = "invai"
    s3_secret_access_key: str | None = "invai-secret"
    s3_create_bucket: bool = True

    # Scratch space for downloads and intermediate renders (big sheets need a few GB).
    tmp_dir: str | None = None


settings = Settings()
