from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Public, dev-only default. Production refuses to start with it (production_config_error).
DEV_SHARED_SECRET = "invai-imaging-dev-secret"


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

    # --- limits and service auth (T-9-5, B-19) ---
    # "production" refuses to start on the dev shared secret. The Docker image sets it.
    imaging_env: Literal["development", "test", "production"] = "development"
    # Shared secret the backend sends in the X-Imaging-Secret header. A comma-separated list
    # accepts each entry, for zero-downtime rotation (the backend sends one).
    imaging_shared_secret: str = DEV_SHARED_SECRET
    # /labels/mock and /sample-art (mock carrier labels, seed art) answer 404 unless this is set.
    imaging_dev_endpoints: bool = False
    # Heavy jobs (compose, personalization render) running at once in this process; the rest get
    # 429 + Retry-After. A 240in compose peaks near 0.5-1 GB, so size this to the task's memory.
    imaging_max_heavy_jobs: int = Field(2, ge=1)
    imaging_retry_after_s: int = Field(5, ge=1)
    # Pixel cap on any decode (raster header, SVG size, PDF page x DPI), checked before pixels
    # are decoded: 22 x 240in at 300 DPI x 1.2 is about 570M pixels.
    imaging_max_decode_pixels: int = Field(570_000_000, ge=1)
    # Listing-photo templates kept drawn per (garment, view, size), ~20-40 MB each at 2000 px.
    imaging_photo_template_cache: int = Field(4, ge=1, le=64)

    def shared_secrets(self) -> list[str]:
        return [s.strip() for s in self.imaging_shared_secret.split(",") if s.strip()]


def production_config_error(s: "Settings") -> str | None:
    """Why this config must not serve production traffic, or None."""
    if s.imaging_env != "production":
        return None
    secrets = s.shared_secrets()
    if not secrets or DEV_SHARED_SECRET in secrets:
        return "IMAGING_SHARED_SECRET must be set to a real secret in production"
    if any(len(x) < 32 for x in secrets):
        return "IMAGING_SHARED_SECRET must be at least 32 characters in production"
    return None


settings = Settings()
