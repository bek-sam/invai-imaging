from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    s3_bucket: str = "invai-local"
    s3_endpoint: str | None = None
    redis_url: str = "redis://localhost:6379"
    dpi: int = 300
    sheet_width_in: float = 22.0
    spacing_in: float = 0.25


settings = Settings()
