from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    max_upload_files: int = 10
    max_upload_bytes_per_file: int = 2 * 1024 * 1024
    max_total_request_bytes: int = 25 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
