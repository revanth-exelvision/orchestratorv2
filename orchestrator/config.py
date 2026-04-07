from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    max_upload_files: int = 10
    max_upload_bytes_per_file: int = 2 * 1024 * 1024
    max_total_request_bytes: int = 25 * 1024 * 1024

    log_level: str = "INFO"
    log_dir: str = "logs"
    llm_audit_enabled: bool = True

    # API route toggles. Disabled routes are not registered (404). By default only GET /health is on.
    api_health_enabled: bool = True
    api_orchestrate_enabled: bool = False  # POST /orchestrate, /orchestrate/json
    api_orchestrate_plan_enabled: bool = False
    api_orchestrate_execute_enabled: bool = False
    api_orchestrate_tools_enabled: bool = False
    api_orchestrate_flows_enabled: bool = False  # GET /flows, POST /flows/{id}

    @classmethod
    def with_all_orchestration_routes(cls) -> Settings:
        """Enable every HTTP route (health + all ``/orchestrate*`` handlers)."""
        return cls(
            api_health_enabled=True,
            api_orchestrate_enabled=True,
            api_orchestrate_plan_enabled=True,
            api_orchestrate_execute_enabled=True,
            api_orchestrate_tools_enabled=True,
            api_orchestrate_flows_enabled=True,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
