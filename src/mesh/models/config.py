"""Runtime configuration for the mesh, loaded from environment or .env."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr so the key cannot leak through a repr, log line, or traceback.
    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")

    chat_model: str = Field("gpt-5-nano", validation_alias="MESH_CHAT_MODEL")
    embed_model: str = Field("text-embedding-3-small", validation_alias="MESH_EMBED_MODEL")

    route_confidence_threshold: float = Field(
        0.6, validation_alias="MESH_ROUTE_CONFIDENCE_THRESHOLD"
    )
