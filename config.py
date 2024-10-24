from functools import cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_model: str = "gpt-4o-mini"
    hf_embeddings_model: str = "intfloat/multilingual-e5-base"
    travel_guide_store_path: str = "travel_guide_store"
    travel_guide_data_path: str = "data"
    openai_api_key: str = "key"
    log_file: str = "trip.json"


@cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()
