from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings) :
  model_config = SettingsConfigDict(env_file = ".env", env_file_encoding="utf-8")
  gemini_api_key: str
  qdrant_url: str = "http://localhost:6333"
  postgres_url: str



settings = Settings()