from pydantic_settings import BaseSettings, SettingsConfigDict
from root_folders import ENV_FILE

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, 
        env_file_encoding="utf-8",
        case_sensitive=False  
    )

    openai_api_key: str
    mistral_api_key: str
    collection_name: str
    embedding_model_openai: str = "text-embedding-3-small"
    embedding_model_size_openai: int = 1536
    embedding_model_sentence_transformer: str = "google/embeddinggemma-300m"
    embedding_model_size_sentence_transformer: int = 768


settings = Settings()