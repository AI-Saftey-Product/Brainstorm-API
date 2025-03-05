from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_VERSION: str = "v1"
    API_TITLE: str = "AI Safety Testing API"
    API_DESCRIPTION: str = "API for testing AI models for safety concerns"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str
    
    # OpenAI API (for testing NLP models)
    OPENAI_API_KEY: Optional[str] = None
    
    # Storage
    STORAGE_PATH: str = "./storage"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings() 