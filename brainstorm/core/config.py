import os

import sqlalchemy
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_VERSION: str = "v1"
    API_TITLE: str = "AI Safety Testing API"
    API_DESCRIPTION: str = "API for testing AI models for safety concerns"
    DEBUG: bool = False
    
    # Project settings
    PROJECT_NAME: str = "AI Safety Testing API"
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",  # React default port
        "http://localhost:3001",  # React default port
        "http://localhost:8000",  # FastAPI default port
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
        "https://hirundo-trial.uc.r.appspot.com",
    ]
    
    # Security
    SECRET_KEY: str = "development_key_not_for_production_use"  # Default for development
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database - default to in-memory SQLite for development
    DATABASE_URL: str = "sqlite:///:memory:"
    # Set to True to disable database features
    DISABLE_DATABASE: bool = True
    
    # OpenAI API (for testing NLP models)
    OPENAI_API_KEY: Optional[str] = None
    
    # Storage
    STORAGE_PATH: str = "./storage"

    if os.environ.get('LOCAL_DOCKER'):
        POSTGRES_HOST: str = "postgresql://user:password@localhost:5432/brainstorm_test_api_local"
    elif os.environ.get('CLOUD_VIA_PROXY'):
        POSTGRES_HOST: str = "postgresql://backend_user:LPTd*6AeOy)1m8Al@127.0.0.1:5633/backend_db"
    else:
        # POSTGRES_HOST: str = "postgresql://user:password@/brainstorm_test_api_local"
        unix_socket_path = '/cloudsql/hirundo-trial:us-central1:main-db'
        POSTGRES_HOST: str = sqlalchemy.engine.url.URL.create(
            drivername="postgresql+pg8000",
            username='backend_user',
            password='LPTd*6AeOy)1m8Al',
            database='backend_db',
            query={"unix_sock": f"{unix_socket_path}/.s.PGSQL.5432"},
        ),

    DATA_BUCKET: str = "eval_datasets_store"
    
    # Don't load from .env files
    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"


settings = Settings()