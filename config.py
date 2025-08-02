"""
Configuration settings for the LLM Document Retrieval API
"""
import os
from typing import Dict, Any

class Config:
    # Authentication
    AUTHORIZED_TOKEN = "479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba"

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCyLEILSjE96HexvyxwFw_S-aEvz8GQ3N")

    # Document processing
    MAX_CHUNK_SIZE = 1000
    OVERLAP_SIZE = 100
    MAX_CHUNKS_PER_QUERY = 5
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    # Timeout settings
    DOWNLOAD_TIMEOUT = 30
    API_TIMEOUT = 60

    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Rate limiting
    RATE_LIMIT_PER_MINUTE = 60

    # Supported file types
    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".eml", ".msg"]
    SUPPORTED_MIME_TYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "message/rfc822",
        "application/vnd.ms-outlook"
    ]

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"

class TestingConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    TESTING = True

# Configuration mapping
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

def get_config() -> Config:
    env = os.getenv("ENVIRONMENT", "production")
    return config_map.get(env, ProductionConfig)
