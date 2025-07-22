"""
Configuration management for the image-to-text application.
"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    openai_api_key: Optional[str] = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama2")

    # OCR Configuration
    tesseract_path: Optional[str] = Field(default=None)
    ocr_language: str = Field(default="eng")

    # Processing Configuration
    default_llm_provider: str = Field(default="openai")
    max_image_size: int = Field(default=2048)
    batch_size: int = Field(default=5)

    # Output Configuration
    output_format: str = Field(default="markdown")
    include_summaries: bool = Field(default=True)
    include_metadata: bool = Field(default=True)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()
