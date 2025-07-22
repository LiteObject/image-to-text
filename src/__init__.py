"""
Image-to-Text Processing System

A modular Python application for extracting text from images and generating 
AI-powered summaries using LangChain with support for OpenAI and Ollama.
"""

__version__ = "1.0.0"
__author__ = "LiteObject"
__description__ = "Extract text from images and generate AI summaries"

from .config import settings
from .ocr import OCRProcessor
from .llm import LLMFactory, BaseLLMProvider, OpenAIProvider, OllamaProvider
from .formatter import MarkdownFormatter, OutputManager
from .processor import ImageProcessor

__all__ = [
    'settings',
    'OCRProcessor',
    'LLMFactory',
    'BaseLLMProvider',
    'OpenAIProvider',
    'OllamaProvider',
    'MarkdownFormatter',
    'OutputManager',
    'ImageProcessor',
]
