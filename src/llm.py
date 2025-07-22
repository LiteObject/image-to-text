"""
LLM abstraction layer using LangChain for text summarization and processing.
"""
import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

from src.config import settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def summarize_text(self, text: str, context: Optional[str] = None) -> str:
        """Summarize the given text."""

    @abstractmethod
    def process_text(self, text: str, instruction: str) -> str:
        """Process text with custom instructions."""

    def extract_text_from_image(self, image_path: Path) -> tuple[str, dict]:
        """
        Extract text from an image using vision capabilities.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        # Default implementation returns empty - override in vision-capable providers
        logger.warning(
            "Vision-based text extraction not supported by this provider")
        return "", {
            'file_path': str(image_path),
            'method': 'llm_vision',
            'provider': self.__class__.__name__,
            'error': 'Vision not supported'
        }


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using LangChain."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: Model name to use (defaults to gpt-4o for vision support)
        """
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        from pydantic import SecretStr

        self.model_name = model_name
        self.supports_vision = model_name in [
            "gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"]

        self.llm = ChatOpenAI(
            api_key=SecretStr(self.api_key),
            model=model_name,
            temperature=0.3
        )

        logger.info("OpenAI provider initialized with model: %s (vision: %s)",
                    model_name, self.supports_vision)

    def summarize_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Summarize text using OpenAI.

        Args:
            text: Text to summarize
            context: Additional context for summarization

        Returns:
            Summarized text
        """
        prompt_template = PromptTemplate(
            input_variables=["text", "context_part"],
            template="""
            Please provide a concise summary of the following text extracted from an image.
            Focus on the key information and main points.
            
            {context_part}
            
            Text to summarize:
            {text}
            
            Summary:
            """
        )

        context_part = f"Additional context: {context}" if context else ""

        # Create LCEL chain
        chain = (
            {"text": RunnablePassthrough(), "context_part": lambda x: context_part}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke(text)
            logger.info("Successfully generated summary using OpenAI")
            return result.strip()
        except Exception as e:
            logger.error("Error generating summary with OpenAI: %s", e)
            raise

    def process_text(self, text: str, instruction: str) -> str:
        """
        Process text with custom instructions using OpenAI.

        Args:
            text: Text to process
            instruction: Processing instruction

        Returns:
            Processed text
        """
        prompt_template = PromptTemplate(
            input_variables=["text", "instruction"],
            template="""
            {instruction}
            
            Text to process:
            {text}
            
            Result:
            """
        )

        # Create LCEL chain
        chain = (
            {"text": RunnablePassthrough(), "instruction": lambda x: instruction}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke(text)
            return result.strip()
        except Exception as e:
            logger.error("Error processing text with OpenAI: %s", e)
            raise

    def extract_text_from_image(self, image_path: Path) -> tuple[str, dict]:
        """
        Extract text from an image using OpenAI vision models.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not self.supports_vision:
            logger.warning("Model %s does not support vision. Use gpt-4o or gpt-4-vision-preview",
                           self.model_name)
            return super().extract_text_from_image(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode('utf-8')

            # Determine image format
            image_format = image_path.suffix.lower().lstrip('.')
            if image_format == 'jpg':
                image_format = 'jpeg'

            # Create vision message
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": """Please extract all text content from this image. 
                        Provide the text exactly as it appears, maintaining formatting where possible.
                        If there is no text in the image, respond with "No text detected"."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_data}"
                        }
                    }
                ]
            )

            # Get response from vision model
            response = self.llm.invoke([message])
            extracted_text = str(response.content).strip()

            # Prepare metadata
            metadata = {
                'file_path': str(image_path),
                'method': 'llm_vision',
                'provider': 'OpenAI',
                'model': self.model_name,
                'image_format': image_format,
                'word_count': len(extracted_text.split()) if extracted_text != "No text detected" else 0,
                'char_count': len(extracted_text) if extracted_text != "No text detected" else 0
            }

            logger.info(
                "Successfully extracted text from %s using OpenAI vision", image_path.name)
            return extracted_text, metadata

        except Exception as e:
            logger.error(
                "Error extracting text from image %s: %s", image_path, e)
            raise RuntimeError(
                f"Vision-based text extraction failed: {e}") from e


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local inference."""

    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama server URL
            model_name: Model name to use
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model_name = model_name or settings.ollama_model

        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url
            )
            logger.info(
                "Ollama provider initialized with model: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize Ollama provider: %s", e)
            raise

    def summarize_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Summarize text using Ollama.

        Args:
            text: Text to summarize
            context: Additional context for summarization

        Returns:
            Summarized text
        """
        prompt_template = PromptTemplate(
            input_variables=["text", "context_part"],
            template="""
            Please provide a concise summary of the following text extracted from an image.
            Focus on the key information and main points.
            
            {context_part}
            
            Text to summarize:
            {text}
            
            Summary:
            """
        )

        context_part = f"Additional context: {context}" if context else ""

        # Create LCEL chain
        chain = (
            {"text": RunnablePassthrough(), "context_part": lambda x: context_part}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke(text)
            logger.info("Successfully generated summary using Ollama")
            return result.strip()
        except Exception as e:
            logger.error("Error generating summary with Ollama: %s", e)
            raise

    def process_text(self, text: str, instruction: str) -> str:
        """
        Process text with custom instructions using Ollama.

        Args:
            text: Text to process
            instruction: Processing instruction

        Returns:
            Processed text
        """
        prompt_template = PromptTemplate(
            input_variables=["text", "instruction"],
            template="""
            {instruction}
            
            Text to process:
            {text}
            
            Result:
            """
        )

        # Create LCEL chain
        chain = (
            {"text": RunnablePassthrough(), "instruction": lambda x: instruction}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke(text)
            return result.strip()
        except Exception as e:
            logger.error("Error processing text with Ollama: %s", e)
            raise


class LLMFactory:
    """Factory class for creating LLM providers."""

    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Type of provider ('openai' or 'ollama')
            **kwargs: Additional arguments for provider initialization

        Returns:
            LLM provider instance
        """
        provider_type = provider_type.lower()

        if provider_type == "openai":
            return OpenAIProvider(**kwargs)
        if provider_type == "ollama":
            return OllamaProvider(**kwargs)
        raise ValueError(f"Unsupported provider type: {provider_type}")

    @staticmethod
    def get_default_provider() -> BaseLLMProvider:
        """Get the default LLM provider based on configuration."""
        return LLMFactory.create_provider(settings.default_llm_provider)
