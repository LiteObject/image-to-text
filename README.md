# Image-to-Text Processing System

A modular Python application for extracting text from images and generating AI-powered summaries using LangChain with support for both OpenAI and local Ollama models.

## 🚀 Features

- **OCR Text Extraction**: Extract text from images using Tesseract OCR with preprocessing
- **AI-Powered Summaries**: Generate intelligent summaries using OpenAI or Ollama
- **Modular Architecture**: Extensible design with clean separation of concerns
- **Multiple Output Formats**: Export results in Markdown with rich formatting
- **Batch Processing**: Process multiple images concurrently with progress tracking
- **CLI Interface**: User-friendly command-line interface with rich output
- **Configuration Management**: Environment-based configuration with sensible defaults
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📋 Requirements

- Python 3.8+
- Tesseract OCR installed on your system
- OpenAI API key (for OpenAI provider) or Ollama installation (for local inference)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LiteObject/image-to-text.git
   cd image-to-text
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   
   **Windows:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH or set TESSERACT_PATH in .env
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:

```env
# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# OCR Configuration
TESSERACT_PATH=  # Optional: path to tesseract executable
OCR_LANGUAGE=eng

# Processing Configuration
DEFAULT_LLM_PROVIDER=openai  # or 'ollama'
MAX_IMAGE_SIZE=2048
BATCH_SIZE=5

# Output Configuration
OUTPUT_FORMAT=markdown
INCLUDE_SUMMARIES=true
INCLUDE_METADATA=true
```

## 🚀 Usage

### Command Line Interface

#### Process a Directory of Images
```bash
python main.py process /path/to/images --output ./output --llm openai
```

#### Process a Single Image
```bash
python main.py single /path/to/image.jpg --output result.md --llm ollama
```

#### View Configuration
```bash
python main.py config
```

### Command Options

- `--llm, -l`: Choose LLM provider (`openai` or `ollama`)
- `--output, -o`: Specify output directory or file
- `--no-summaries`: Skip generating AI summaries
- `--batch-size, -b`: Number of concurrent processes
- `--verbose, -v`: Enable detailed logging
- `--individual/--no-individual`: Create individual files per image
- `--combined/--no-combined`: Create combined results file
- `--index/--no-index`: Create summary index file

### Python API

```python
from src import ImageProcessor, LLMFactory

# Initialize components
llm_provider = LLMFactory.create_provider("openai")
processor = ImageProcessor(llm_provider=llm_provider)

# Process single image
result = processor.process_single_image(Path("image.jpg"))

# Process folder
summary = processor.process_folder(
    input_folder=Path("images/"),
    output_folder=Path("output/"),
    generate_summaries=True
)
```

## 🏗️ Architecture

The application follows a modular architecture with clear separation of concerns:

```
src/
├── config.py          # Configuration management
├── ocr.py             # OCR text extraction
├── llm.py             # LLM provider abstraction
├── formatter.py       # Output formatting
├── processor.py       # Main processing orchestration
├── cli.py             # Command-line interface
└── __init__.py        # Package initialization
```

### Key Components

- **OCRProcessor**: Handles image preprocessing and text extraction using Tesseract
- **LLMProvider**: Abstract interface for different LLM providers (OpenAI, Ollama)
- **OutputManager**: Formats and saves results in various formats
- **ImageProcessor**: Orchestrates the entire workflow
- **CLI**: User-friendly command-line interface

## 📁 Output Structure

The application generates organized output with multiple file types:

```
output/
├── index.md                    # Summary index of all images
├── all_results.md             # Combined results file
├── image1_extracted.md        # Individual result files
├── image2_extracted.md
└── ...
```

### Output Format

Each result includes:
- **Metadata**: File info, processing stats, OCR confidence
- **Summary**: AI-generated summary (if enabled)
- **Extracted Text**: Raw OCR output

## 🔧 Extending the System

### Adding New LLM Providers

Implement the `BaseLLMProvider` interface:

```python
from src.llm import BaseLLMProvider

class CustomProvider(BaseLLMProvider):
    def summarize_text(self, text: str, context: str = None) -> str:
        # Implement summarization logic
        pass
    
    def process_text(self, text: str, instruction: str) -> str:
        # Implement custom text processing
        pass
```

### Adding New Output Formats

Extend the `OutputManager` class:

```python
from src.formatter import OutputManager

class CustomOutputManager(OutputManager):
    def __init__(self):
        super().__init__()
        self.custom_formatter = CustomFormatter()
```

## 🧪 Testing

Run the example script to test functionality:

```bash
python examples.py
```

This will demonstrate:
- Single image processing
- Batch processing
- Custom text processing
- Different LLM providers

## 📝 Logging

The application provides comprehensive logging:

- Console output with progress indicators
- File logging to `image_processor.log`
- Configurable log levels (INFO, DEBUG)
- Rich formatting for better readability

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Ensure Tesseract is installed and in PATH
   - Set TESSERACT_PATH in .env file

2. **OpenAI API errors**
   - Verify API key is correct
   - Check API quota and billing

3. **Ollama connection issues**
   - Ensure Ollama is running locally
   - Verify OLLAMA_BASE_URL in configuration

4. **Memory issues with large images**
   - Reduce MAX_IMAGE_SIZE setting
   - Process images in smaller batches

### Performance Tips

- Adjust BATCH_SIZE based on system resources
- Use appropriate MAX_IMAGE_SIZE for your needs
- Consider using Ollama for privacy-sensitive content
- Enable verbose logging for debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
- [LangChain](https://github.com/langchain-ai/langchain) for LLM abstraction
- [OpenAI](https://openai.com/) for GPT models
- [Ollama](https://ollama.ai/) for local LLM inference
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output