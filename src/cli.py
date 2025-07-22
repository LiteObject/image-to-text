"""
Command-line interface for the image-to-text processing application.
"""
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from src.config import settings
from src.formatter import OutputManager
from src.llm import LLMFactory
from src.ocr import OCRProcessor
from src.processor import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processor.log'),
        logging.StreamHandler()
    ]
)

app = typer.Typer(
    name="image-to-text",
    help="Extract text from images and generate summaries using AI.",
    add_completion=False
)
console = Console()


@app.command()
def process(
    input_path: Path = typer.Argument(
        ...,
        help="Path to image file or directory containing images",
        exists=True
    ),
    output_dir: Path = typer.Option(
        "./output",
        "--output", "-o",
        help="Output directory for results"
    ),
    llm_provider: str = typer.Option(
        settings.default_llm_provider,
        "--llm", "-l",
        help="LLM provider: 'openai' or 'ollama'"
    ),
    no_summaries: bool = typer.Option(
        False,
        "--no-summaries",
        help="Skip generating summaries"
    ),
    individual_files: bool = typer.Option(
        True,
        "--individual/--no-individual",
        help="Create individual files for each image"
    ),
    combined_file: bool = typer.Option(
        True,
        "--combined/--no-combined",
        help="Create combined file with all results"
    ),
    create_index: bool = typer.Option(
        True,
        "--index/--no-index",
        help="Create summary index file"
    ),
    batch_size: int = typer.Option(
        settings.batch_size,
        "--batch-size", "-b",
        help="Number of images to process concurrently"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """Process images to extract text and generate summaries."""

    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(Panel.fit(
        "🖼️ Image-to-Text Processing System",
        style="bold blue"
    ))

    try:
        # Initialize components
        with console.status("[bold green]Initializing components..."):
            llm = LLMFactory.create_provider(llm_provider)
            # Pass LLM for vision fallback
            ocr = OCRProcessor(llm_provider=llm)
            output_manager = OutputManager()
            processor = ImageProcessor(llm, ocr, output_manager)

        console.print(f"✅ Initialized with LLM provider: {llm_provider}")

        # Process images
        console.print(f"📁 Processing: {input_path}")
        console.print(f"📤 Output directory: {output_dir}")

        # Update batch size setting
        settings.batch_size = batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:

            task = progress.add_task("Processing images...", total=None)

            result = processor.process_folder(
                input_path,
                output_dir,
                generate_summaries=not no_summaries,
                create_individual_files=individual_files,
                create_combined_file=combined_file,
                create_index=create_index
            )

            progress.update(task, completed=result['total_images'])

        # Display results
        _display_results(result)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def single(
    image_path: Path = typer.Argument(
        ...,
        help="Path to a single image file",
        exists=True
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (default: <image_name>_extracted.md)"
    ),
    llm_provider: str = typer.Option(
        settings.default_llm_provider,
        "--llm", "-l",
        help="LLM provider: 'openai' or 'ollama'"
    ),
    no_summary: bool = typer.Option(
        False,
        "--no-summary",
        help="Skip generating summary"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """Process a single image file."""

    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(Panel.fit(
        "🖼️ Single Image Processing",
        style="bold blue"
    ))

    try:
        # Initialize components
        with console.status("[bold green]Initializing components..."):
            llm = LLMFactory.create_provider(llm_provider)
            # Pass LLM for vision fallback
            ocr = OCRProcessor(llm_provider=llm)
            output_manager = OutputManager()
            processor = ImageProcessor(llm, ocr, output_manager)

        console.print(f"✅ Initialized with LLM provider: {llm_provider}")

        # Process single image
        console.print(f"📷 Processing: {image_path.name}")

        with console.status("[bold green]Extracting text and generating summary..."):
            result = processor.process_single_image(
                image_path,
                generate_summary=not no_summary
            )

        if not result['success']:
            console.print("❌ Processing failed", style="bold red")
            console.print(
                f"Error: {result['metadata'].get('error', 'Unknown error')}")
            raise typer.Exit(1)

        # Save result
        if not output_file:
            output_dir = image_path.parent / "output"
            output_file = output_manager.save_single_result(result, output_dir)
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            content = output_manager.formatter.format_single_image(
                image_path,
                result['extracted_text'],
                result['summary'],
                result['metadata']
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

        # Display results
        _display_single_result(result, output_file)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def config():
    """Display current configuration."""

    console.print(Panel.fit("⚙️ Current Configuration", style="bold blue"))

    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value")

    # API Configuration
    config_table.add_row(
        "OpenAI API Key", "***" if settings.openai_api_key else "Not set")
    config_table.add_row("Ollama Base URL", settings.ollama_base_url)
    config_table.add_row("Ollama Model", settings.ollama_model)

    # Processing Configuration
    config_table.add_row("Default LLM Provider", settings.default_llm_provider)
    config_table.add_row("Max Image Size", str(settings.max_image_size))
    config_table.add_row("Batch Size", str(settings.batch_size))

    # OCR Configuration
    config_table.add_row("OCR Language", settings.ocr_language)
    config_table.add_row(
        "Tesseract Path", settings.tesseract_path or "Default")

    # Output Configuration
    config_table.add_row("Output Format", settings.output_format)
    config_table.add_row("Include Summaries", str(settings.include_summaries))
    config_table.add_row("Include Metadata", str(settings.include_metadata))

    console.print(config_table)


def _display_results(result: dict):
    """Display processing results in a formatted table."""

    console.print("\n📊 Processing Results", style="bold green")

    # Summary statistics
    stats_table = Table(show_header=False)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value")

    stats_table.add_row("Total Images", str(result['total_images']))
    stats_table.add_row("Successfully Processed", str(
        result['processed_successfully']))
    stats_table.add_row("Processing Time",
                        f"{result['processing_time']} seconds")
    stats_table.add_row("Output Files", str(len(result['output_files'])))

    console.print(stats_table)

    # Output files
    if result['output_files']:
        console.print("\n📄 Generated Files:", style="bold")
        for file_path in result['output_files']:
            console.print(f"  • {file_path}")

    # Failed images
    failed_results = [r for r in result.get('results', []) if not r['success']]
    if failed_results:
        console.print(
            f"\n⚠️ Failed to process {len(failed_results)} images:", style="bold yellow")
        for failed in failed_results:
            error_msg = failed['metadata'].get('error', 'Unknown error')
            console.print(
                f"  • {Path(failed['image_path']).name}: {error_msg}")


def _display_single_result(result: dict, output_file: Path):
    """Display single image processing result."""

    console.print("\n📊 Processing Result", style="bold green")

    metadata = result['metadata']

    # Result information
    info_table = Table(show_header=False)
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value")

    info_table.add_row("Image", Path(result['image_path']).name)
    info_table.add_row("Word Count", str(metadata.get('word_count', 0)))
    info_table.add_row("Character Count", str(metadata.get('char_count', 0)))
    info_table.add_row(
        "OCR Confidence", f"{metadata.get('avg_confidence', 0)}%")
    info_table.add_row("Processing Time",
                       f"{metadata.get('processing_time', 0)} seconds")
    info_table.add_row("Output File", str(output_file))

    console.print(info_table)

    # Show extracted text preview
    if result['extracted_text']:
        preview = result['extracted_text'][:200]
        if len(result['extracted_text']) > 200:
            preview += "..."

        console.print("\n📝 Text Preview:", style="bold")
        console.print(Panel(preview, title="Extracted Text"))

    # Show summary preview
    if result['summary']:
        console.print("\n📄 Summary:", style="bold")
        console.print(Panel(result['summary'], title="AI Summary"))


if __name__ == "__main__":
    app()
