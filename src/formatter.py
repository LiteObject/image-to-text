"""
Output formatting and export functionality for processed image text.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class MarkdownFormatter:
    """Formats extracted text and summaries into Markdown documents."""

    def __init__(self, include_metadata: bool = True, include_summaries: bool = True):
        """
        Initialize Markdown formatter.

        Args:
            include_metadata: Whether to include metadata in output
            include_summaries: Whether to include summaries in output
        """
        self.include_metadata = include_metadata
        self.include_summaries = include_summaries

    def format_single_image(self,
                            image_path: Path,
                            extracted_text: str,
                            summary: Optional[str],
                            metadata: Dict[str, Any]) -> str:
        """
        Format results for a single image.

        Args:
            image_path: Path to the processed image
            extracted_text: OCR extracted text
            summary: LLM generated summary
            metadata: Processing metadata

        Returns:
            Formatted Markdown string
        """
        markdown_content = []

        # Title
        markdown_content.append(f"# {image_path.name}")
        markdown_content.append("")

        # Metadata section
        if self.include_metadata and metadata:
            markdown_content.append("## Metadata")
            markdown_content.append("")
            markdown_content.append(
                f"- **File Path**: `{metadata.get('file_path', 'N/A')}`")
            markdown_content.append(
                f"- **Original Size**: {metadata.get('original_size', 'N/A')}")
            markdown_content.append(
                f"- **Processed Size**: {metadata.get('processed_size', 'N/A')}")
            markdown_content.append(
                f"- **Language**: {metadata.get('language', 'N/A')}")
            markdown_content.append(
                f"- **OCR Confidence**: {metadata.get('avg_confidence', 'N/A')}%")
            markdown_content.append(
                f"- **Word Count**: {metadata.get('word_count', 'N/A')}")
            markdown_content.append(
                f"- **Character Count**: {metadata.get('char_count', 'N/A')}")
            markdown_content.append(
                f"- **Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            markdown_content.append("")

        # Summary section
        if self.include_summaries and summary:
            markdown_content.append("## Summary")
            markdown_content.append("")
            markdown_content.append(summary)
            markdown_content.append("")

        # Extracted text section
        markdown_content.append("## Extracted Text")
        markdown_content.append("")
        if extracted_text.strip():
            markdown_content.append("```")
            markdown_content.append(extracted_text)
            markdown_content.append("```")
        else:
            markdown_content.append("*No text was extracted from this image.*")

        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")

        return "\n".join(markdown_content)

    def format_batch_results(self,
                             results: List[Dict[str, Any]],
                             _output_path: Path) -> str:
        """
        Format results for multiple images into a single document.

        Args:
            results: List of processing results
            _output_path: Path where the output will be saved (unused, for future use)

        Returns:
            Formatted Markdown string
        """
        markdown_content = []

        # Document header
        markdown_content.append("# Image Text Extraction Report")
        markdown_content.append("")
        markdown_content.append(
            f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append(f"**Total Images Processed**: {len(results)}")
        markdown_content.append("")

        # Table of contents
        markdown_content.append("## Table of Contents")
        markdown_content.append("")
        for i, result in enumerate(results, 1):
            image_name = Path(result['image_path']).name
            anchor = image_name.lower().replace(' ', '-').replace('.', '')
            markdown_content.append(f"{i}. [{image_name}](#{anchor})")
        markdown_content.append("")

        # Individual image results
        for result in results:
            image_path = Path(result['image_path'])
            single_result = self.format_single_image(
                image_path,
                result.get('extracted_text', ''),
                result.get('summary', ''),
                result.get('metadata', {})
            )
            markdown_content.append(single_result)

        # Document footer
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append(
            "*Report generated by Image-to-Text Processing System*")

        return "\n".join(markdown_content)


class OutputManager:
    """Manages output formatting and file operations."""

    def __init__(self, output_format: str = "markdown"):
        """
        Initialize output manager.

        Args:
            output_format: Output format type
        """
        self.output_format = output_format.lower()

        if self.output_format == "markdown":
            self.formatter = MarkdownFormatter(
                include_metadata=settings.include_metadata,
                include_summaries=settings.include_summaries
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def save_single_result(self,
                           result: Dict[str, Any],
                           output_dir: Path) -> Path:
        """
        Save result for a single image.

        Args:
            result: Processing result dictionary
            output_dir: Output directory

        Returns:
            Path to the saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(result['image_path'])
        output_filename = f"{image_path.stem}_extracted.md"
        output_path = output_dir / output_filename

        if self.output_format == "markdown":
            content = self.formatter.format_single_image(
                image_path,
                result.get('extracted_text', ''),
                result.get('summary', ''),
                result.get('metadata', {})
            )
        else:
            raise ValueError(
                f"Unsupported output format: {self.output_format}")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info("Saved result to: %s", output_path)
        return output_path

    def save_batch_results(self,
                           results: List[Dict[str, Any]],
                           output_path: Path) -> Path:
        """
        Save results for multiple images.

        Args:
            results: List of processing results
            output_path: Output file path

        Returns:
            Path to the saved file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_format == "markdown":
            content = self.formatter.format_batch_results(results, output_path)
        else:
            raise ValueError(
                f"Unsupported output format: {self.output_format}")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info("Saved batch results to: %s", output_path)
        return output_path

    def create_summary_index(self,
                             results: List[Dict[str, Any]],
                             output_dir: Path) -> Path:
        """
        Create an index file with summaries of all processed images.

        Args:
            results: List of processing results
            output_dir: Output directory

        Returns:
            Path to the index file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / "index.md"

        markdown_content = []

        # Header
        markdown_content.append("# Image Processing Summary Index")
        markdown_content.append("")
        markdown_content.append(
            f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append(f"**Total Images**: {len(results)}")
        markdown_content.append("")

        # Summary table
        markdown_content.append(
            "| Image | Word Count | Confidence | Summary |")
        markdown_content.append(
            "|-------|------------|------------|---------|")

        for result in results:
            image_name = Path(result['image_path']).name
            metadata = result.get('metadata', {})
            word_count = metadata.get('word_count', 0)
            confidence = metadata.get('avg_confidence', 0)
            summary = result.get('summary', 'No summary available')

            # Truncate summary for table
            summary_short = summary[:100] + \
                "..." if len(summary) > 100 else summary
            summary_short = summary_short.replace(
                '\n', ' ').replace('|', '\\|')

            markdown_content.append(
                f"| {image_name} | {word_count} | {confidence}% | {summary_short} |")

        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append(
            "*Generated by Image-to-Text Processing System*")

        content = "\n".join(markdown_content)

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info("Created summary index: %s", index_path)
        return index_path
