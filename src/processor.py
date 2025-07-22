"""
Main processor for image-to-text extraction and summarization.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.config import settings
from src.ocr import OCRProcessor
from src.llm import LLMFactory, BaseLLMProvider
from src.formatter import OutputManager

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Main processor for handling image-to-text extraction workflow."""

    def __init__(self,
                 llm_provider: Optional[BaseLLMProvider] = None,
                 ocr_processor: Optional[OCRProcessor] = None,
                 output_manager: Optional[OutputManager] = None):
        """
        Initialize the image processor.

        Args:
            llm_provider: LLM provider instance
            ocr_processor: OCR processor instance
            output_manager: Output manager instance
        """
        self.llm_provider = llm_provider or LLMFactory.get_default_provider()
        self.ocr_processor = ocr_processor or OCRProcessor()
        self.output_manager = output_manager or OutputManager()

        logger.info("ImageProcessor initialized successfully")

    def find_images(self, input_path: Path) -> List[Path]:
        """
        Find all supported image files in the given path.

        Args:
            input_path: Path to search for images (file or directory)

        Returns:
            List of image file paths
        """
        image_files = []

        if input_path.is_file():
            if self.ocr_processor.is_supported_format(input_path):
                image_files.append(input_path)
            else:
                logger.warning("Unsupported file format: %s", input_path)
        elif input_path.is_dir():
            # Recursively find all supported image files
            for file_path in input_path.rglob('*'):
                if file_path.is_file() and self.ocr_processor.is_supported_format(file_path):
                    image_files.append(file_path)
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        logger.info("Found %d image files to process", len(image_files))
        return sorted(image_files)

    def process_single_image(self,
                             image_path: Path,
                             generate_summary: bool = True) -> Dict[str, Any]:
        """
        Process a single image: extract text and generate summary.

        Args:
            image_path: Path to the image file
            generate_summary: Whether to generate LLM summary

        Returns:
            Processing result dictionary
        """
        logger.info("Processing image: %s", image_path.name)
        start_time = time.time()

        try:
            # Extract text using OCR
            extracted_text, metadata = self.ocr_processor.extract_text(
                image_path)

            # Generate summary if requested and text is available
            summary = None
            if generate_summary and extracted_text.strip():
                try:
                    summary = self.llm_provider.summarize_text(
                        extracted_text,
                        context=f"Text extracted from image: {image_path.name}"
                    )
                except (ConnectionError, TimeoutError) as e:
                    logger.error(
                        "Network error generating summary for %s: %s", image_path.name, e)
                    summary = "Summary generation failed: Network error"
                except (ValueError, AttributeError, TypeError) as e:
                    logger.error(
                        "LLM error generating summary for %s: %s", image_path.name, e)
                    summary = "Summary generation failed: LLM error"
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Unexpected error generating summary for %s: %s", image_path.name, e)
                    summary = "Summary generation failed: Unexpected error"

            # Calculate processing time
            processing_time = time.time() - start_time
            metadata['processing_time'] = round(processing_time, 2)

            result = {
                'image_path': str(image_path),
                'extracted_text': extracted_text,
                'summary': summary,
                'metadata': metadata,
                'success': True
            }

            logger.info("Successfully processed %s in %.2f seconds",
                        image_path.name, processing_time)
            return result

        except (FileNotFoundError, PermissionError) as e:
            logger.error("File access error processing %s: %s",
                         image_path.name, e)
            return {
                'image_path': str(image_path),
                'extracted_text': '',
                'summary': None,
                'metadata': {'error': f"File access error: {e}"},
                'success': False
            }
        except (OSError, IOError) as e:
            logger.error("I/O error processing %s: %s", image_path.name, e)
            return {
                'image_path': str(image_path),
                'extracted_text': '',
                'summary': None,
                'metadata': {'error': f"I/O error: {e}"},
                'success': False
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Unexpected error processing %s: %s",
                         image_path.name, e)
            return {
                'image_path': str(image_path),
                'extracted_text': '',
                'summary': None,
                'metadata': {'error': f"Unexpected error: {e}"},
                'success': False
            }

    def process_batch(self,
                      image_paths: List[Path],
                      generate_summaries: bool = True,
                      max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple images concurrently.

        Args:
            image_paths: List of image file paths
            generate_summaries: Whether to generate LLM summaries
            max_workers: Maximum number of worker threads

        Returns:
            List of processing results
        """
        max_workers = max_workers or min(len(image_paths), settings.batch_size)
        results = []

        logger.info("Starting batch processing of %d images with %d workers",
                    len(image_paths), max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, path, generate_summaries): path
                for path in image_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except (FileNotFoundError, PermissionError) as e:
                    logger.error(
                        "File access error processing %s: %s", path, e)
                    results.append({
                        'image_path': str(path),
                        'extracted_text': '',
                        'summary': None,
                        'metadata': {'error': f"File access error: {e}"},
                        'success': False
                    })
                except (OSError, IOError) as e:
                    logger.error("I/O error processing %s: %s", path, e)
                    results.append({
                        'image_path': str(path),
                        'extracted_text': '',
                        'summary': None,
                        'metadata': {'error': f"I/O error: {e}"},
                        'success': False
                    })
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Unexpected error processing %s: %s", path, e)
                    results.append({
                        'image_path': str(path),
                        'extracted_text': '',
                        'summary': None,
                        'metadata': {'error': f"Unexpected error: {e}"},
                        'success': False
                    })

        # Sort results by image path for consistent output
        results.sort(key=lambda x: x['image_path'])

        successful = sum(1 for r in results if r['success'])
        logger.info("Batch processing completed: %d/%d images processed successfully",
                    successful, len(results))

        return results

    def process_folder(self,
                       input_folder: Path,
                       output_folder: Path,
                       generate_summaries: bool = True,
                       create_individual_files: bool = True,
                       create_combined_file: bool = True,
                       create_index: bool = True) -> Dict[str, Any]:
        """
        Process all images in a folder and save results.

        Args:
            input_folder: Folder containing images to process
            output_folder: Folder to save results
            generate_summaries: Whether to generate LLM summaries
            create_individual_files: Create separate file for each image
            create_combined_file: Create single file with all results
            create_index: Create summary index file

        Returns:
            Processing summary
        """
        start_time = time.time()

        # Find all images
        image_paths = self.find_images(input_folder)

        if not image_paths:
            logger.warning("No images found in: %s", input_folder)
            return {
                'total_images': 0,
                'processed_successfully': 0,
                'processing_time': 0,
                'output_files': []
            }

        # Process all images
        results = self.process_batch(image_paths, generate_summaries)

        # Save results
        output_files = []

        # Individual files
        if create_individual_files:
            for result in results:
                if result['success']:
                    try:
                        output_path = self.output_manager.save_single_result(
                            result, output_folder
                        )
                        output_files.append(str(output_path))
                    except (OSError, PermissionError) as e:
                        logger.error(
                            "File system error saving individual result: %s", e)
                    except (ValueError, TypeError) as e:
                        logger.error(
                            "Data error saving individual result: %s", e)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.error(
                            "Unexpected error saving individual result: %s", e)

        # Combined file
        if create_combined_file:
            try:
                combined_path = output_folder / "all_results.md"
                output_path = self.output_manager.save_batch_results(
                    results, combined_path
                )
                output_files.append(str(output_path))
            except (OSError, PermissionError) as e:
                logger.error(
                    "File system error saving combined results: %s", e)
            except (ValueError, TypeError) as e:
                logger.error("Data error saving combined results: %s", e)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Unexpected error saving combined results: %s", e)

        # Index file
        if create_index:
            try:
                index_path = self.output_manager.create_summary_index(
                    results, output_folder
                )
                output_files.append(str(index_path))
            except (OSError, PermissionError) as e:
                logger.error("File system error creating index: %s", e)
            except (ValueError, TypeError) as e:
                logger.error("Data error creating index: %s", e)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Unexpected error creating index: %s", e)

        # Calculate summary statistics
        successful = sum(1 for r in results if r['success'])
        total_time = time.time() - start_time

        summary = {
            'total_images': len(results),
            'processed_successfully': successful,
            'processing_time': round(total_time, 2),
            'output_files': output_files,
            'results': results
        }

        logger.info("Folder processing completed in %.2f seconds: %d/%d images successful",
                    total_time, successful, len(results))

        return summary

    def custom_text_processing(self,
                               text: str,
                               instruction: str) -> str:
        """
        Process text with custom instructions using the LLM.

        Args:
            text: Text to process
            instruction: Custom processing instruction

        Returns:
            Processed text
        """
        return self.llm_provider.process_text(text, instruction)
