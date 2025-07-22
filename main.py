#!/usr/bin/env python3
"""
Main entry point for the image-to-text processing application.
"""
from src.cli import app
import sys
from pathlib import Path

# Add src directory to Python path FIRST
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import after path setup


if __name__ == "__main__":
    app()
