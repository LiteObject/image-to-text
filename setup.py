"""
Setup script for the image-to-text processing application.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Extract text from images and generate AI-powered summaries"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip()
                        and not line.startswith("#")]
else:
    requirements = []

setup(
    name="image-to-text",
    version="1.0.0",
    author="LiteObject",
    description="Extract text from images and generate AI-powered summaries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LiteObject/image-to-text",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "image-to-text=src.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.env.example"],
    },
)
