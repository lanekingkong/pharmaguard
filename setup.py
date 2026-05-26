#!/usr/bin/env python
"""
PharmaGuard Setup
"""

from setuptools import setup, find_packages
import os

# Read version
with open("src/pharmaguard/__init__.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
def parse_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "torch-geometric>=2.4.0",
    "transformers>=4.35.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "shap>=0.42.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.17.0",
    "rdkit>=2023.03.0",
    "neo4j>=5.14.0",
    "py2neo>=2021.2.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.5.0",
    "streamlit>=1.28.0",
    "click>=8.1.0",
    "rich>=13.7.0",
    "httpx>=0.25.0",
    "requests>=2.31.0",
    "tqdm>=4.66.0",
    "joblib>=1.3.0",
    "python-multipart>=0.0.6",
    "aiohttp>=3.9.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.23.0",
        "black>=23.11.0",
        "isort>=5.12.0",
        "mypy>=1.7.0",
        "ruff>=0.1.0",
        "pre-commit>=3.5.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.1.0",
    ],
    "gpu": [
        "torch-cuda>=2.0.0",
    ],
    "dashboard": [
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "dash>=2.14.0",
    ],
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=2.0.0",
        "myst-parser>=2.0.0",
        "nbsphinx>=0.9.0",
    ],
    "all": [
        "torch-cuda>=2.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "dash>=2.14.0",
        "pytest>=7.4.0",
        "black>=23.11.0",
        "isort>=5.12.0",
        "mypy>=1.7.0",
        "ruff>=0.1.0",
        "pre-commit>=3.5.0",
    ],
}

setup(
    name="pharmaguard",
    version=version,
    author="PharmaGuard Team",
    author_email="team@pharmaguard.ai",
    description="PharmaGuard: Intelligent Medication Safety & DDI Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pharmaguard/pharmaguard",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "pharmaguard=pharmaguard.cli:cli",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/pharmaguard/pharmaguard",
        "Documentation": "https://pharmaguard.readthedocs.io",
        "Repository": "https://github.com/pharmaguard/pharmaguard.git",
        "Issues": "https://github.com/pharmaguard/pharmaguard/issues",
    },
)