"""
Dependency installer for the Brainstorm API Test suite.
This script installs all required dependencies for advanced NLP testing.
"""
import subprocess
import sys
import importlib
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dependency_installer")

# List of required packages with their import names
DEPENDENCIES = [
    ("sentence-transformers", "sentence_transformers", "SentenceTransformer for semantic evaluation"),
    ("detoxify", "detoxify", "Detoxify for toxicity evaluation"),
    ("tensorflow", "tensorflow", "TensorFlow for USE evaluation"),
    ("tensorflow-hub", "tensorflow_hub", "TF Hub for USE evaluation"),
    ("bert-score", "bert_score", "BERTScore for evaluation"),
    ("datasets", "datasets", "HuggingFace datasets"),
    ("transformers", "transformers", "HuggingFace transformers"),
    ("nltk", "nltk", "NLTK for NLP utilities"),
]

def check_dependency(import_name: str) -> bool:
    """Check if a dependency is installed."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name: str) -> bool:
    """Install a package using pip."""
    logger.info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def main():
    """Main function to install dependencies."""
    logger.info("Starting dependency installation...")
    
    # Check and install each dependency
    missing_deps = []
    for package_name, import_name, description in DEPENDENCIES:
        if not check_dependency(import_name):
            logger.info(f"Missing: {description} ({package_name})")
            missing_deps.append((package_name, import_name, description))
        else:
            logger.info(f"Already installed: {description} ({package_name})")
    
    # Install missing dependencies
    if missing_deps:
        logger.info(f"Installing {len(missing_deps)} missing dependencies...")
        for package_name, import_name, description in missing_deps:
            if install_package(package_name):
                logger.info(f"Successfully installed {package_name}")
            else:
                logger.error(f"Failed to install {package_name}")
    else:
        logger.info("All dependencies are already installed.")
    
    # Download NLTK data if needed
    try:
        import nltk
        nltk.download('wordnet')
        logger.info("Downloaded NLTK wordnet data")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
    
    logger.info("Dependency installation completed.")

if __name__ == "__main__":
    main() 