"""
Diagnostic script to debug initialization issues with TensorFlow and detoxify.
This script attempts to load each dependency and provides detailed error information.
"""

import sys
import logging
import traceback
from importlib.metadata import version, PackageNotFoundError
import os

# Configure logging to show detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("initialization_debug")

def print_section(title):
    """Print a section header."""
    logger.info("=" * 50)
    logger.info(f" {title} ".center(50, "="))
    logger.info("=" * 50)

def check_package_version(package_name):
    """Check if a package is installed and get its version."""
    try:
        ver = version(package_name)
        logger.info(f"{package_name} is installed (version: {ver})")
        return True
    except PackageNotFoundError:
        logger.error(f"{package_name} is NOT installed")
        return False

def try_import_tensorflow():
    """Try to import TensorFlow and provide detailed error information."""
    print_section("TensorFlow Initialization")
    
    # Check if TensorFlow is installed
    tf_installed = check_package_version("tensorflow")
    hub_installed = check_package_version("tensorflow-hub")
    
    if not tf_installed or not hub_installed:
        logger.error("Missing required packages. Please install tensorflow and tensorflow-hub")
        return False
    
    # Try to import TensorFlow
    try:
        logger.info("Attempting to import TensorFlow...")
        import tensorflow as tf
        logger.info(f"TensorFlow successfully imported (version: {tf.__version__})")
        
        # Check for GPU availability
        logger.info("Checking GPU availability...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU is available: {gpus}")
            for gpu in gpus:
                logger.info(f"GPU details: {gpu}")
        else:
            logger.info("No GPU available, using CPU")
        
        # Try to import TensorFlow Hub
        logger.info("Attempting to import TensorFlow Hub...")
        import tensorflow_hub as hub
        logger.info("TensorFlow Hub successfully imported")
        
        # Try to load Universal Sentence Encoder
        logger.info("Attempting to load Universal Sentence Encoder...")
        try:
            logger.info("Loading model, this may take a moment...")
            model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
            logger.info("Universal Sentence Encoder loaded successfully")
            
            # Try a simple embedding
            logger.info("Testing model with a simple embedding...")
            embeddings = model(["Hello, world!"])
            logger.info(f"Embedding shape: {embeddings.shape}")
            logger.info("Model works correctly!")
            return True
        except Exception as e:
            logger.error(f"Failed to load Universal Sentence Encoder: {e}")
            logger.error(traceback.format_exc())
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import TensorFlow: {e}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"Unexpected error with TensorFlow: {e}")
        logger.error(traceback.format_exc())
        return False

def try_import_detoxify():
    """Try to import detoxify and provide detailed error information."""
    print_section("Detoxify Initialization")
    
    # Check if detoxify is installed
    detoxify_installed = check_package_version("detoxify")
    
    if not detoxify_installed:
        logger.error("Detoxify is not installed. Please install it with pip install detoxify")
        return False
    
    # Try to import detoxify
    try:
        logger.info("Attempting to import detoxify...")
        from detoxify import Detoxify
        logger.info("Detoxify successfully imported")
        
        # Try to initialize the model
        logger.info("Attempting to initialize detoxify model...")
        model = Detoxify('original')
        logger.info("Detoxify model initialized successfully")
        
        # Try a simple prediction
        logger.info("Testing model with a simple prediction...")
        test_text = "This is a test."
        results = model.predict(test_text)
        logger.info(f"Prediction results: {results}")
        logger.info("Model works correctly!")
        return True
    except ImportError as e:
        logger.error(f"Failed to import detoxify: {e}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"Unexpected error with detoxify: {e}")
        logger.error(traceback.format_exc())
        return False

def print_environment_info():
    """Print relevant environment information."""
    print_section("Environment Information")
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Check CUDA availability
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # Check relevant environment variables
    for var in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']:
        logger.info(f"{var}: {os.environ.get(var, 'Not Set')}")

def enhanced_debugging():
    """Run the enhanced debugging for TensorFlow and detoxify."""
    print_section("Starting Initialization Debugging")
    
    # Print environment information
    print_environment_info()
    
    # Test TensorFlow initialization
    tf_success = try_import_tensorflow()
    
    # Test detoxify initialization
    detoxify_success = try_import_detoxify()
    
    # Summary
    print_section("Debugging Summary")
    logger.info(f"TensorFlow initialization: {'SUCCESS' if tf_success else 'FAILED'}")
    logger.info(f"Detoxify initialization: {'SUCCESS' if detoxify_success else 'FAILED'}")
    
    if not tf_success or not detoxify_success:
        logger.info("""
Recommendations:
1. Make sure all required packages are installed: 
   pip install tensorflow tensorflow-hub detoxify
2. Check for version conflicts or incompatibilities
3. Verify GPU drivers and CUDA installation if using GPU
4. Check internet connectivity for downloading models
5. Ensure enough disk space for model downloads
""")
    else:
        logger.info("All dependencies initialized successfully!")

if __name__ == "__main__":
    enhanced_debugging() 