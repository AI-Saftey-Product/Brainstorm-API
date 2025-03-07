"""
Debug runner script for adversarial testing components.
This script will run the diagnostic tests and check the enhanced initialization code.
"""

import asyncio
import logging
import sys
import traceback
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_initialization.log')
    ]
)

logger = logging.getLogger("debug_runner")

# Adjust path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# Also add the project root to path
project_root = str(Path(current_dir).parent.parent.parent.parent)
sys.path.insert(0, project_root)

async def test_universal_sentence_encoder():
    """Test the Universal Sentence Encoder initialization."""
    logger.info("=" * 50)
    logger.info(" Testing Universal Sentence Encoder ".center(50, "="))
    logger.info("=" * 50)
    
    try:
        # Direct import from the same directory
        from advanced_attacks import UniversalSentenceEncoder
        
        # Create an instance (this will trigger initialization)
        logger.info("Creating UniversalSentenceEncoder instance...")
        use = UniversalSentenceEncoder()
        
        # Check if initialization was successful
        logger.info(f"Initialization successful: {use.is_initialized}")
        
        if use.is_initialized:
            # Test with some text
            text1 = "This is a test sentence."
            text2 = "This is another test sentence."
            similarity = use.similarity(text1, text2)
            logger.info(f"Similarity between texts: {similarity}")
        else:
            logger.error("Universal Sentence Encoder failed to initialize.")
            
    except Exception as e:
        logger.error(f"Error testing Universal Sentence Encoder: {e}")
        logger.error(traceback.format_exc())

async def test_toxicity_evaluator():
    """Test the Toxicity Evaluator initialization."""
    logger.info("=" * 50)
    logger.info(" Testing Toxicity Evaluator ".center(50, "="))
    logger.info("=" * 50)
    
    try:
        # Direct import from the same directory
        from advanced_attacks import ToxicityEvaluator
        
        # Create an instance (this will trigger initialization)
        logger.info("Creating ToxicityEvaluator instance...")
        evaluator = ToxicityEvaluator()
        
        # Check if initialization was successful
        logger.info(f"Initialization successful: {evaluator.is_initialized}")
        
        if evaluator.is_initialized:
            # Test with some text
            text = "This is a neutral test sentence."
            toxicity = evaluator.evaluate(text)
            logger.info(f"Toxicity scores: {toxicity}")
        else:
            logger.error("Toxicity Evaluator failed to initialize.")
            
    except Exception as e:
        logger.error(f"Error testing Toxicity Evaluator: {e}")
        logger.error(traceback.format_exc())

async def test_attack_initialization():
    """Test the initialization of adversarial attacks."""
    logger.info("=" * 50)
    logger.info(" Testing Adversarial Attacks Initialization ".center(50, "="))
    logger.info("=" * 50)
    
    try:
        # Direct import from the same directory
        from advanced_attacks import (
            TextFoolerAttack, BERTAttack, NeuralParaphraseAttack, RedTeamAttack
        )
        
        # Test TextFoolerAttack
        logger.info("Initializing TextFoolerAttack...")
        textfooler = TextFoolerAttack()
        logger.info(f"TextFoolerAttack initialized: {getattr(textfooler, 'initialized', False)}")
        
        # Test BERTAttack
        logger.info("Initializing BERTAttack...")
        bert_attack = BERTAttack()
        logger.info(f"BERTAttack initialized: {getattr(bert_attack, 'is_initialized', False)}")
        
        # Test NeuralParaphraseAttack
        logger.info("Initializing NeuralParaphraseAttack...")
        paraphrase = NeuralParaphraseAttack()
        logger.info(f"NeuralParaphraseAttack initialized: {getattr(paraphrase, 'is_initialized', False)}")
        
        # Test RedTeamAttack
        logger.info("Initializing RedTeamAttack...")
        redteam = RedTeamAttack()
        logger.info(f"RedTeamAttack initialized: {getattr(redteam, 'is_initialized', False)}")
        
    except Exception as e:
        logger.error(f"Error testing adversarial attacks: {e}")
        logger.error(traceback.format_exc())

async def test_standalone_imports():
    """Test standalone imports of TensorFlow and detoxify."""
    logger.info("=" * 50)
    logger.info(" Testing Standalone Imports ".center(50, "="))
    logger.info("=" * 50)
    
    # Test TensorFlow import
    try:
        logger.info("Attempting to import tensorflow...")
        import tensorflow as tf
        logger.info(f"TensorFlow imported successfully (version: {tf.__version__})")
        
        # Test if TensorFlow can be used
        logger.info("Testing basic TensorFlow operations...")
        tensor = tf.constant([1, 2, 3])
        logger.info(f"Tensor created: {tensor}")
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU is available: {gpus}")
        else:
            logger.info("No GPU available, using CPU")
    except ImportError as e:
        logger.error(f"Failed to import tensorflow: {e}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error with tensorflow: {e}")
        logger.error(traceback.format_exc())
    
    # Test detoxify import
    try:
        logger.info("Attempting to import detoxify...")
        from detoxify import Detoxify
        logger.info("detoxify imported successfully")
        
        # Test if detoxify can be initialized
        logger.info("Testing detoxify initialization...")
        model = Detoxify('original')
        logger.info("detoxify model initialized successfully")
        
        # Test if detoxify can be used
        logger.info("Testing detoxify prediction...")
        result = model.predict("This is a test")
        logger.info(f"detoxify prediction successful: {result}")
    except ImportError as e:
        logger.error(f"Failed to import detoxify: {e}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error with detoxify: {e}")
        logger.error(traceback.format_exc())

async def main():
    """Run all debug tests."""
    logger.info("=" * 50)
    logger.info(" Starting Debug Tests ".center(50, "="))
    logger.info("=" * 50)
    
    # Print environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Script directory: {current_dir}")
    logger.info(f"Python path: {sys.path}")
    
    # Run standalone import tests
    await test_standalone_imports()
    
    # Test USE
    await test_universal_sentence_encoder()
    
    # Test Toxicity Evaluator
    await test_toxicity_evaluator()
    
    # Test attack initialization
    await test_attack_initialization()
    
    logger.info("=" * 50)
    logger.info(" Debug Tests Complete ".center(50, "="))
    logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 