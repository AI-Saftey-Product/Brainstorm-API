#!/usr/bin/env python
"""
Helper script to install the correct version of huggingface_hub
"""

import sys
import subprocess
import os

def install_requirements():
    print("Installing required packages for HuggingFace integration...")
    
    packages = [
        "huggingface_hub==0.17.3",
        "transformers",
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("\nAll packages installed successfully!")
    return True

def test_imports():
    print("\nTesting imports...")
    try:
        from huggingface_hub import InferenceClient
        print("✓ Successfully imported huggingface_hub.InferenceClient")
        
        # Try creating a client (without connecting)
        client = InferenceClient(api_key="test")
        print("✓ Successfully created InferenceClient instance")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("HuggingFace Integration Setup")
    print("=" * 60)
    
    # Ask for API key if not in environment
    if "HUGGINGFACE_API_KEY" not in os.environ:
        print("\nNote: To use the HuggingFace API, you need an API key.")
        print("You can set it as an environment variable HUGGINGFACE_API_KEY.")
        print("Or you'll be prompted for it when running tests.")
    
    # Install requirements
    if install_requirements():
        # Test imports
        if test_imports():
            print("\n✅ Setup complete! You can now use the HuggingFace integration.")
        else:
            print("\n❌ Import test failed. Please check your Python environment.")
    else:
        print("\n❌ Installation failed. Please try installing packages manually:")
        print("pip install huggingface_hub==0.17.3 transformers")
    
    print("\nIf you encounter any issues, please check the documentation at:")
    print("https://huggingface.co/docs/huggingface_hub/") 