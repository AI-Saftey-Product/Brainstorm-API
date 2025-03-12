"""Setup script for the custom testing framework."""

from setuptools import setup, find_packages

setup(
    name="custom_test_framework",
    version="0.1.0",
    description="Custom testing framework for AI models",
    author="AI Framework Team",
    author_email="example@example.com",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "custom-test=custom_test_framework.cli.main:main",
        ],
    },
    install_requires=[
        "pydantic>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 