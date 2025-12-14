"""Setup script for ForesightBench."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="foresight_bench",
    version="0.1.0",
    author="ForesightBench Team",
    description="A benchmark for evaluating LLM plan-execution faithfulness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/foresight-bench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies only - API clients are optional
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "all": ["openai>=1.0.0", "anthropic>=0.18.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "foresight-bench=foresight_bench.__main__:main",
        ],
    },
)
