from setuptools import setup, find_packages

setup(
    name="seals",
    version="0.1.0",
    description="SEALS: Self-Evolving AI Lifecycle Simulator - A reference implementation for studying deployed ML system dynamics",
    author="Author Name",
    author_email="author@example.com",
    url="https://github.com/username/seals",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "shap>=0.41.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="machine learning, drift, feedback, retraining, ml lifecycle",
)
