"""
Setup script for Satellite Terrain Classification System
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="satellite-terrain-classifier",
    version="2.0.0",
    author="BigData Satellite Surveillance Team",
    description="Production-ready satellite terrain classification with state-of-the-art AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexv879/BigDataSatelliteSurveillanceTerrain",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "matplotlib>=3.8.0",
        "loguru>=0.7.2",
        "pydantic>=2.5.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "rasterio>=1.3.0",
        "pyproj>=3.6.0",
        "scikit-image>=0.22.0",
        "PyWavelets>=1.4.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "full": [
            "torch>=2.1.0",
            "albumentations>=1.3.1",
            "mlflow>=2.8.0",
            "wandb>=0.16.0",
            "optuna>=3.4.0",
            "shap>=0.43.0",
            "folium>=0.15.0",
            "pydensecrf>=1.0.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "satellite-classifier=main:main",
        ],
    },
)
