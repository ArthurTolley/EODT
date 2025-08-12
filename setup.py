# setup.py (Corrected Version)

from setuptools import setup, find_packages

setup(
    name="road_det_data_gen",  # <-- The correct name of your package
    version="0.1.0",
    description="A tool to generate training data for road detection from satellite imagery and OSM.",
    author="Arthur Tolley", # Replace with your name/org
    author_email="your@email.com", # Replace with your email
    url="https://github.com/ArthurTolley/road-det-data-gen", # Replace with your repo URL
    package_dir={"": "src"},  # <-- This tells setuptools to look for packages in the 'src' directory
    packages=find_packages(where="src"), # <-- This finds your 'road-det-data-gen' package inside 'src'
    install_requires=[
        "earthengine-api",
        "requests",
        "rasterio",
        "numpy",
        "Pillow",
        "pyproj",
        "osmnx",
        "geopandas",
        "scikit-learn",
        "matplotlib",
        "jupyter",
        "rtree",
        "opencv-python"
        # Removed FastAPI/Uvicorn as per our decision
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9", # Changed from 3.13 to a more common version
)