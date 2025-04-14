from setuptools import setup, find_packages

setup(
    name="eodt",
    version="0.1.0",
    description="Earth Observation Digital Twin for Post-Disaster Road Usability",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/your-org/EODT",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "torch",
        "torchvision",
        "rasterio",
        "osmnx",
        "geopandas",
        "pydantic",
        "scikit-learn",
        "Pillow",
        "python-multipart"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.13",
)
