# Earth Observation Digital Twin (EODT) - Transport Network

**EODT** is an ESA funded, open-source Python package for identifying usable roads and other transport infrastructure after a natural disaster using satellite imagery, OpenStreetMap data, and ML/AI.

---

## Purpose

- Ingest satellite imagery and OSM data
- Use a CNN-based model to detect road usability (blocked, damaged, etc.)
- Serve predictions via API for integration into disaster response systems

---

## Features

- FastAPI-based REST API for querying road status
- PyTorch-powered CNN for image recognition
- Support for pre/post-disaster image analysis
- OpenStreetMap integration via `osmnx`

---

## Installation

```bash
pip install -r requirements.txt
```

or for development

```bash
git clone https://github.com/ArthurTolley/EODT.git
cd EODT
pip install -e .
```