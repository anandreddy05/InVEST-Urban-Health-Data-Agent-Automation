# 🌆 Urban Mental Health Data Agent

Automated Geospatial Data Acquisition & Processing using LLMs, FastAPI, and Google Earth Engine

## 🧠 Overview

- This project automates the retrieval, processing, and management of spatial datasets (like NDVI, Land Cover, Tree Cover, and Population Density) through a Large Language Model (LLM)-powered agent.
- It was developed as part of the InVEST Urban Mental Health Automation Assignment.

The system enables users to request data in natural language, such as:

“Get NDVI and Land Cover data for Austin, Texas in 2020.”

and automatically performs:

📍 Location extraction using an LLM agent (GeoLLMAgent)

🌍 Bounding box retrieval using OpenStreetMap geocoding

🛰️ Dataset acquisition from Google Earth Engine and WorldPop

🧩 Raster preprocessing (clipping, reprojection, validation)

📑 Manifest generation and caching

🎨 Visualization via Streamlit dashboard

🏗️ System Architecture

```plaintext
 ┌──────────────────────┐
 │ Streamlit Frontend   │
 │  - Natural Language  │
 │  - Manifest Viewer   │
 └──────────┬───────────┘
            │ REST API (FastAPI)
 ┌──────────▼───────────────────────┐
 │ FastAPI Backend                  │ 
 │  - /agents/data/fetch            │ 
 │  - /agents/data/natural-language │ 
 │  - /agents/data/conversational   │ 
 └──────────┬───────────────────────┘
            │
 ┌──────────▼──────────────────┐
 │ GeoLLMAgent (LangChain)     │
 │  → Extract city, data_types │
 │  → Uses OpenAI GPT-4o-mini  │
 └──────────┬──────────────────┘
            │
 ┌──────────▼───────────────────────────────────────────┐
 │ EarthEngineDataAgent                                 │
 │  → Fetch NDVI, LandCover, TreeCover, Population      │
 │  → Generate raster thumbnails                        │
 │  → Save manifest.json                                │
 └──────────┬───────────────────────────────────────────┘
            │
 ┌──────────▼────────────────────────┐
 │ SpatialProcessor                  │
 │  → Clip, reproject, align rasters │
 │  → Validate CRS, shape, resolution│
 └──────────┬────────────────────────┘
            │
 ┌──────────▼───────────┐
 │ DataValidator        │
 │  → Validate rasters  │ 
 │  → Generate logs     │
 └──────────────────────┘

```

## ⚙️ Core Components

### 🧭 1. GeoLLMAgent (LLM-Powered Geospatial Parser)

- Uses LangChain + GPT-4o-mini to interpret natural language prompts.

- Extracts: City / region name

- Requested data types (NDVI, land_cover, tree_cover, population)

- Falls back to keyword-based parsing if the LLM fails.

- Returns structured JSON (GeoResponse) with bounding box and data types.

### 🌍 2. EarthEngineDataAgent

- Connects with Google Earth Engine for dataset retrieval:

- Land Cover → USGS/NLCD_RELEASES/2019_REL/NLCD/2019

- Tree Cover → NLCD_TCC dataset

- NDVI → Computed from Sentinel-2 bands (B8, B4)

- Population → WorldPop REST endpoint

- Generates raster PNG thumbnails dynamically using .getThumbURL()

- Saves outputs and manifest JSON in /outputs/

### 🗺️ 3. Geocoder

- Uses OpenStreetMap (Nominatim) via geopy to fetch bounding boxes.

- Converts city names into lat/lon bounding coordinates for Earth Engine queries.

Example:

```json

{
  "min_lat": 30.0985,
  "max_lat": 30.5166,
  "min_lon": -97.9367,
  "max_lon": -97.5605,
  "center_lat": 30.2711,
  "center_lon": -97.7437,
  "name": "Austin, Travis County, Texas, United States of America"
}
```

### 🧩 4. SpatialProcessor

- Performs raster preprocessing steps:

- Clip raster to AOI bounding box

- Reproject to target CRS (e.g., EPSG:5070)

- Align rasters to ensure consistent grid resolution

- Uses rasterio, geopandas, and shapely.

### ✅ 5. DataValidator

- Checks raster integrity:

- CRS consistency

- Resolution & bounds

- Nodata handling

- Min/Max pixel range

- Generates detailed manifest and validation report in JSON format.

## 🔌 API Endpoints (FastAPI)

| Endpoint                        | Method | Description                                 |
| ------------------------------- | ------ | ------------------------------------------- |
| `/`                             | GET    | Health check                                |
| `/agents/data/fetch`            | POST   | Fetch data using structured request         |
| `/agents/data/natural-language` | POST   | Fetch data using natural language (via LLM) |
| `/agents/data/conversational`   | POST   | Chat-style retrieval with contextual memory |
| `/agents/data/parse-prompt`     | POST   | Parse prompt only (debugging)               |
| `/jobs/{job_id}/status`         | GET    | Retrieve job completion status              |

## Example Request (Natural Language)

```bash
curl -X POST "http://127.0.0.1:8000/agents/data/natural-language" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Get NDVI and land cover for New York City", "year": 2020}'

```

Example Response

```json
{
  "job_id": "data_20251101_085058_796964cd",
  "city": "\u90a3\u4e48, \u6765\u5bbe\u5e02, \u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a, \u4e2d\u56fd",
  "timestamp": "2025-11-01T08:51:06.240468",
  "datasets": {
    "land_cover": {
      "source": "NLCD Land Cover",
      "year": 2020,
      "bbox": {
        "min_lat": 23.8444,
        "max_lat": 23.8844,
        "min_lon": 109.039,
        "max_lon": 109.079,
        "center_lat": 23.8644,
        "center_lon": 109.059,
        "name": "\u90a3\u4e48, \u6765\u5bbe\u5e02, \u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a, \u4e2d\u56fd"
      },
      "format": "PNG"
    },
    "ndvi": {
      "source": "Sentinel-2 NDVI",
      "year": 2020,
      "bbox": {
        "min_lat": 23.8444,
        "max_lat": 23.8844,
        "min_lon": 109.039,
        "max_lon": 109.079,
        "center_lat": 23.8644,
        "center_lon": 109.059,
        "name": "\u90a3\u4e48, \u6765\u5bbe\u5e02, \u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a, \u4e2d\u56fd"
      },
      "format": "PNG"
    }
  },
  "outputs": {
    "land_cover": "outputs/data_20251101_085058_796964cd_land_cover.png",
    "ndvi": "outputs/data_20251101_085058_796964cd_ndvi.png"
  },
  "status": "completed"
}
```

## 💻 Streamlit Dashboard

### Features

- 🎤 Accepts natural language or structured inputs

- 📂 Displays all available manifests in /outputs/

- 🖼️ Visualizes raster PNGs side-by-side

- 🧾 Shows job metadata and manifest summaries

- 💬 Provides direct integration with FastAPI backend

## Run Streamlit App

```bash
streamlit run app.py
```

📦 Directory Structure

```plaintext
Data_Agent/
│
├── agents/
│   ├── data_agent.py
│   ├── geollm_agent.py
│   ├── test_data_and_geollm.py
│
├── utils/
│   ├── geocoding.py
│   ├── spatial_processing.py
│   ├── validator.py
│
├── outputs/              # Generated images & manifests
│   ├── data_xxx_land_cover.png
│   ├── data_xxx_ndvi.png
│   ├── data_xxx_manifest.json
│
├── main.py               # FastAPI backend
├── app.py                # Streamlit frontend
├── requirements.txt
└── README.md

```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Authenticate Google Earth Engine

```bash
earthengine authenticate
```

## Run Backend

```bash
uvicorn main:app --reload
```

→ Visit <http://127.0.0.1:8000/docs>

## Run Streamlit Frontend

```bash
streamlit run app.py

```

## Streamlit Output Images

Below are the output maps generated by the Data Agent.

![Land and NDVI](https://raw.githubusercontent.com/<your-username>/<repo-name>/main/img_outputs/land_and_nvdi.png)
![Tree Cover](https://raw.githubusercontent.com/<your-username>/<repo-name>/main/img_outputs/Tree_Cover.png)


## Technologies Used

| Category               | Tools / Frameworks                           |
| ---------------------- | -------------------------------------------- |
| **Backend**            | FastAPI, Uvicorn                             |
| **LLM Processing**     | LangChain, OpenAI GPT-4o-mini                |
| **Data Sources**       | Google Earth Engine, WorldPop, OpenStreetMap |
| **Spatial Analysis**   | Rasterio, GeoPandas, Shapely                 |
| **Frontend**           | Streamlit                                    |
| **Validation & Utils** | Pydantic, NumPy, Requests, PIL               |

## 🧠 Key Learnings & Highlights

- Built a modular agent pipeline combining LLM + Earth Engine.

- Automated geospatial data retrieval without manual GEE scripting.

- Integrated LLM reasoning for flexible natural language interpretation.

- Designed end-to-end system (FastAPI + Streamlit + Earth Engine).

- Added validation, logging, and manifest-based data lineage.

- Demonstrated scalable architecture for future AI-Geo pipelines.
