## 📊 Dataset Description
# Ride Demand Prediction Engine

This repository implements a demand prediction for ride-hailing. It contains data ingestion, feature engineering, model training pipelines, and a Flask-based dashboard/inference service.

**Quick links:**

- Usage and dataset details: [docs/USAGE.md](docs/USAGE.md)
- Project components and API: [docs/COMPONENTS.md](docs/COMPONENTS.md)

## Quickstart

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Create a `.env` file in the project root with any required secrets (example):

```
HOPSWORKS_API_KEY=your_hopsworks_api_key
API_KEY=your_visualcrossing_api_key
NYC_OPEN_DATA_APP_TOKEN=optional_socrata_token
```

3. Run the dashboard / API server:

```bash
python app.py
```

The Flask app serves the frontend at `http://localhost:5000`. The API endpoint `/api/demand` returns the prediction JSON used by the map/dashboard.

## HTTP API

- `GET /api/demand` : Returns the latest predictions JSON used by the frontend map and dashboard. It contacts Hopsworks feature store to fetch prepared prediction rows and returns a structure with `metadata` and `predictions` keyed by zone id.
- `GET /taxi_zones.json` : Serves the local `taxi_zones.json` geojson file for the frontend map.

Example (local):

```bash
curl http://localhost:5000/api/demand | jq .
```

## Environment variables

- `HOPSWORKS_API_KEY` — API key for Hopsworks feature store and model registry.
- `API_KEY` — API key for the VisualCrossing weather API (used by ingestion/inference).
- `NYC_OPEN_DATA_APP_TOKEN` — (optional) Socrata app token for NYC open data speed feeds.
- `PORT` — Flask port override (defaults to 5000).

## Developer quick commands

- Run the Flask dashboard locally:

```bash
python app.py
```

- Run the prediction pipeline manually (CLI):

```bash
python /Dynamic-Pricing-Engine/prediction_pipeline.py
```

- Run the training pipeline manually (CLI):

```bash
python /Dynamic-Pricing-Engine/training_pipeline.py
```

## Project layout

Key folders and files:

- `app.py` : Flask dashboard & API endpoint
- `src/DynamicPricingEngine` : core application modules (ingestion, transformation, training, inference)
- `training_pipeline.py` and `prediction_pipeline.py` : pipeline runners
- `requirements.txt` : pinned Python dependencies

For more details see the docs directory.