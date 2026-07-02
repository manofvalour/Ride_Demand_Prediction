# Ride Demand Prediction Engine

This repository implements a demand prediction for ride-hailing. It contains data ingestion, feature engineering, model training pipelines, and a Flask-based dashboard/inference service.

**Quick links:**

- Usage and dataset details: [docs/USAGE.md](docs/USAGE.md)
- Project components and API: [docs/COMPONENTS.md](docs/COMPONENTS.md)

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python app.py          # dashboard at http://localhost:5000
```

Or with Docker:

```bash
docker compose up --build
```

## API

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard frontend |
| `GET /api/demand` | Latest predictions JSON (metadata + per-zone demand) |
| `GET /taxi_zones.json` | NYC taxi zone GeoJSON for the map |

## Pipelines

| Script | Schedule | What it does |
|---|---|---|
| `python pipelines/data_retrieval.py` | Monthly | Downloads NYC TLC + weather data, engineers features, pushes to Hopsworks |
| `python pipelines/training.py` | Monthly | Optuna tuning → trains MultiOutputRegressor → registers model |
| `python pipelines/prediction.py` | Hourly | Live weather → inference → pushes predictions to Hopsworks |

## Project layout

```
├── app.py                 # Flask dashboard + API
├── data/                  # GeoJSON, shapefiles
├── pipelines/             # CLI entrypoints (data_retrieval, training, prediction)
├── scripts/               # Dev utilities (template generator)
├── src/DynamicPricingEngine/
│   ├── component/         # data_ingestion, data_transformation, model_training, inference
│   ├── config/            # ConfigurationManager → YAML → typed dataclasses
│   ├── pipeline/          # feature_pipeline, training_data_ingestion_pipeline
│   ├── utils/             # common_utils, ml_utils (Optuna), model_utils
│   ├── logger/            # file + stdout logging
│   ├── exception/         # RideDemandException
│   └── constants/         # path constants
├── config/config.yaml     # pipeline paths and URLs
├── params.yaml            # ML hyperparameters and split ratios
├── templates/index.html   # frontend dashboard
├── static/style.css       # dashboard styles
├── tests/                 # test suite
├── Dockerfile             # multi-stage (dev / prod)
└── requirements.txt       # pinned dependencies
```

## Environment variables

| Variable | Required | Source |
|---|---|---|
| `HOPSWORKS_API_KEY` | Yes | Hopsworks feature store + model registry |
| `API_KEY` | Yes | VisualCrossing weather API |
| `NYC_OPEN_DATA_APP_TOKEN` | No | Socrata NYC speed feed |
| `PORT` | No | Flask port (default 5000) |

## Docker

```bash
# Dev mode (Flask debug server, hot-reload)
docker compose up --build

# Production mode (gunicorn + uvicorn)
DOCKER_TARGET=prod docker compose up --build
```

## Tests

```bash
python -m pytest tests/
```

## Project layout

Key folders and files:

- `app.py` : Flask dashboard & API endpoint
- `src/DynamicPricingEngine` : core application modules (ingestion, transformation, training, inference)
- `training_pipeline.py` and `prediction_pipeline.py` : pipeline runners
- `requirements.txt` : pinned Python dependencies

For more details see the docs directory.