## Project Components

This file describes the main modules and their purpose inside `src/DynamicPricingEngine`.

- `src/DynamicPricingEngine/component/`
  - `data_ingestion.py`: helpers to read and ingest raw data for training and feature building.
  - `data_transformation.py`: feature engineering and transformation logic used to prepare training inputs.
  - `model_training.py`: training routines and model persistence logic.
  - `inference.py`: inference wrappers used by pipelines to produce predictions.

- `src/DynamicPricingEngine/pipeline/`
  - `training_pipeline.py`: orchestrates data ingestion, transformation, model training, and artifact saving.
  - `prediction_pipeline.py`: builds features and runs the model to produce predictions.
  - `training_data_ingestion_pipeline.py`: pipeline to collect and persist training datasets.

- `src/DynamicPricingEngine/inference/`
  - `feature_builder.py`: constructs features for inference from live inputs or feature store
  - `predict.py`: prediction utilities used by the Flask app and other consumers
  - `schema.py`: input/output schema definitions

- `src/DynamicPricingEngine/config/`
  - `configuration.py`: central configuration helpers and config entity definitions

- `src/DynamicPricingEngine/utils/`
  - `data_ingestion_utils.py`, `common_utils.py`, `ml_utils.py`, `model_utils.py`: utility functions used across pipelines

### Entry points

- `app.py` : starts a Flask app that serves a dashboard and the `/api/demand` endpoint. It uses `prediction_pipeline` and the `inference` modules to retrieve predictions (from Hopsworks feature store in this project).
- `prediction_pipeline.py` and `training_pipeline.py` : CLI-friendly runners to produce predictions or train models.

### HTTP API Endpoints

- `GET /api/demand` — Returns the prediction JSON the frontend consumes. The response contains `metadata` and a `predictions` mapping keyed by `pulocationid`.
- `GET /taxi_zones.json` — Serves the `taxi_zones.json` geojson file used by the frontend map.

### CLI & Scripts

- `src/DynamicPricingEngine/pipeline/feature_pipeline.py` — Runs ingestion + transformation end-to-end.
- `src/DynamicPricingEngine/pipeline/training_pipeline.py` — Triggers model training and registry upload.
- `src/DynamicPricingEngine/pipeline/prediction_pipeline.py` — Builds inference rows and pushes predictions.

### Extending components

When adding a new model or feature group:

1. Add feature generation in `data_transformation.py` or a dedicated module under `inference/feature_builder.py`.
2. Add training logic to `model_training.py` and wire it into `training_pipeline.py`.
3. Persist model artifacts and update `prediction_pipeline.py` / `inference/predict.py` to load and use the model.
