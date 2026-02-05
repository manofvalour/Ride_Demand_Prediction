# Reference: Main classes and responsibilities

This reference lists the most important classes and where to find them.

- `DataIngestion` — `src/DynamicPricingEngine/component/data_ingestion.py`
  - Downloads monthly taxi parquet files and hourly weather data.
  - Normalizes column names across taxi types and produces hourly `bin` timestamps.
  - Methods: `get_NYC_ride_data`, `derive_targets`, `extract_nyc_weather_data`, `save_data_to_artifact`, `initiate_data_ingestion`.

- `DataTransformation` — `src/DynamicPricingEngine/component/data_transformation.py`
  - Merges weather and target data, engineers temporal, neighbor, and autoregressive features at scale using Dask.
  - Methods: `merge_weather_features`, `engineer_temporal_feature`, `generate_neighbor_features`, `engineer_autoregressive_signals`, `push_transformed_data_to_feature_store`.

- `ModelTrainer` — `src/DynamicPricingEngine/component/model_training.py`
  - Retrieves engineered features from the feature store, splits per-zone time series, runs Optuna tuning and trains multi-output regressors, and registers the best model.
  - Methods: `retrieve_engineered_feature`, `split_data`, `_prepare_features`, `model_training_and_evaluation`, `save_model_to_model_store`.

- `Inference` — `src/DynamicPricingEngine/component/inference.py`
  - Builds prediction rows using live weather, historical features and neighbor/congestion signals, downloads the registered model and produces predictions, then pushes them to the `demandpred` feature group.
  - Methods: `get_nyc_prediction_weather_data`, `engineer_temporal_prediction_features`, `extract_historical_pickup_data`, `get_zone_speeds`, `prepare_and_predict`, `push_prediction_to_feature_store`, `initiate_inference`.

- `FeaturePipeline` — `src/DynamicPricingEngine/pipeline/feature_pipeline.py`
  - Simple orchestration that runs ingestion and transformation and persists features.

See the individual modules for detailed docstrings and function-level documentation.
