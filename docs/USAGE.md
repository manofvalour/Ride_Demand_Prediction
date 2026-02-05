## Usage

This document explains how to set up and run the Dynamic Pricing / Demand Prediction Engine locally.

### Prerequisites

- Python 3.10+ (recommended)
- A virtual environment tool (venv, conda)
- Hopsworks account / API key if you want to retrieve features from Hopsworks Feature Store

### Install

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root with at least the following when using Hopsworks feature store:

```
HOPSWORKS_API_KEY=your_hopsworks_api_key
```

Other environment variables may be present in your deployment or CI configuration; check `app.py` and other files for usages of `os.getenv`.

### Run the app (local)

The Flask frontend and API can be started with:

```bash
python app.py
```

Open `http://localhost:5000` in your browser to view the dashboard. The API endpoint `GET /api/demand` returns a JSON payload of predictions consumed by the frontend.

### Docker

There is a `DockerFile` in the repository you can use to containerize the app. Build and run:

```bash
docker build -t dynamic-pricing:latest .
docker run -p 5000:5000 --env-file .env dynamic-pricing:latest
```

### Data & Features

The dataset description and feature inventory used by the project are in `docs/USAGE.md`. (Includes weather, calendar/time, traffic, and lagged history features.)

### Troubleshooting

- If the Flask app fails to connect to Hopsworks, verify `HOPSWORKS_API_KEY` and network access.
- For dependency issues, ensure Python version matches requirements and recreate the virtual environment.
