import asyncio
import os
import sys
from datetime import datetime

from flask import Flask, render_template, send_from_directory, jsonify
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
import hopsworks

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException

load_dotenv()

app = Flask(__name__)

_hopsworks_project = None
_hopsworks_lock = asyncio.Lock()


async def _get_hopsworks_project():
    """Return a cached Hopsworks project handle, re-logging only on first call."""
    global _hopsworks_project
    if _hopsworks_project is not None:
        return _hopsworks_project
    async with _hopsworks_lock:
        if _hopsworks_project is not None:
            return _hopsworks_project
        api_key = os.getenv("HOPSWORKS_API_KEY")
        _hopsworks_project = await asyncio.to_thread(
            hopsworks.login, project='RideDemandPrediction', api_key_value=api_key
        )
        logger.info("Hopsworks connection established")
        return _hopsworks_project


def _compute_prediction_window(now_ny):
    """Round to the nearest prediction hour (cutoff at minute 35)."""
    target = now_ny.replace(tzinfo=None)
    if target.minute < 35:
        return target.replace(minute=0, second=0, microsecond=0)
    return target.replace(minute=0, second=0, microsecond=0) + relativedelta(hours=1)


def _build_response(df, now_ny):
    """Convert a Hopsworks result DataFrame into the API response dict."""
    df.columns = df.columns.str.replace('nycdemandprediction_', '', regex=False)
    df['bin'] = df['bin'].astype(str)
    predictions = df.set_index('pulocationid').sort_index().to_dict(orient='index')
    return {
        "metadata": {
            "generated_at": now_ny.isoformat(),
            "total_zones": len(df),
            "prediction_window": df['bin'].iloc[0] if 'bin' in df.columns else "Unknown",
        },
        "predictions": predictions,
    }


async def _fetch_demand_data():
    """Core async data-fetching logic, reused by the Flask route."""
    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)
    pred_time = _compute_prediction_window(now_ny)

    project = await _get_hopsworks_project()
    fs = await asyncio.to_thread(project.get_feature_store)

    fg = await asyncio.to_thread(
        fs.get_feature_group, name='demandpred', version=1
    )

    final_features = [
        'bin', 'humidity', 'precip', 'windspeed', 'feelslike', 'visibility',
        'pulocationid', 'zone_congestion_index', "city_congestion_index",
        'target_yellow', 'target_green', 'target_hvfhv',
    ]

    for attempt in range(2):
        window = pred_time if attempt == 0 else pred_time
        try:
            query = fg.select(final_features).filter(
                fg.get_feature('bin') == window
            )
            df = await asyncio.to_thread(query.read)
            if df.empty:
                if attempt == 0:
                    pred_time = now_ny.replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
                    continue
                return {"metadata": {"generated_at": now_ny.isoformat(), "total_zones": 0, "prediction_window": str(window)}, "predictions": {}}
            logger.info("Retrieved %d rows for window: %s", len(df), window)
            return _build_response(df, now_ny)
        except Exception:
            if attempt == 0:
                pred_time = now_ny.replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
                continue
            raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/taxi_zones.json')
def get_geojson():
    return send_from_directory('data', 'taxi_zones.json')


@app.route("/api/demand")
async def get_demand_data():
    try:
        result = await _fetch_demand_data()
        return jsonify(result)
    except Exception as e:
        logger.error("Failed to fetch demand data: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
