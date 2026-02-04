from flask import Flask, render_template, send_from_directory, jsonify
import os,sys
from datetime import datetime, timedelta

from prediction_pipeline import prediction
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract
import pandas as pd
from zoneinfo import ZoneInfo
import hopsworks
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta

load_dotenv()

app = Flask(__name__)

# Route to serve the main dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Crucial: Flask needs a route to serve your local JSON file
@app.route('/taxi_zones.json')
def get_geojson():
    return send_from_directory(os.getcwd(), 'taxi_zones.json')

@app.route("/api/demand")
def get_demand_data():
    """Returns the full prediction JSON for the frontend map and dashboard."""
    try:
        hopsworks_api = os.getenv("HOPSWORKS_API_KEY")
        ny_tz = ZoneInfo("America/New_York")
        project = hopsworks.login(project='RideDemandPrediction', api_key_value=hopsworks_api)

        now = datetime.now(ny_tz)
        target_time = now.replace(tzinfo=None)
        print(target_time)

        if target_time.minute < 35:
            pred_time = target_time.replace(minute=0, second=0, microsecond=0)
        else:
            pred_time = target_time.replace(minute=0, second=0, microsecond=0) + relativedelta(hours=1)

        logger.info('Retrieving the dataset from hopsworks feature store')

        fs = project.get_feature_store()
            
        fg = fs.get_feature_group(
                name = 'demandpred',
                version = 1)

        final_features = ['bin', 'humidity', 'precip','windspeed', 'feelslike', 'visibility', 
                            'pulocationid', 'zone_congestion_index', "city_congestion_index",
                                    'target_yellow', 'target_green', 'target_hvfhv']
        try:
            
            query = fg.select(final_features).filter(
                (fg.get_feature('bin') > pred_time))

            df = query.read()
            logger.info(f"Successfully retrieved {len(df)} rows for window: {pred_time}")
            df.columns = df.columns.str.replace('nycdemandprediction_', '', regex=False)

            df['bin'] = df['bin'].astype(str)
            predictions_dict = df.set_index('pulocationid').sort_index().to_dict(orient='index')

            final_output = {
                    "metadata": {
                        "generated_at": datetime.now(ny_tz).isoformat(),
                        "total_zones": len(df),
                        "prediction_window": df['bin'][0]# if 'bin' in df.columns else "Unknown"
                    },
                    "predictions": predictions_dict
                }
        
        except:
            pred_time = target_time.replace(minute=0, second=0, microsecond=0)
            query = fg.select(final_features).filter(
                (fg.get_feature('bin') > pred_time))

            df = query.read()
            logger.info(f"Successfully retrieved {len(df)} rows for window: {pred_time}")
            df.columns = df.columns.str.replace('nycdemandprediction_', '', regex=False)

            df['bin'] = df['bin'].astype(str)
            predictions_dict = df.set_index('pulocationid').sort_index().to_dict(orient='index')

            final_output = {
                    "metadata": {
                        "generated_at": datetime.now(ny_tz).isoformat(),
                        "total_zones": len(df),
                        "prediction_window": df['bin'][0]# if 'bin' in df.columns else "Unknown"
                    },
                    "predictions": predictions_dict
                }


        return final_output

    except Exception as e:
        logger.error(f"Failed to extract NYC demand prediction pickup data: {e}")
        raise RideDemandException(e,sys)
        

if __name__ == '__main__':
    # Default Flask port is 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)