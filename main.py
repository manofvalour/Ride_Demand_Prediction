from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="NYC Ride Demand API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED: Pointing to the actual filename in your directory
DATA_FILE = "artifacts/inference/predictions.json"

@app.get("/")
async def root():
    """Welcome message to confirm the server is running"""
    return {
        "status": "online",
        "message": "NYC Demand API is active. Go to /api/demand to see data.",
        "endpoints": ["/api/demand", "/docs"]
    }

@app.get("/api/demand")
async def get_demand_data():
    """Returns the full prediction JSON for the frontend map and dashboard."""
    if not os.path.exists(DATA_FILE):
        raise HTTPException(
            status_code=404, 
            detail=f"File {DATA_FILE} not found. Ensure the file is in the same folder as main.py"
        )
    
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # If running locally, this will start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)