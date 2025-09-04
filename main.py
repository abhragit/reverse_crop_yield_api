from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("reverse_yield_pipeline.pkl")

# Define request schema
class PredictInput(BaseModel):
    Crop_Type: str
    Soil_Type: str
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Crop_Yield: float

# Create FastAPI app
app = FastAPI(title="Fertilizer Recommendation API")

@app.get("/")
def home():
    return {"message": "Fertilizer Recommendation API is running"}

@app.post("/predict")
def predict(data: PredictInput):
    try:
        # Prepare input
        input_data = pd.DataFrame([{
            "Crop_Type": data.Crop_Type,
            "Soil_Type": data.Soil_Type,
            "Temperature": data.Temperature,
            "Humidity": data.Humidity,
            "Wind_Speed": data.Wind_Speed,
            "Crop_Yield": data.Crop_Yield
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        return {
            "input_used": input_data.to_dict(orient="records")[0],
            "recommended_NPK": {
                "N": round(prediction[0], 2),
                "P": round(prediction[1], 2),
                "K": round(prediction[2], 2),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
