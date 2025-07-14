from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

model = joblib.load("california_knn_pipeline.pkl")


app = FastAPI()

class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float



@app.post("/predict")

def predict_price(data: HouseData):
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)

    return {"predicted price": prediction[0]}