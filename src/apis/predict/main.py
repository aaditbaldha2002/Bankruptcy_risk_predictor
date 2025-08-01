from fastapi import FastAPI, HTTPException
import logging

import pandas as pd
from src.apis.predict.constants import FEATURE_NAMES
from src.apis.predict.make_prediction import make_prediction
from src.apis.predict.schemas import BankruptcyPredictionInput, BankruptcyPredictionResponse
from mangum import Mangum

app=FastAPI()
handler=Mangum(app)

@app.post("/predict",tags=['Prediction'],response_model=BankruptcyPredictionResponse)
def predict(payload: BankruptcyPredictionInput):
    try:
        # Convert input data to model features array
        logging.info('Calling the make_prediction() function...')
        prediction=make_prediction(payload)
        logging.info(f'make_prediction function made the prediction:{prediction}')
        return BankruptcyPredictionResponse(
            prediction=prediction,
        )

    except Exception as e:
        logging.error(f"Some error occurred in the api call:{e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/debug-input")
def debug_input(payload: BankruptcyPredictionInput):
    data_dict=payload.dict()
    ordered_data = {col: data_dict.get(col, 0) for col in FEATURE_NAMES}
    df = pd.DataFrame([ordered_data])
    return {"columns": list(df.columns)}