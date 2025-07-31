from fastapi import FastAPI, HTTPException
import logging
from src.apis.predict.load_model import load_model
from src.apis.predict.make_prediction import make_prediction
from src.apis.predict.schemas import BankruptcyPredictionInput

app=FASTAPI()
handler=Magnum(app)

@app.post("/predict")
def predict(payload: BankruptcyPredictionInput):
    try:
        # Convert input data to model features array
        logging.info('Calling the make_prediction() function...')
        prediction=make_prediction(payload)
        logging.info('make_prediction function made the prediction')
        
    except Exception as e:
        logging.error(f"Some error occured in the api call:{e}")
        raise HTTPException(status_code=400, detail=str(e))