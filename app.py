from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="ML Project",description="Linear Regression Model for House Price")

#load Trained Model
model = joblib.load("model/house_price_model.pkl")

class HouseInput(BaseModel):
    square_feet: float


@app.get("/")
def index():
    return {'Msg':'welcome ML Project with fastapi'}

@app.post("/predict")
def predict_house_price(data: HouseInput):
    prediction = model.predict([[data.square_feet]])

    return {'square_feet':data.square_feet,"Prediction Price ":prediction[0]}