from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

# load model components
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

@app.get("/")
def home():
    return {"message": "Bank Marketing API is running"}

@app.get("/predict")
def predict(age: int, campaign: int, previous: int, euribor: float):
    
    # create input vector (same size as training)
    input_data = np.zeros((1, scaler.n_features_in_))

    input_data[0][0] = age
    input_data[0][1] = campaign
    input_data[0][2] = previous
    input_data[0][3] = euribor

    # preprocessing
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    # prediction
    result = model.predict(input_pca)

    return {"prediction": str(result[0])}