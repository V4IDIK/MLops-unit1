from fastapi import FastAPI
import joblib
import random

app = FastAPI(title="MLOps A/B Testing API")

# Load both models into memory when the server starts
model_a = joblib.load("models/model_a.pkl")
model_b = joblib.load("models/model_b.pkl")

# Define target names for the Iris dataset
species = ["setosa", "versicolor", "virginica"]

@app.get("/")
def home():
    return {"message": "Iris A/B Testing API is running!"}

@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    # 1. Format the incoming data
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    # 2. The A/B Router: Randomly select between Model A and Model B
    selected_model_name = random.choice(["Model A (Logistic Regression)", "Model B (Random Forest)"])
    
    # 3. Make the prediction using the selected model
    if "Model A" in selected_model_name:
        prediction_idx = model_a.predict(features)[0]
    else:
        prediction_idx = model_b.predict(features)[0]
        
    # 4. Return the result along with which model handled the request
    return {
        "predicted_species": species[prediction_idx],
        "served_by": selected_model_name,
        "measurements_received": features[0]
    }