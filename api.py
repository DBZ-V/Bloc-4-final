from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
import logging
import mlflow
import mlflow.sklearn
from datetime import datetime

# 📌 Initialisation FastAPI
app = FastAPI(title="Bike Sharing Prediction API", description="Prédit le nombre de vélos disponibles dans Londres.")

# 📌 Logger pour suivre les requêtes
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📌 Charger le modèle et l’encoder
MODEL_PATH = "models/model_bike_sharing.pkl"
ENCODER_PATH = "models/encoder_bike.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    weather_encoder = pickle.load(f)

# 📌 Initialisation de MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")  # Lancer MLflow en local
mlflow.set_experiment("Bike Sharing Prediction")

# 📌 Définition du format des requêtes
class PredictionInput(BaseModel):
    station: str
    hour: int
    day_of_week: int
    weather: str

@app.get("/")
def home():
    return {"message": "API de prédiction des vélos - Active 🚲"}

@app.post("/predict/")
def predict(data: PredictionInput):
    try:
        # 📌 Encodage de la météo
        encoded_weather = weather_encoder.transform([data.weather])[0]

        # 📌 Préparation des données pour Prophet
        future = pd.DataFrame({
            "ds": [pd.Timestamp.now()],  # Date actuelle
            "Hour": [data.hour],
            "Day_of_Week": [data.day_of_week],
            "Weather_encoded": [encoded_weather]
        })

        # 📌 Prédiction
        forecast = model.predict(future)
        predicted_bikes = max(0, int(forecast["yhat"].values[0]))  # Évite les valeurs négatives

        # 🔥 Logger dans MLflow
        with mlflow.start_run() as run:
            mlflow.log_param("station", data.station)
            mlflow.log_param("hour", data.hour)
            mlflow.log_param("day_of_week", data.day_of_week)
            mlflow.log_param("weather", data.weather)
            mlflow.log_metric("prediction", predicted_bikes)
            mlflow.log_metric("timestamp", datetime.now().timestamp())

            logging.info(f"✅ MLflow Run ID: {run.info.run_id}")  # 🔹 Vérifier si MLflow logge bien

        # 📌 Logging standard
        logging.info(f"Prediction demandée: {data}")
        logging.info(f"Résultat: {predicted_bikes} vélos")

        return {"station": data.station, "hour": data.hour, "day_of_week": data.day_of_week,
                "weather": data.weather, "predicted_bikes": predicted_bikes}

    except Exception as e:
        logging.error(f"🚨 Erreur de prédiction: {e}")
        
        # 🔥 Logger aussi dans MLflow en cas d'erreur
        with mlflow.start_run():
            mlflow.log_param("error", str(e))

        return {"error": str(e)}
