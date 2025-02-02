import pandas as pd
import pickle
import os
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder

# 🔹 Chargement du dataset
DATA_PATH = "data/London_Bike_Sharing_Dataset.csv"
df = pd.read_csv(DATA_PATH)

# 🔹 Conversion de la date
print("📌 Conversion des dates...")
df['Date'] = pd.to_datetime(df['Date'])
df.rename(columns={'Date': 'ds', 'Bike_Count': 'y'}, inplace=True)

# 🔹 Encodage de la météo
print("📌 Encodage de la météo...")
weather_encoder = LabelEncoder()
df['Weather_encoded'] = weather_encoder.fit_transform(df['Weather'])

# 🔹 Sélection des features utiles
df = df[['ds', 'y', 'Hour', 'Day_of_Week', 'Weather_encoded']]

# 🔹 Entraînement du modèle Prophet
print("📌 Entraînement du modèle Prophet...")
model = Prophet()
model.add_regressor('Hour')
model.add_regressor('Day_of_Week')
model.add_regressor('Weather_encoded')
model.fit(df)

# 🔹 Sauvegarde du modèle et de l’encoder
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/model_bike_sharing.pkl"
ENCODER_PATH = "models/encoder_bike.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(weather_encoder, f)

print(f"✅ Modèle et encoder sauvegardés dans {MODEL_PATH} et {ENCODER_PATH}")