# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import folium_static
# import pickle
# import datetime

# # 🔹 Charger le fichier des stations et des modèles
# df = pd.read_csv("data/London_Bike_Sharing_Dataset.csv")

# # 🔹 Charger le modèle Prophet
# MODEL_PATH = "models/model_bike_sharing.pkl"
# ENCODER_PATH = "models/encoder_bike.pkl"

# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# with open(ENCODER_PATH, "rb") as f:
#     weather_encoder = pickle.load(f)

# # 🔹 Configuration de l'application Streamlit
# st.set_page_config(page_title="📊 Prédiction des vélos à Londres", layout="wide")
# st.title("🚲 Prédiction de l'affluence des vélos à Londres")

# # 📍 Sélection de la station
# station_filter = st.selectbox("📍 Sélectionnez une station", df["Station"].unique())

# # ⏰ Sélection de l'heure
# time_input = st.time_input("⏰ Sélectionnez une heure")

# # ☁️ Sélection de la météo
# weather_options = list(weather_encoder.classes_)  # Récupérer les valeurs encodées
# weather_selected = st.selectbox("☁️ Sélectionnez la météo", weather_options)

# # 📆 Définir une date moyenne issue du dataset
# date_moyenne = pd.to_datetime("2024-01-01")  # Référence fixe pour Prophet

# # 🕒 Fusionner date moyenne et heure pour la prédiction
# full_datetime = pd.to_datetime(f"{date_moyenne.date()} {time_input}")

# if st.button("📈 Prédire le nombre de vélos disponibles"):
#     # 🔹 Récupérer les infos de la station
#     station_info = df[df["Station"] == station_filter].iloc[0]
#     lat, lon = station_info["Latitude"], station_info["Longitude"]

#     # 🔹 Transformer la météo avec l'encoder
#     if weather_selected not in weather_encoder.classes_:
#         st.write(f"⚠️ Météo inconnue: {weather_selected}. Utilisation de 'Ensoleillé' par défaut.")
#         weather_selected = "Ensoleillé"

#     encoded_weather = weather_encoder.transform([weather_selected])[0]  # 🔥 Correction ici

#     # 🔹 Construire la DataFrame pour la prédiction
#     future = pd.DataFrame({
#         "ds": [full_datetime],
#         "Hour": [full_datetime.hour],
#         "Day_of_Week": [full_datetime.dayofweek],
#         "Weather_encoded": [encoded_weather]
#     })

#     # 🔹 Afficher les données envoyées au modèle
#     st.write("### 📊 Données envoyées au modèle :")
#     st.write(future)

#     # 🔹 Prédiction
#     forecast = model.predict(future)
#     predicted_bikes = max(0, forecast["yhat"].values[0])  # Assurer que le résultat est positif

#     # 🔹 Affichage du résultat
#     st.write(f"📊 **Prédiction : {predicted_bikes:.0f} vélos disponibles à {station_filter} à {time_input}.**")

#     # 🔹 Afficher la carte centrée sur la station
#     m = folium.Map(location=[lat, lon], zoom_start=15)
#     folium.Marker(
#         location=[lat, lon],
#         popup=station_filter,
#         tooltip=station_filter,
#         icon=folium.Icon(color="red", icon="info-sign")
#     ).add_to(m)
#     folium_static(m)
# #code pour streamlit non suivit
import streamlit as st
import pandas as pd
import folium
import requests  # 🔥 Pour envoyer les requêtes API
from streamlit_folium import folium_static
import pickle

# 🔹 Charger le fichier des stations
df = pd.read_csv("data/df_index_station.csv")

# 🔹 Configuration de l'application Streamlit
st.set_page_config(page_title="📊 Prédiction des vélos à Londres", layout="wide")
st.title("🚲 Prédiction de l'affluence des vélos à Londres")

# 📍 Sélection de la station
station_filter = st.selectbox("📍 Sélectionnez une station", df["Station"].unique())

# ⏰ Sélection de l'heure
time_input = st.time_input("⏰ Sélectionnez une heure")

# ☁️ Sélection de la météo
weather_options = ["Ensoleillé", "Pluvieux", "Orageux", "Nuageux"]  # 🔥 Renseigner les valeurs directement
weather_selected = st.selectbox("☁️ Sélectionnez la météo", weather_options)

# 📆 Définir une date moyenne issue du dataset
date_moyenne = pd.to_datetime("2024-01-01")  # Référence fixe pour la prédiction
full_datetime = pd.to_datetime(f"{date_moyenne.date()} {time_input}")

if st.button("📈 Prédire le nombre de vélos disponibles"):
    # 🔹 Récupérer les infos de la station
    station_info = df[df["Station"] == station_filter].iloc[0]
    lat, lon = station_info["Latitude"], station_info["Longitude"]

    # 🔹 Préparer la requête pour FastAPI
    api_url = "http://52.47.103.192:8000/predict/" # pour del=loyment local : bike-api
    data = {
        "station": station_filter,
        "hour": full_datetime.hour,
        "day_of_week": full_datetime.dayofweek,
        "weather": weather_selected
    }

    # 🔥 Envoyer la requête à FastAPI
    response = requests.post(api_url, json=data)

    if response.status_code == 200:
        result = response.json()
        predicted_bikes = result["predicted_bikes"]

        # 🔹 Affichage du résultat
        st.write(f"📊 **Prédiction : {predicted_bikes} vélos disponibles à {station_filter} à {time_input}.**")

        # 🔹 Afficher la carte centrée sur la station
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker(
            location=[lat, lon],
            popup=station_filter,
            tooltip=station_filter,
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        folium_static(m)

    else:
        st.error("❌ Erreur lors de la prédiction, vérifiez FastAPI.")