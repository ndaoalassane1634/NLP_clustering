import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict/"

st.title("Application de Clustering")

user_input = st.text_input("Entrez un texte pour obtenir des recommandations:")

if user_input:
    try:
        response = requests.post(API_URL, json={"text": user_input})
        response.raise_for_status()  # Vérifie si la requête a réussi (code 200)

        result = response.json()
        cluster = result.get('predicted_cluster', 'Non défini')
        recommendations = result.get('recommendations', ["Aucune recommandation disponible"])

        st.write(f"Cluster prédit : {cluster}")
        st.write("Recommandations :")
        st.write(recommendations)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la demande à l'API : {e}")
