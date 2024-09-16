from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle et le vectoriseur
model = joblib.load('dbscan_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
X_tfidf = joblib.load('X_tfidf.joblib')

# Définir les recommandations par cluster
recommandations_par_cluster = {
    4: [
        "Découvrez les dernières offres sur les voitures Chevrolet.",
        "Renault: Les meilleurs modèles à des prix compétitifs.",
        "Offre spéciale sur les voitures Ford cette semaine.",
        "Top 10 des voitures d'occasion fiables."
    ],
    1: [
        "Achetez le dernier équipement de basketball au meilleur prix.",
        "Revivez les meilleurs moments de la saison NBA 2024.",
        "Obtenez des billets pour les prochains matchs de basketball.",
        "Guide complet pour choisir le meilleur ballon de basketball."
    ],
    3: [
        "Lisez les dernières nouvelles sur Donald Trump.",
        "Analyse des politiques récentes de Donald Trump.",
        "Documentaire sur la présidence de Donald Trump.",
        "Livres recommandés sur Donald Trump et son impact."
    ],
    0: [
        "Découvrez nos nouveaux tapis de yoga Manduka.",
        "Abonnement à des cours de yoga en ligne.",
        "Les meilleurs vêtements pour pratiquer le yoga confortablement.",
        "Conseils pour améliorer votre pratique du yoga à la maison."
    ],
    2: [
        "Apprenez Python pour le marketing digital : Guide complet pour débutants.",
        "Cours en ligne pour maîtriser Python dans le marketing.",
        "Formation avancée en Python pour les professionnels du marketing.",
        "Comment Python peut transformer vos stratégies de marketing digital.",
        "Les meilleures pratiques pour appliquer le machine learning dans le marketing.",
        "Études de cas : Comment les marketeurs utilisent Python pour améliorer leurs campagnes.",
        "Top 5 des bibliothèques Python pour le marketing digital.",
        "Outils Python recommandés pour l'analyse des données marketing.",
        "Ressources en Python pour automatiser vos tâches marketing.",
        "Étude de cas : Augmenter les conversions grâce à l'analyse des données marketing avec Python.",
        "Réussites dans le marketing digital : Comment Python a fait la différence.",
        "Les derniers articles sur l'application de Python dans le marketing digital.",
        "Discussions et conseils sur les forums de marketing numérique pour l'utilisation de Python."
    ]
}

# Définir le modèle de données pour l'entrée utilisateur
class TextInput(BaseModel):
    text: str

# Fonction pour prédire le cluster d'un nouveau point
def predict_dbscan(model, X_new, X_train):
    # Trouver le point le plus proche dans le jeu de données d'entraînement
    closest_point_index, _ = pairwise_distances_argmin_min(X_new, X_train)
    # Assigner le même cluster que celui du point le plus proche
    return model.labels_[closest_point_index]

# Route pour la prédiction
@app.post("/predict/")
def predict_cluster(text_input: TextInput):
    text = text_input.text

    # Vectoriser le texte de l'utilisateur
    X = vectorizer.transform([text])

    # Prévoir le cluster en utilisant la fonction définie
    cluster = predict_dbscan(model, X.toarray(), X_tfidf.toarray())[0]

    # Obtenir les recommandations correspondantes
    recommendations = recommandations_par_cluster.get(cluster, ["Aucune recommandation disponible"])

    # Créer une réponse
    response = {
        "text": text,
        "predicted_cluster": int(cluster),
        "recommendations": recommendations
    }

    return response
