from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.pyfunc
import joblib
import mlflow
import logging
import os
import numpy as np
from pydantic import BaseModel

# Importation de la configuration dynamique
import config

# ==============================
# 📌 Configuration de l'API FastAPI
# ==============================
app = FastAPI(title="Sentiment Analysis API", description="API de prédiction des sentiments avec MLflow et sélection dynamique du modèle")

# Activation des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================
# 📌 Configuration MLflow
# ==============================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://projet-7-app-f2apebe4hxawe9eg.westeurope-01.azurewebsites.net")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Classe Pydantic pour l'entrée JSON
class Tweet(BaseModel):
    tweet: str

# Fonction pour charger dynamiquement le modèle selon config.py
def load_model():
    """Charge le modèle MLflow selon la configuration et sauvegarde localement."""
    model_name = config.MODEL_NAME
    model_path = f"models/{model_name}/model"

    try:
        logging.info(f"🔄 Chargement du modèle MLflow : {model_name}...")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        os.makedirs(f"models/{model_name}", exist_ok=True)
        mlflow.pyfunc.save_model(path=model_path, python_model=model)
        logging.info(f"✅ Modèle sauvegardé dans {model_path}")
        return model
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement/sauvegarde du modèle : {e}")
        return None

# Fonction pour charger le tokenizer spécifique au modèle
def load_tokenizer():
    """Charge le tokenizer spécifique au modèle."""
    model_name = config.MODEL_NAME
    tokenizer_path = f"models/{model_name}/tokenizer.pkl"

    try:
        logging.info(f"🔄 Chargement du tokenizer pour {model_name}...")
        return joblib.load(tokenizer_path)
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement du tokenizer : {e}")
        return None

model = load_model()
tokenizer = load_tokenizer()

# ==============================
# 📌 Configuration CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 📌 Endpoints de l'API
# ==============================

@app.get("/health")
async def health_check():
    """Vérifie que l'API fonctionne."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_name": config.MODEL_NAME
    }

@app.post("/predict")
async def predict_sentiment(tweet: Tweet):
    """Prédit le sentiment d'un tweet."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Le modèle ou le tokenizer n'est pas disponible.")

    try:
        transformed_tweet = tokenizer.texts_to_sequences([tweet.tweet])
        prediction = model.predict(transformed_tweet)[0]
        sentiment = "positif" if np.array(prediction).item() >= 0.5 else "négatif"
        logging.info(f"✅ Prédiction réussie : {tweet.tweet} → {sentiment}")
        return {"tweet": tweet.tweet, "sentiment": sentiment}
    except Exception as e:
        logging.error(f"❌ Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la requête.")

@app.post("/update-model")
async def update_model():
    """Met à jour le modèle MLflow en chargeant la dernière version selon la config."""
    global model
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Échec de la mise à jour du modèle.")
    return {"message": f"✅ Modèle '{config.MODEL_NAME}' mis à jour avec succès"}