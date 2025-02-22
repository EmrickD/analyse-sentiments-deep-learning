from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# ==============================
# 📌 Configuration de l'API FastAPI
# ==============================
app = FastAPI(title="Sentiment Analysis API", description="API de prédiction des sentiments avec sélection dynamique du modèle")

# Activation des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================
# 📌 Configuration pour sélectionner le modèle
# ==============================

# Modèles disponibles :
# - "regression_logistique"
# - "lstm_w2v"
# - "lstm_glove"
# - "bert"

MODEL_NAME = "lstm_w2v"  # Modifie cette ligne pour choisir ton modèle