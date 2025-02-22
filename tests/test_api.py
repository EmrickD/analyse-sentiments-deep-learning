import pytest
import requests
import logging

# ==============================
# 📌 Configuration des tests
# ==============================
API_URL = "http://localhost:8000"

# Activation des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================
# 📌 Vérification si l'API est bien en ligne
# ==============================
@pytest.fixture(scope="session", autouse=True)
def check_api_running():
    """Vérifie si l'API est bien en ligne avant d'exécuter les tests."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200, "❌ L'API ne répond pas !"
        logging.info("✅ API accessible, début des tests...")
    except requests.exceptions.RequestException as e:
        pytest.exit(f"❌ Impossible de contacter l'API : {e}")

# ==============================
# 📌 Test de l'endpoint /health
# ==============================
def test_health():
    """Teste si l'endpoint /health renvoie bien un statut OK."""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "running", "❌ L'API ne semble pas active."
    assert "model_loaded" in data, "❌ Clé 'model_loaded' absente de la réponse."
    assert "tokenizer_loaded" in data, "❌ Clé 'tokenizer_loaded' absente de la réponse."

    logging.info("✅ Test /health : API en ligne et modèle chargé")

# ==============================
# 📌 Tests de prédiction avec plusieurs tweets
# ==============================
@pytest.mark.parametrize("tweet", [
    ("J'adore ce produit, il est génial!", "positif"),
    ("Ce service est terrible, je suis déçu.", "négatif"),
    ("C'est une belle journée ensoleillée.", None),  # Cas neutre possible
    ("Horrible expérience, je ne reviendrai plus !", "négatif"),
    ("Fantastique ! Excellent service client !", "positif"),
])
def test_predict(tweet):
    """Teste la prédiction d'un tweet."""
    response = requests.post(f"{API_URL}/predict", json={"tweet": tweet[0]})
    assert response.status_code == 200
    assert "sentiment" in response.json(), "❌ Clé 'sentiment' absente de la réponse."
    sentiment = response.json()["sentiment"]

    if tweet[1]:  # Vérifier si un sentiment attendu est défini
        assert sentiment in ["positif", "négatif"], f"❌ Sentiment inattendu : {sentiment}"
    
    logging.info(f"✅ Test prédiction : '{tweet[0]}' → {sentiment}")

# ==============================
# 📌 Test d'entrée invalide (absence de champ 'tweet')
# ==============================
def test_predict_invalid_input():
    """Teste une requête invalide sans champ 'tweet'."""
    response = requests.post(f"{API_URL}/predict", json={})  # JSON vide
    assert response.status_code == 422, "❌ L'API aurait dû renvoyer une erreur 422."
    logging.info("✅ Test requête invalide : erreur 422 bien renvoyée.")

# ==============================
# 📌 Test de mise à jour du modèle
# ==============================
def test_update_model():
    """Teste la mise à jour du modèle via l'API."""
    response = requests.post(f"{API_URL}/update-model")
    assert response.status_code == 200
    assert response.json()["message"] == "✅ Modèle '" + response.json().get("model_name", "") + "' mis à jour avec succès", "❌ Message inattendu après mise à jour."
    logging.info("✅ Test mise à jour du modèle réussi")

# ==============================
# 📌 Exécution des tests
# ==============================
if __name__ == "__main__":
    pytest.main()
