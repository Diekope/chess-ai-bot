# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier les dépendances et les installer
# On le fait avant de copier le code pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du projet (code source, modèle, etc.)
COPY . .

# Exposer le port 8000 (celui de FastAPI)
EXPOSE 8000

# Commande de démarrage : lancer le serveur API
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
