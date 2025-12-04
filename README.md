# Chess Bot Project

Ce projet est un bot d'échecs basé sur l'apprentissage par renforcement (Deep Q-Learning). Il comprend une interface graphique pour l'entraînement visuel et le jeu, ainsi qu'un mode sans tête (headless) pour l'entraînement sur serveur et une API pour le déploiement.

## Structure du Projet

*   `src/agent.py`: Contient le réseau de neurones (`ChessNet`) et l'algorithme d'apprentissage (`Agent`).
*   `src/game_logic.py`: Gère la logique du jeu d'échecs, la représentation de l'état et la fonction de récompense.
*   `src/gui.py`: Interface graphique utilisateur (Tkinter) pour visualiser les parties et l'entraînement.
*   `src/main.py`: Point d'entrée pour lancer l'application GUI.
*   `src/trainer.py`: Script pour l'entraînement "headless" (sans interface graphique), idéal pour les serveurs.
*   `src/server.py`: Serveur API (FastAPI) pour exposer le bot en tant que service Web.

## Installation

Assurez-vous d'avoir Python installé (recommandé 3.8+).

1.  Clonez le dépôt (si ce n'est pas déjà fait).
2.  Installez les dépendances :

```bash
pip install -r requirements.txt
```

*Note : Pour l'accélération GPU, assurez-vous d'avoir la version de PyTorch compatible avec votre matériel (CUDA pour Nvidia, MPS pour Mac M1/M2).*

## Utilisation

### 1. Interface Graphique (Mode Bureau)

C'est le mode par défaut pour tester et voir le bot apprendre en temps réel.

```bash
python src/main.py
```

*   **Jouer Blancs / Noirs** : Choisissez votre couleur.
*   **Mode Turbo** : Accélère l'entraînement en ne rafraîchissant pas l'affichage à chaque coup.
*   **Go Entraînement** : Lance une session d'auto-apprentissage (Self-Play).

### 2. Entraînement Serveur (Mode Headless)

Pour entraîner le modèle rapidement sur un serveur sans écran.

```bash
python src/trainer.py --games 1000
```

*   `--games` : Nombre de parties d'auto-apprentissage à jouer.
*   Le modèle est sauvegardé automatiquement dans `chess_model.pth`.
*   Un graphique de progression (`training_graph.png`) est généré à la fin.

### 3. API Serveur (Déploiement)

Pour utiliser le bot via une API HTTP (par exemple pour le connecter à une interface Web externe).

```bash
python src/server.py
```

Le serveur démarre sur `http://0.0.0.0:8000`.

**Exemple d'appel API (Obtenir un coup) :**

*   **URL** : `POST /predict`
*   **Body (JSON)** :
    ```json
    {
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }
    ```
*   **Réponse (JSON)** :
    ```json
    {
      "move": "e2e4"
    }
    ```

## Modèle

Le fichier `chess_model.pth` contient les poids du réseau de neurones et l'état de l'agent (ELO, epsilon). Il est partagé entre tous les scripts. Si vous entraînez avec `trainer.py`, le nouveau modèle sera chargé la prochaine fois que vous lancerez `main.py` ou `server.py`.
