import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Entrée: 12 canaux (pièces) x 8 x 8
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 32 filtres * 8 * 8 = 2048 neurones
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1) 

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

class Agent:
    def __init__(self, color=True):
        # DÉTECTION DU GPU APPLE (MPS) OU CUDA
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Accélération Apple Metal (MPS) activée !")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Accélération CUDA activée !")
        else:
            self.device = torch.device("cpu")
            print("⚙️ Mode CPU Standard")

        self.net = ChessNet().to(self.device) # On envoie le réseau sur le GPU
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.color = color 
        self.epsilon = 1.0 
        self.gamma = 0.9
        self.model_path = "chess_model.pth"
        self.stats = {"wins": 0, "draws": 0, "losses": 0} 
        self.games_since_last_win = 0 
        self.elo = 400 
        
        self.load_model()

    def select_move(self, game):
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        # Exploration aléatoire
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        best_move = None
        best_value = -float('inf') if self.color else float('inf')

        # On teste chaque coup
        for move in legal_moves:
            game.board.push(move)
            state = game.get_state()
            # Envoi du tenseur sur le GPU
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                value = self.net(state_tensor).item() # .item() ramène la valeur sur le CPU pour la comparaison
            
            game.board.pop()

            if self.color: # Blanc veut maximiser
                if value > best_value:
                    best_value = value
                    best_move = move
            else: # Noir veut minimiser
                if value < best_value:
                    best_value = value
                    best_move = move
        
        return best_move if best_move else random.choice(legal_moves)

    def train_step(self, state, reward, next_state, done):
        self.optimizer.zero_grad()
        
        # Envoi des données sur le GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device)

        target = reward_tensor
        if not done:
            with torch.no_grad():
                target = reward_tensor + self.gamma * self.net(next_state_tensor)
        
        prediction = self.net(state_tensor)
        loss = self.loss_fn(prediction, target)
        loss.backward()
        self.optimizer.step()

    def update_adaptive_epsilon(self, result):
        if result == 1: # VICTOIRE
            self.games_since_last_win = 0
            self.elo += 10 
            if self.epsilon > 0.05:
                self.epsilon *= 0.99 
        elif result == 0: # NUL
            self.games_since_last_win += 1
            self.elo -= 2 
            if self.epsilon > 0.05:
                self.epsilon *= 0.9999
        else: # DEFAITE
            self.games_since_last_win += 1
            self.elo -= 5 
            if self.epsilon > 0.05:
                self.epsilon *= 0.9999
        
        if self.elo < 100: self.elo = 100
        
        if self.games_since_last_win > 100 and self.epsilon < 0.5:
            # print("⚠️ STAGNATION DETECTÉE") # Commenté pour réduire le bruit en turbo
            self.epsilon = 0.5
            self.games_since_last_win = 0 

    def save_model(self):
        # On sauvegarde toujours sur CPU pour compatibilité
        torch.save({
            'model_state_dict': self.net.to("cpu").state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': self.stats,
            'games_since_last_win': self.games_since_last_win,
            'elo': self.elo
        }, self.model_path)
        self.net.to(self.device) # On renvoie sur le GPU après sauvegarde
        print(f"Modèle sauvegardé. Epsilon={self.epsilon:.4f}, ELO={self.elo}")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.stats = checkpoint.get('stats', {"wins": 0, "draws": 0, "losses": 0})
                self.games_since_last_win = checkpoint.get('games_since_last_win', 0)
                self.elo = checkpoint.get('elo', 400)
                print(f"Modèle chargé sur {self.device} ! ELO: {self.elo}")
            except Exception as e:
                print(f"Erreur chargement modèle: {e}")
        else:
            print("Aucun modèle sauvegardé trouvé. Démarrage à partir d'un modèle vierge.")

    def reset_model(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            print("Fichier modèle supprimé.")
        
        self.net = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.epsilon = 1.0 
        self.stats = {"wins": 0, "draws": 0, "losses": 0}
        self.games_since_last_win = 0
        self.elo = 400
        print("Modèle réinitialisé.")