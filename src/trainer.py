import argparse
import time
import chess
from game_logic import ChessGame
from agent import Agent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training_results(bot):
    try:
        history = bot.elo_history
        if not history or len(history) < 2: 
            return None

        plt.figure(figsize=(10, 5))
        plt.plot(history, color='blue', linewidth=2, label='ELO')
        
        if len(history) > 50:
            window = 50
            moving_avg = [sum(history[max(0, i-window):i+1])/len(history[max(0, i-window):i+1]) for i in range(len(history))]
            plt.plot(moving_avg, color='red', linestyle='dashed', label='Moyenne (50)')

        plt.title(f"Progression de l'Apprentissage (ELO Actuel: {bot.elo})")
        plt.xlabel("Parties Jouées")
        plt.ylabel("Score ELO Estimé")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = "training_graph.png"
        plt.savefig(filename)
        plt.close()
        print(f"Graphique sauvegardé sous {filename}")
    except Exception as e:
        print(f"Erreur graphique: {e}")

def train_loop(n_games=100):
    print(f"Début entraînement pour {n_games} parties...")
    
    # Bot principal (qui apprend)
    bot = Agent(color=False) # Noir par défaut ici, mais peu importe pour le self-play
    
    # Adversaire (copie du bot principal pour self-play)
    white_bot = Agent(color=True)
    white_bot.net = bot.net 
    
    for i in range(n_games):
        temp_game = ChessGame()
        move_count = 0
        
        while not temp_game.board.is_game_over() and move_count < 150:
            move_count += 1
            
            # --- Tour des Blancs ---
            m1 = white_bot.select_move(temp_game)
            if m1 is None: break
            s1 = temp_game.get_state()
            _, r1, done1 = temp_game.make_move(m1)
            
            # L'adversaire apprend aussi (c'est le même réseau)
            white_bot.train_step(s1, r1, temp_game.get_state(), done1)
            
            if temp_game.board.is_game_over(): break
            
            # --- Tour des Noirs ---
            m2 = bot.select_move(temp_game) 
            if m2 is None: break
            s2 = temp_game.get_state()
            _, r2, done2 = temp_game.make_move(m2)
            
            bot.train_step(s2, r2, temp_game.get_state(), done2)
        
        # Fin de partie
        outcome = temp_game.board.outcome()
        result = 0
        if outcome:
            if outcome.winner == chess.WHITE: 
                bot.stats["wins"] += 1 
                result = -1 
            elif outcome.winner == chess.BLACK: 
                bot.stats["losses"] += 1 
                result = 1
            else: 
                bot.stats["draws"] += 1
                result = 0
        else:
            bot.stats["draws"] += 1
            result = 0
        
        bot.update_adaptive_epsilon(result)
        
        if (i + 1) % 10 == 0:
             print(f"Partie {i+1}/{n_games} terminée. ELO: {bot.elo}, Epsilon: {bot.epsilon:.3f}")
        
    print("Entraînement fini.")
    bot.save_model() 
    plot_training_results(bot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner le bot d'échecs sans GUI")
    parser.add_argument("--games", type=int, default=100, help="Nombre de parties à jouer")
    args = parser.parse_args()
    
    train_loop(args.games)
