import tkinter as tk
from tkinter import messagebox
import chess
from game_logic import ChessGame
from agent import Agent
import threading
import time

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Bot Low-RAM - Entraînement Visuel")
        
        self.training_delay = 0.001 
        
        self.game = ChessGame()
        self.bot = Agent(color=False) 
        
        self.selected_square = None
        self.square_size = 60
        self.canvas_size = self.square_size * 8
        
        # --- Layout Principal ---
        
        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.stats_label = tk.Label(self.info_frame, text="", font=("Arial", 12, "bold"))
        self.stats_label.pack()
        self.update_stats_label()

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.TOP)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Choix de la couleur
        self.player_color_var = tk.StringVar(value="white")
        tk.Radiobutton(self.btn_frame, text="Jouer Blancs", variable=self.player_color_var, value="white", command=self.set_player_color_and_reset).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.btn_frame, text="Jouer Noirs", variable=self.player_color_var, value="black", command=self.set_player_color_and_reset).pack(side=tk.LEFT, padx=5)

        # Option Turbo
        self.turbo_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(self.btn_frame, text="Mode Turbo (Pas de visuel)", variable=self.turbo_mode).pack(side=tk.LEFT, padx=5)

        tk.Label(self.btn_frame, text="Runs:").pack(side=tk.LEFT, padx=2)
        self.runs_entry = tk.Entry(self.btn_frame, width=5)
        self.runs_entry.insert(0, "100") 
        self.runs_entry.pack(side=tk.LEFT)

        tk.Button(self.btn_frame, text="Go Entraînement", command=self.start_training).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Nouveau Modèle", command=self.reset_model_gui).pack(side=tk.LEFT, padx=5) 
        tk.Button(self.btn_frame, text="Reset Partie", command=self.reset_game).pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(self.btn_frame, text="Tour: Blancs")

        self.piece_symbols = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟'
        }
        
        self.player_color = chess.WHITE 
        self.bot_playing_color = chess.BLACK 
        self.set_player_color_and_reset() 
        self.draw_board()

    def set_player_color_and_reset(self):
        if self.player_color_var.get() == "white":
            self.player_color = chess.WHITE
            self.bot_playing_color = chess.BLACK
            self.bot.color = False 
        else:
            self.player_color = chess.BLACK
            self.bot_playing_color = chess.WHITE
            self.bot.color = True 
        self.reset_game()
        
    def update_stats_label(self):
        stats = self.bot.stats
        epsilon_pct = int(self.bot.epsilon * 100)
        elo = self.bot.elo
        self.stats_label.config(text=f"ELO: {elo} | Total: {stats['wins']}V - {stats['draws']}N - {stats['losses']}D | Exploration: {epsilon_pct}%")

    def draw_board(self, custom_board=None):
        board_to_draw = custom_board if custom_board else self.game.board
        self.canvas.delete("all")
        
        color_light = "#F0F8FF" 
        color_dark = "#779556" 
        
        for r in range(8):
            for c in range(8):
                is_light_square = (r + c) % 2 == 0
                color = color_light if is_light_square else color_dark
                
                x1 = c * self.square_size
                y1 = r * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                square_index = chess.square(c, 7-r)
                piece = board_to_draw.piece_at(square_index)
                
                if piece:
                    txt_color = "#FFFFFF" if piece.color == chess.WHITE else "#000000"
                    symbol = self.piece_symbols.get(piece.symbol(), "?")
                    if piece.color == chess.WHITE:
                        self.canvas.create_text(x1+30+1, y1+30+1, text=symbol, font=("Arial", 32), fill="black")
                    self.canvas.create_text(x1+30, y1+30, text=symbol, font=("Arial", 32), fill=txt_color)
                
                if custom_board is None and self.selected_square == square_index:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FF0000", width=3)

    def on_click(self, event):
        if self.game.board.turn != self.player_color: 
            return 

        col = event.x // self.square_size
        row = event.y // self.square_size
        sq = chess.square(col, 7-row)
        
        if self.selected_square is None:
            piece = self.game.board.piece_at(sq)
            if piece and piece.color == self.player_color: 
                self.selected_square = sq
                self.draw_board()
        else:
            move = chess.Move(self.selected_square, sq)
            piece = self.game.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if chess.square_rank(sq) in [0, 7]:
                    move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
            
            if move in self.game.board.legal_moves:
                self.game.make_move(move)
                self.selected_square = None
                self.draw_board()
                self.root.update()
                if self.game.board.is_game_over():
                    self.check_game_over()
                    self.root.after(1000, self.reset_game) 
                else:
                    self.root.after(50, self.bot_move)
            else:
                self.selected_square = None
                self.draw_board()

    def bot_move(self):
        self.status_label.config(text=f"Tour: Bot ({'Blancs' if self.bot_playing_color == chess.WHITE else 'Noirs'})...")
        self.root.update()
        
        move = self.bot.select_move(self.game)
        if move:
            state = self.game.get_state()
            _, reward, done = self.game.make_move(move)
            next_state = self.game.get_state()
            self.bot.train_step(state, reward, next_state, done)
            self.draw_board()
            self.status_label.config(text=f"Tour: Joueur ({'Blancs' if self.player_color == chess.WHITE else 'Noirs'})")
            
            if self.game.board.is_game_over():
                self.check_game_over()
                self.root.after(1000, self.reset_game) 
        else: 
            if self.game.board.is_game_over():
                self.check_game_over()
                self.root.after(1000, self.reset_game)

    def start_training(self):
        try:
            n_games = int(self.runs_entry.get())
        except ValueError:
            n_games = 10
        
        self.status_label.config(text=f"Entraînement: {n_games} parties...")
        for child in self.btn_frame.winfo_children():
            if isinstance(child, tk.Button) or isinstance(child, tk.Radiobutton) or isinstance(child, tk.Checkbutton): 
                child.config(state=tk.DISABLED)

        threading.Thread(target=self.train_loop, args=(n_games,), daemon=True).start()

    def train_loop(self, n_games):
        print(f"Début entraînement ({'Turbo' if self.turbo_mode.get() else 'Normal'}) pour {n_games} parties...")
        white_bot = Agent(color=True) 
        white_bot.net = self.bot.net 
        
        is_turbo = self.turbo_mode.get()

        for i in range(n_games):
            temp_game = ChessGame()
            move_count = 0
            
            while not temp_game.board.is_game_over() and move_count < 150:
                move_count += 1
                
                # --- Tour Blanc ---
                m1 = white_bot.select_move(temp_game)
                if m1 is None: break
                s1 = temp_game.get_state()
                _, r1, done1 = temp_game.make_move(m1)
                
                if not is_turbo:
                    self.draw_board(custom_board=temp_game.board)
                    self.root.update_idletasks() 
                    time.sleep(self.training_delay)
                
                white_bot.train_step(s1, r1, temp_game.get_state(), done1)
                if temp_game.board.is_game_over(): break
                
                # --- Tour Noir ---
                m2 = self.bot.select_move(temp_game) 
                if m2 is None: break
                s2 = temp_game.get_state()
                _, r2, done2 = temp_game.make_move(m2)
                
                if not is_turbo:
                    self.draw_board(custom_board=temp_game.board)
                    self.root.update_idletasks()
                    time.sleep(self.training_delay)
                
                self.bot.train_step(s2, r2, temp_game.get_state(), done2)
            
            outcome = temp_game.board.outcome()
            result = 0
            if outcome:
                if outcome.winner == chess.WHITE: 
                    self.bot.stats["wins"] += 1
                    result = 1
                elif outcome.winner == chess.BLACK: 
                    self.bot.stats["losses"] += 1
                    result = -1
                else: 
                    self.bot.stats["draws"] += 1
                    result = 0
            else:
                self.bot.stats["draws"] += 1
                result = 0
            
            self.bot.update_adaptive_epsilon(result)
            
            # En mode turbo, on met à jour l'interface seulement toutes les 10 parties pour aller vite
            if not is_turbo or i % 10 == 0:
                self.root.after(0, self.update_stats_label)
            
        print("Entraînement fini.")
        self.bot.save_model() 
        self.root.after(0, lambda: self.status_label.config(text="Prêt à jouer (Modèle Sauvegardé)."))
        self.root.after(0, self.draw_board)
        self.root.after(0, self.enable_buttons)
        self.root.after(0, self.update_stats_label) # Maj finale
        self.root.after(0, lambda: messagebox.showinfo("Entraînement Terminé", f"L'entraînement de {n_games} parties est terminé !\nELO final: {self.bot.elo}"))


    def enable_buttons(self):
        for child in self.btn_frame.winfo_children():
            if isinstance(child, tk.Button) or isinstance(child, tk.Radiobutton) or isinstance(child, tk.Checkbutton):
                child.config(state=tk.NORMAL)

    def reset_game(self):
        self.game.reset()
        self.selected_square = None
        self.draw_board()
        
        if self.game.board.turn == self.player_color:
            self.status_label.config(text=f"Tour: Joueur ({'Blancs' if self.player_color == chess.WHITE else 'Noirs'})")
        else:
            self.status_label.config(text=f"Tour: Bot ({'Blancs' if self.bot_playing_color == chess.WHITE else 'Noirs'})")

        if self.game.board.turn == self.bot_playing_color:
            self.root.after(100, self.bot_move)
        
    def reset_model_gui(self):
        self.bot.reset_model() 
        self.reset_game() 
        self.update_stats_label()
        messagebox.showinfo("Nouveau Modèle", "Le modèle d'IA a été réinitialisé et le fichier chess_model.pth supprimé.")
        
    def check_game_over(self):
        if self.game.board.is_game_over():
            res = self.game.board.outcome()
            winner = "Personne"
            if res.winner == chess.WHITE: winner = "Blancs"
            elif res.winner == chess.BLACK: winner = "Noirs"
            messagebox.showinfo("Fin de partie", f"Gagnant: {winner}")