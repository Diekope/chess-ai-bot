import tkinter as tk
from tkinter import messagebox, simpledialog
import chess
import requests
import threading
import itertools
import time

class OnlineChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Bot Online - Client API")
        
        self.api_url = "https://chess-ai-bot-5tni.onrender.com/predict"

        self.board = chess.Board()
        
        self.selected_square = None
        self.square_size = 60
        self.canvas_size = self.square_size * 8
        
        # --- Layout ---
        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.status_label = tk.Label(self.info_frame, text="Connecté à l'API", font=("Arial", 12))
        self.status_label.pack()

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.TOP)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        tk.Button(self.btn_frame, text="Nouvelle Partie", command=self.reset_game).pack(side=tk.LEFT, padx=5)
        
        self.piece_symbols = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟'
        }
        
        # Variables d'état
        self.is_waiting_for_bot = False
        self.loading_animation_id = None
        self.loading_spinner = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])

        self.player_color = chess.WHITE # Humain joue Blanc
        self.draw_board()

    def draw_board(self):
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
                piece = self.board.piece_at(square_index)
                
                if piece:
                    txt_color = "#FFFFFF" if piece.color == chess.WHITE else "#000000"
                    symbol = self.piece_symbols.get(piece.symbol(), "?")
                    # Petit contour pour visibilité
                    if piece.color == chess.WHITE:
                        self.canvas.create_text(x1+30+1, y1+30+1, text=symbol, font=("Arial", 32), fill="black")
                    self.canvas.create_text(x1+30, y1+30, text=symbol, font=("Arial", 32), fill=txt_color)
                
                if self.selected_square == square_index:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FF0000", width=3)

        # Overlay de chargement si le bot réfléchit
        if self.is_waiting_for_bot:
            self.draw_loading_overlay()

    def draw_loading_overlay(self):
        # Fond semi-transparent
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#000000", stipple="gray50")
        
        # Texte animé
        char = next(self.loading_spinner)
        text = f"{char} Le Bot réfléchit..."
        
        self.canvas.create_text(self.canvas_size//2, self.canvas_size//2, text=text, font=("Arial", 24, "bold"), fill="white")
        
        # Rappeler cette fonction dans 100ms pour animer
        self.loading_animation_id = self.root.after(100, self.draw_board)

    def on_click(self, event):
        if self.is_waiting_for_bot or self.board.is_game_over():
            return

        col = event.x // self.square_size
        row = event.y // self.square_size
        sq = chess.square(col, 7-row)
        
        if self.selected_square is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.player_color: 
                self.selected_square = sq
                self.draw_board()
        else:
            move = chess.Move(self.selected_square, sq)
            
            # Promotion automatique Dame pour simplifier
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if chess.square_rank(sq) in [0, 7]:
                    move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                
                if not self.board.is_game_over():
                    self.request_bot_move()
                else:
                    self.check_game_over()
            else:
                self.selected_square = None
                self.draw_board()

    def request_bot_move(self):
        self.is_waiting_for_bot = True
        self.status_label.config(text="Bot en train de réfléchir (ou réveil serveur)...")
        self.draw_board() # Lance l'animation
        
        # Requête dans un thread pour ne pas geler l'UI
        threading.Thread(target=self.fetch_move_thread, daemon=True).start()

    def fetch_move_thread(self):
        fen = self.board.fen()
        try:
            response = requests.post(self.api_url, json={"fen": fen}, timeout=120)
            if response.status_code == 200:
                data = response.json()
                move_uci = data.get("move")
                self.root.after(0, lambda: self.apply_bot_move(move_uci))
            else:
                self.root.after(0, lambda: self.show_error(f"Erreur serveur: {response.status_code}"))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Erreur connexion: {e}"))

    def apply_bot_move(self, move_uci):
        # Arrêter l'animation
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None
        
        self.is_waiting_for_bot = False
        
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.status_label.config(text="À vous de jouer")
            else:
                messagebox.showerror("Erreur", f"Le bot a envoyé un coup illégal : {move_uci}")
        except Exception:
             messagebox.showerror("Erreur", f"Format de coup invalide reçu : {move_uci}")
             
        self.draw_board()
        self.check_game_over()

    def show_error(self, msg):
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None
        self.is_waiting_for_bot = False
        self.draw_board()
        messagebox.showerror("Erreur API", msg)

    def check_game_over(self):
        if self.board.is_game_over():
            res = self.board.outcome()
            winner = "Personne"
            if res.winner == chess.WHITE: winner = "Blancs"
            elif res.winner == chess.BLACK: winner = "Noirs"
            messagebox.showinfo("Fin de partie", f"Gagnant: {winner}")

    def reset_game(self):
        self.board.reset()
        self.selected_square = None
        self.is_waiting_for_bot = False
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None
        self.draw_board()

if __name__ == "__main__":
    root = tk.Tk()
    app = OnlineChessGUI(root)
    root.mainloop()
