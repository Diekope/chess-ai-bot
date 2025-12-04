import chess
import numpy as np

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        # Valeurs standards ajustées pour l'IA
        self.piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.05, # Les cavaliers sont précieux en début de jeu
            chess.BISHOP: 3.33, # Les fous sont forts en fin de jeu ouvert
            chess.ROOK: 5.63,
            chess.QUEEN: 9.5,
            chess.KING: 0 # Le roi ne vaut rien "matériellement", sa vie est le jeu
        }

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_material_score(self):
        score = 0
        for piece_type in self.piece_values:
            score += len(self.board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            score -= len(self.board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        return score

    def get_positional_bonus(self, move, piece):
        bonus = 0
        row, col = divmod(move.to_square, 8)
        if not self.board.turn: # Noir
            row = 7 - row

        # Centre
        if 3 <= row <= 4 and 3 <= col <= 4:
            bonus += 0.2
        elif 2 <= row <= 5 and 2 <= col <= 5: 
            bonus += 0.05

        # Pièces spécifiques
        if piece.piece_type == chess.KNIGHT:
            if col == 0 or col == 7: bonus -= 0.3 # Pénalité bords
        
        elif piece.piece_type == chess.PAWN:
            bonus += 0.05 * row # Plus il avance, mieux c'est
            # Bonus structure pion
            if row == 1: bonus += 0.1 # Ne pas bouger les pions devant le roi trop tôt sans raison

        return bonus

    def make_move(self, move):
        # 1. Analyse AVANT le coup
        prev_material = self.get_material_score()
        turn = self.board.turn 
        piece = self.board.piece_at(move.from_square)
        fullmove = self.board.fullmove_number
        
        # Identification de la cible (pour le trading intelligent)
        # Attention: move.to_square peut être vide (déplacement) ou contenir une pièce ennemie
        target_piece = self.board.piece_at(move.to_square) 
        is_capture = target_piece is not None or self.board.is_en_passant(move)

        # Est-ce que la case d'arrivée est dangereuse ? (Contrôlée par l'adversaire)
        # Note: On vérifie si l'adversaire (not turn) attaque la case cible
        is_dangerous_square = self.board.is_attacked_by(not turn, move.to_square)

        # --- JOUER LE COUP ---
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            return self.get_state(), -1000, True 

        # 2. Score Matériel de base
        new_material = self.get_material_score()
        material_diff = new_material - prev_material
        if not turn: material_diff = -material_diff 
        
        reward = material_diff 

        # --- INTELLIGENCE DE L'ÉCHANGE (SMART TRADING) ---
        
        if is_capture and target_piece:
            val_attacker = self.piece_values.get(piece.piece_type, 0)
            val_victim = self.piece_values.get(target_piece.piece_type, 0)
            
            # Bonus "David contre Goliath"
            # Si je mange une pièce plus forte que la mienne (ex: Pion mange Tour)
            if val_victim > val_attacker:
                diff = val_victim - val_attacker
                reward += diff * 0.5 # Gros bonus immédiat pour l'exploit
                
            # Pénalité "Sacrifice Inutile"
            # Si je mange, que la case est dangereuse, et que ma pièce vaut plus cher
            # (ex: Ma Dame mange un Pion protégé)
            if is_dangerous_square and val_attacker > val_victim:
                # L'échange est mauvais par essence, on pénalise
                reward -= (val_attacker - val_victim) * 0.2

        # --- STRATÉGIE GÉNÉRALE ---

        # A. Ouverture (Développement)
        if fullmove <= 10:
            if piece.piece_type == chess.PAWN and move.to_square in [27, 28, 35, 36]: # Centre e4/d4...
                reward += 0.5
            if piece.piece_type == chess.KNIGHT:
                reward += 0.3

        # B. Bonus Positionnel
        reward += self.get_positional_bonus(move, piece)

        # C. Menace (Echec)
        if self.board.is_check(): 
            reward += 1.0

        # D. Efficacité
        reward -= 0.02 

        # 3. Fin de Partie
        done = False
        if self.board.is_game_over():
            done = True
            outcome = self.board.outcome()
            
            if outcome.winner is True: 
                reward += 2000 if turn else -2000
            elif outcome.winner is False: 
                reward += 2000 if not turn else -2000
            else: 
                # Nul (Draw)
                reward -= 100 
        
        return self.get_state(), reward, done

    def get_state(self):
        layers = []
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for color in [chess.WHITE, chess.BLACK]:
                mask = self.board.pieces_mask(piece_type, color)
                layer = np.zeros(64, dtype=np.float32)
                for i in range(64):
                    if mask & (1 << i):
                        layer[i] = 1.0
                layers.append(layer.reshape(8, 8))
        return np.stack(layers)

    def get_legal_moves(self):
        return list(self.board.legal_moves)
