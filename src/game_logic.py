import chess
import numpy as np

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_material_score(self):
        """Calcule le score matériel absolu (Blanc - Noir)."""
        score = 0
        for piece_type in self.piece_values:
            score += len(self.board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            score -= len(self.board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        return score

    def make_move(self, move):
        """
        Joue un coup et retourne (nouvel_état, récompense_pour_le_joueur, fini).
        """
        # 1. Score matériel AVANT le coup
        prev_material = self.get_material_score()
        
        turn = self.board.turn 
        
        # Vérifier si le coup est une capture (AGRESSIVITÉ)
        is_capture = self.board.is_capture(move)
        
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            return self.get_state(), -500, True 

        # 2. Score matériel APRÈS le coup
        new_material = self.get_material_score()
        
        material_diff = new_material - prev_material
        if not turn: 
            material_diff = -material_diff 
            
        reward = material_diff 

        # --- RECOMPENSES INTERMEDIAIRES (AGRESSIVITÉ BOOSTÉE) ---
        
        # A. Prime de Capture (Inciter à l'échange et à l'attaque)
        if is_capture:
            reward += 0.5
        
        # B. Bonus Position (Contrôle du Centre)
        to_sq = move.to_square
        if to_sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
            reward += 0.1 
        
        # C. Bonus Mise en Echec (Check) - Augmenté pour être plus menaçant
        if self.board.is_check():
            reward += 5.0 

        # D. Pénalité de "Temps" (Step Penalty) - Augmentée pour forcer la rapidité
        reward -= 0.1

        # 3. Vérification Fin de Partie
        done = False
        if self.board.is_game_over():
            done = True
            outcome = self.board.outcome()
            
            if outcome.winner is True: # Blanc gagne
                reward += 1000 if turn else -1000
            elif outcome.winner is False: # Noir gagne
                reward += 1000 if not turn else -1000
            else: # Nul (Draw)
                reward -= 50 
        
        return self.get_state(), reward, done

    def get_reward(self):
        """Utilisé uniquement si on veut le statut global, mais make_move retourne déjà la récompense précise."""
        return 0 

    def get_state(self):
        """Encode le plateau en une matrice numpy 8x8x12 (bitboards)."""
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
