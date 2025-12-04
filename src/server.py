from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chess
import uvicorn
from game_logic import ChessGame
from agent import Agent
import torch

app = FastAPI(title="Chess Bot API", description="API pour jouer contre le bot d'échecs")

# Charger l'agent au démarrage
bot = Agent() 
bot.net.eval()

class MoveRequest(BaseModel):
    fen: str

class MoveResponse(BaseModel):
    move: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Chess Bot API is running"}

@app.post("/predict", response_model=MoveResponse)
def predict_move(request: MoveRequest):
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    if board.is_game_over():
         raise HTTPException(status_code=400, detail="Game is already over")

    game = ChessGame()
    game.board = board
    
    # Si c'est aux Blancs de jouer (board.turn = True), le bot doit maximiser.
    bot.color = board.turn
    
    # On force epsilon à 0 pour jouer le meilleur coup
    original_epsilon = bot.epsilon
    bot.epsilon = 0.0 
    
    try:
        move = bot.select_move(game)
    finally:
        bot.epsilon = original_epsilon

    if move is None:
         raise HTTPException(status_code=500, detail="Bot could not find a legal move")

    return MoveResponse(move=move.uci())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
