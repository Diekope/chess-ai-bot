import tkinter as tk
from gui import ChessGUI

if __name__ == "__main__":
    root = tk.Tk()
    # Centrer la fenêtre (optionnel mais sympa)
    # root.eval('tk::PlaceWindow . center') 
    app = ChessGUI(root)
    root.mainloop()
