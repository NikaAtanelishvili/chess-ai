import sys
import os

from PyQt5.QtWidgets import QApplication, QStackedWidget

from src.chess_board import ChessBoard
from start_menu import StartMenu

class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Chess")
        self.setFixedSize(600, 650)

        self.start_menu = StartMenu()
        self.start_menu.start_game.connect(self.launch_game)

        self.addWidget(self.start_menu)
        self.setCurrentWidget(self.start_menu)

    def launch_game(self, color, elo, mode):
        self.chess_board = ChessBoard(color, elo, mode)
        self.addWidget(self.chess_board)
        self.setCurrentWidget(self.chess_board)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())