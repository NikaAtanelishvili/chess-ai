import os
import random
import chess
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import QSize, QTimer


class ChessBoard(QWidget):
    def __init__(self, player_color="white", bot_elo=1200, mode="vs_bot"):
        super().__init__()
        self.setWindowTitle("Chess Game")
        self.player_color = player_color.lower()
        self.bot_elo = bot_elo
        self.mode = mode

        self.board = chess.Board()
        self.board_buttons = []
        self.selected_square = None
        self.timer_seconds = 0

        self.initUI()

        # თუ ბოტი თეთრია, უტრიალებს პირველ სვლას
        if self.mode == "vs_bot" and self.player_color == "black":
            QTimer.singleShot(500, self.make_bot_move)

    def initUI(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Timer და Restart ღილაკი
        top_bar = QHBoxLayout()
        self.timer_label = QLabel("Time: 0:00")
        self.timer_label.setFont(QFont("Arial", 14))
        self.restart_button = QPushButton("↻ Play Again")
        self.restart_button.clicked.connect(self.restart_game)
        top_bar.addWidget(self.timer_label)
        top_bar.addStretch()
        top_bar.addWidget(self.restart_button)
        main_layout.addLayout(top_bar)

        # დაფის გრაიდი
        self.grid = QGridLayout()
        self.grid.setSpacing(0)
        main_layout.addLayout(self.grid)
        print('aaaaaaaaaaa')
        self.load_icons()
        self.create_board()
        self.start_timer()
        self.update_board()

    def load_icons(self):
        base_path = "assets"
        print(os.path.join(base_path, "b_rook_png_128px.png"))

        self.icons = {
            "r": os.path.join(base_path, "b_rook_png_128px.png"),
            "n": os.path.join(base_path, "b_knight_png_128px.png"),
            "b": os.path.join(base_path, "b_bishop_png_128px.png"),
            "q": os.path.join(base_path, "b_queen_png_128px.png"),
            "k": os.path.join(base_path, "b_king_png_128px.png"),
            "p": os.path.join(base_path, "b_pawn_png_128px.png"),
            "R": os.path.join(base_path, "w_rook_png_128px.png"),
            "N": os.path.join(base_path, "w_knight_png_128px.png"),
            "B": os.path.join(base_path, "w_bishop_png_128px.png"),
            "Q": os.path.join(base_path, "w_queen_png_128px.png"),
            "K": os.path.join(base_path, "w_king_png_128px.png"),
            "P": os.path.join(base_path, "w_pawn_png_128px.png"),
        }

    def create_board(self):
        self.board_buttons = [[None] * 8 for _ in range(8)]
        for row in range(8):
            for col in range(8):
                button = QPushButton()
                button.setFixedSize(QSize(64, 64))
                color = "#EEEED2" if (row + col) % 2 == 0 else "#769656"
                button.setStyleSheet(f"background-color: {color}; border: none;")

                display_row = row if self.player_color == "white" else 7 - row
                button.clicked.connect(lambda checked, r=display_row, c=col: self.handle_click(r, c))
                self.grid.addWidget(button, display_row, col)
                self.board_buttons[display_row][col] = button

    def handle_click(self, row, col):
        if self.board.is_game_over():
            return

        square = chess.square(col, 7 - row if self.player_color == "white" else row)

        # 2 მოთამაშის რეჟიმი
        if self.mode == "2_players":
            piece = self.board.piece_at(square)
            if self.selected_square is None:
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
            else:
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.update_board()
                    self.selected_square = None
                else:
                    self.selected_square = None
            return

        # vs_bot რეჟიმი
        if self.mode == "vs_bot":
            if self.board.turn != (self.player_color == "white"):
                return

            piece = self.board.piece_at(square)
            if self.selected_square is None:
                if piece and piece.color == (self.player_color == "white"):
                    self.selected_square = square
            else:
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.update_board()
                    self.selected_square = None
                    if not self.board.is_game_over():
                        QTimer.singleShot(300, self.make_bot_move)
                else:
                    self.selected_square = None

    def make_bot_move(self):
        if self.mode != "vs_bot" or self.board.is_game_over():
            return

        if self.board.turn == (self.player_color == "white"):
            return

        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            self.board.push(move)
            self.update_board()

    def update_board(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            row = 7 - chess.square_rank(square) if self.player_color == "white" else chess.square_rank(square)
            col = chess.square_file(square)
            button = self.board_buttons[row][col]

            if piece:
                icon_path = self.icons[piece.symbol()]
                button.setIcon(QIcon(QPixmap(icon_path)))
                button.setIconSize(QSize(50, 50))
            else:
                button.setIcon(QIcon())

            # საზღვრის გამოჩენა თუ უჯრა არჩეულია
            if self.selected_square == square:
                button.setStyleSheet(button.styleSheet() + "border: 3px solid red;")
            else:
                color = "#EEEED2" if (row + col) % 2 == 0 else "#769656"
                button.setStyleSheet(f"background-color: {color}; border: none;")

    def restart_game(self):
        self.board.reset()
        self.selected_square = None
        self.timer_seconds = 0
        self.timer_label.setText("Time: 0:00")
        self.update_board()
        if self.mode == "vs_bot" and self.player_color == "black":
            QTimer.singleShot(500, self.make_bot_move)

    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)

    def update_timer(self):
        self.timer_seconds += 1
        minutes = self.timer_seconds // 60
        seconds = self.timer_seconds % 60
        self.timer_label.setText(f"Time: {minutes}:{seconds:02d}")