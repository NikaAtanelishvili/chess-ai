from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

class StartMenu(QWidget):
    # ახლა 3 პარამეტრით: ფერი, elo, რეჟიმი
    start_game = pyqtSignal(str, int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess - Start Menu")
        self.setFixedSize(300, 250)

        layout = QVBoxLayout()

        self.color_label = QLabel("Choose your color:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(["White", "Black"])

        self.elo_label = QLabel("Choose bot's ELO:")
        self.elo_combo = QComboBox()
        self.elo_combo.addItems(["800", "1200", "1600", "2000"])

        self.mode_label = QLabel("Choose mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["vs_bot", "2_players"])

        self.start_btn = QPushButton("Start ▶")
        self.start_btn.clicked.connect(self.handle_start)

        layout.addWidget(self.color_label)
        layout.addWidget(self.color_combo)
        layout.addWidget(self.elo_label)
        layout.addWidget(self.elo_combo)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

        # თუ 2 მოთამაშე აირჩიე, ელო სელექცია დაიფაროს
        self.mode_combo.currentTextChanged.connect(self.toggle_elo_visibility)
        self.toggle_elo_visibility(self.mode_combo.currentText())

    def toggle_elo_visibility(self, mode):
        if mode == "2_players":
            self.elo_label.hide()
            self.elo_combo.hide()
        else:
            self.elo_label.show()
            self.elo_combo.show()

    def handle_start(self):
        color = self.color_combo.currentText()
        elo = int(self.elo_combo.currentText())
        mode = self.mode_combo.currentText()
        self.start_game.emit(color.lower(), elo, mode)