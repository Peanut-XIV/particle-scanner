import sys

from PySide6.QtCore import QObject, Qt, QSize
from PySide6.QtWidgets import (
        QApplication, QMainWindow, QPushButton, QVBoxLayout,
        QLabel, QWidget, QHBoxLayout, QDoubleSpinBox,
        QSpinBox, QGridLayout, QLayout, QTextEdit, QDockWidget,
        QSizePolicy
)


class MovementsWidget(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.title = QLabel("Movement Controls", self)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("background-color: lightgrey;")
        self.setTitleBarWidget(self.title)

        button_width = 50
        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize)

        self.button_panel = QWidget(container)
        grid = QGridLayout()
        button_and_coords = [
                ("UP",   0, 0),
                ("DOWN", 1, 0),
                ("FWD",  0, 3),
                ("BACK", 2, 3),
                ("LEFT", 1, 2),
                ("RIGHT",1, 4),
        ]
        for name, row, col in button_and_coords:
            btn = QPushButton(name, self.button_panel)
            btn.setFixedSize(button_width, button_width)
            grid.addWidget(btn, row, col)
        grid.setColumnMinimumWidth(1, button_width)
        grid.setSpacing(10)
        self.button_panel.setLayout(grid)
        # Hint
        self.hint = QLabel("Press Shift to move faster.", container)
        self.hint.setStyleSheet("color: grey; font-style: italic;")
        self.hint.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.button_panel)
        layout.addWidget(self.hint)
        container.setLayout(layout)

        self.setMinimumSize(330, 210)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = QMainWindow()
    center_area = QWidget(mw)
    center_layout = QVBoxLayout()
    center_area.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    movements = MovementsWidget()
    movements2 = MovementsWidget()
    helo = QLabel("HELLO", center_area)
    center_layout.addWidget(helo)
    center_layout.setAlignment(Qt.AlignCenter)
    helo.setAlignment(Qt.AlignCenter)
    center_area.setLayout(center_layout)

    mw.setCentralWidget(center_area)
    mw.addDockWidget(Qt.LeftDockWidgetArea, movements)
    mw.addDockWidget(Qt.LeftDockWidgetArea, movements2)
    mw.resize(1200, 800)
    mw.show()

    sys.exit(app.exec())
