import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
        QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget,
        QGridLayout, QHBoxLayout, QLayout, QDockWidget, QSizePolicy, QSpinBox,
        QDoubleSpinBox,
)


class MovementsWidget(QDockWidget):
    """
    A QDockWidget Class to display the scanner's movement controls
    """
    def __init__(self):
        super().__init__()
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        title = QLabel("Movement Controls", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: lightgrey;")
        self.setTitleBarWidget(title)

        # A container for the content of the Dock QDockWidget
        # The interface doesn't get drawn correctly without it...
        # I think it's because the QDockWidget doesn't like layouts.
        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize)

        # Just 10px of margin between the title bar and the rest
        spacer = QWidget(container)
        spacer.setFixedHeight(10)
        layout.addWidget(spacer)

        button_width = 50

        # A grid containing all directional controls
        grid = QGridLayout()
        button_and_coords = [("UP", 0, 0),
                             ("DOWN", 1, 0),
                             ("FWD", 0, 3),
                             ("BACK", 2, 3),
                             ("LEFT", 1, 2),
                             ("RIGHT", 1, 4)]
        buttons = []
        for name, row, col in button_and_coords:
            btn = QPushButton(name, container)
            btn.setFixedSize(button_width, button_width)
            grid.addWidget(btn, row, col)
            buttons.append(btn)
        self.button_up = buttons[0]
        self.button_down = buttons[1]
        self.button_forward = buttons[2]
        self.button_backward = buttons[3]
        self.button_left = buttons[4]
        self.button_right = buttons[5]
        grid.setColumnMinimumWidth(1, button_width)
        grid.setSpacing(10)
        layout.addLayout(grid)

        home_buttons = QHBoxLayout()
        self.button_set_home = QPushButton("Set Home", container)
        self.button_go_home = QPushButton("Go Home", container)
        home_buttons.addWidget(self.button_set_home)
        home_buttons.addWidget(self.button_go_home)
        layout.addLayout(home_buttons)

        # Hint
        hint = QLabel("Press Shift to move faster.", container)
        hint.setStyleSheet("color: grey; font-style: italic;")
        hint.setAlignment(Qt.AlignCenter)

        layout.addWidget(hint)
        container.setLayout(layout)
        self.setMinimumSize(330, 260)

    def bind_buttons(self, worker):
        ...

    def get_buttons(self):
        return [self.button_up,
                self.button_down,
                self.button_left,
                self.button_right,
                self.button_forward,
                self.button_backward,
                self.button_set_home,
                self.button_go_home]

    @Slot()
    def deactivate_buttons(self):
        for button in self.get_buttons():
            button.setDisabled(True)

    @Slot()
    def activate_buttons(self):
        for button in self.get_buttons():
            button.setEnabled(True)


class ImageControlsWidget(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        title = QLabel("Image Settings", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: lightgrey;")
        self.setTitleBarWidget(title)

        # A container for the content of the QDockWidget
        # The interface doesn't get drawn correctly without it...
        # I think it's because the QDockWidget doesn't like layouts.
        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize)

        # Just 10px of margin between the title bar and the rest
        spacer = QWidget(container)
        spacer.setFixedHeight(10)
        layout.addWidget(spacer)


        color_buttons = QHBoxLayout()
        button_names = ["BGR", "B", "G", "R"]
        buttons = []
        for name in button_names:
            btn = QPushButton(name, container)
            color_buttons.addWidget(btn)
            buttons.append(btn)
        self.button_BGR = buttons[0]
        self.button_B = buttons[1]
        self.button_G = buttons[2]
        self.button_R = buttons[3]
        layout.addLayout(color_buttons)

        exposure = QHBoxLayout()
        exposure.addWidget(QLabel("Exposure (µs):", container))
        self.exposure_spinbox = QSpinBox()
        self.exposure_spinbox.setRange(100, 20_000)
        self.exposure_spinbox.setSingleStep(500)
        exposure.addWidget(self.exposure_spinbox)
        layout.addLayout(exposure)

        gain = QHBoxLayout()
        gain.addWidget(QLabel("Gain (dB):", container))
        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setRange(0.0, 20.0)
        self.gain_spinbox.setSingleStep(0.5)
        gain.addWidget(self.gain_spinbox)
        layout.addLayout(gain)

        container.setLayout(layout)
        self.setMinimumSize(250, 160)


class ScanEditWidget(QDockWidget):
    ...


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = QMainWindow()
    center_area = QWidget(mw)
    center_layout = QVBoxLayout()
    center_area.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    movements = MovementsWidget()
    cam = ImageControlsWidget()
    helo = QLabel("HELLO", center_area)
    center_layout.addWidget(helo)
    center_layout.setAlignment(Qt.AlignCenter)
    helo.setAlignment(Qt.AlignCenter)
    center_area.setLayout(center_layout)

    mw.setCentralWidget(center_area)
    mw.addDockWidget(Qt.LeftDockWidgetArea, movements)
    mw.addDockWidget(Qt.LeftDockWidgetArea, cam)
    mw.resize(1200, 800)
    mw.show()

    sys.exit(app.exec())
