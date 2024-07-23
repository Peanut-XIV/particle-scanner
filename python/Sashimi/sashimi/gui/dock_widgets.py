import sys
from importlib.resources import files
from enum import IntEnum
from pathlib import Path

from PySide6.QtCore import Qt, Slot, QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
        QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget,
        QGridLayout, QHBoxLayout, QLayout, QDockWidget, QSizePolicy, QSpinBox,
        QDoubleSpinBox, QListWidget, QTextEdit, QCheckBox
)

import sashimi.resources
from sashimi.configuration.configuration import Configuration
from sashimi.gui.dialogs import CNNDirectoryDialog


class DirectionalButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAutoRepeat(True)
        self.setAutoRepeatDelay(800)
        self.setAutoRepeatInterval(300)


class CheckState(IntEnum):
    OFF = 0
    ON = 1


class MovementsWidget(QDockWidget):
    """
    A QDockWidget Class to display the scanner's movement controls
    """
    user_changed_cnn = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        title = QLabel("Movement Controls", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: lightgrey;")
        self.setTitleBarWidget(title)

        # Inner state for the dialog box
        self.current_cnn_model = str(Path("~").expanduser())

        # A container for the content of the Dock QDockWidget
        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        button_size = QSize(50, 50)

        start_stop = QHBoxLayout()

        icons = files(sashimi.resources).joinpath("icons")
        self.start_button = QPushButton(container)
        self.start_button.setFixedSize(button_size)
        self.start_button.setIcon(QIcon(str(icons.joinpath("start.png"))))
        self.start_button.setIconSize(button_size)
        start = QVBoxLayout()
        start.addWidget(QLabel("Start Scan:", container))
        start.addWidget(self.start_button)

        self.stop_button = QPushButton(container)
        self.stop_button.setFixedSize(button_size)
        self.stop_button.setIcon(QIcon(str(icons.joinpath("stop.png"))))
        self.stop_button.setIconSize(button_size)
        self.stop_button.setDisabled(True)
        stop = QVBoxLayout()
        stop.addWidget(QLabel("Stop Scan:", container))
        stop.addWidget(self.stop_button)

        start_stop.addLayout(start)
        start_stop.addLayout(stop)
        layout.addLayout(start_stop)

        layout.addWidget(QLabel("Stage position:", container))
        position = QHBoxLayout()
        self.x_label = QLabel("X:", container)
        position.addWidget(self.x_label)
        self.y_label = QLabel("Y:", container)
        position.addWidget(self.y_label)
        self.z_label = QLabel("Z:", container)
        position.addWidget(self.z_label)
        layout.addLayout(position)

        directional = QHBoxLayout() # A layout for all directional controls
        up_down = QVBoxLayout()
        up_down.addStretch(1)
        self.button_up = DirectionalButton("UP", container)
        self.button_down = DirectionalButton("DOWN", container)
        self.button_up.setFixedSize(button_size)
        self.button_down.setFixedSize(button_size)
        up_down.addWidget(self.button_up)
        up_down.addWidget(self.button_down)
        up_down.addStretch(1)
        directional.addLayout(up_down)

        directional.addStretch(1)

        horizontal = QGridLayout()
        button_and_coords = [("FWD", 0, 1),
                             ("BACK", 2, 1),
                             ("LEFT", 1, 0),
                             ("RIGHT", 1, 2)]
        buttons = []
        for name, row, col in button_and_coords:
            btn = DirectionalButton(name, container)
            btn.setFixedSize(button_size)
            horizontal.addWidget(btn, row, col)
            buttons.append(btn)
        self.button_forward = buttons[0]
        self.button_backward = buttons[1]
        self.button_left = buttons[2]
        self.button_right = buttons[3]
        horizontal.setSpacing(10)
        directional.addLayout(horizontal)
        layout.addLayout(directional)

        # Hint
        # hint = QLabel("Press Shift to move faster.", container)
        # hint.setStyleSheet("color: grey; font-style: italic;")
        # hint.setAlignment(Qt.AlignCenter)
        # layout.addWidget(hint)

        home_buttons = QHBoxLayout()
        self.button_set_home = QPushButton("Set Home", container)
        self.button_go_home = QPushButton("Go Home", container)
        home_buttons.addWidget(self.button_set_home)
        home_buttons.addWidget(self.button_go_home)

        layout.addLayout(home_buttons)
        detection = QVBoxLayout()
        self.recalibrate_button = QCheckBox("calibrate while scanning", container)
        self.recalibrate_button.setCheckState(Qt.Unchecked)
        self.dws_button = QCheckBox("Detect while scanning", container)
        self.dws_button.setCheckState(Qt.Unchecked)
        self.skip_stacks_button = QCheckBox("Skip all stacks", container)
        self.save_detection_frame_button = QCheckBox("Save detection frames", container)
        self.skip_stacks_button.setCheckState(Qt.Unchecked)
        self.save_detection_frame_button.setCheckState(Qt.Unchecked)
        detection.addWidget(self.recalibrate_button)
        detection.addWidget(self.dws_button)
        detection.addWidget(self.skip_stacks_button)
        detection.addWidget(self.save_detection_frame_button)
        model_layout = QHBoxLayout()
        self.choose_model_button = QPushButton("set CNN model", container)
        self.current_model_txtbox = QLabel("Current model: None", container)
        model_layout.addWidget(self.current_model_txtbox)
        model_layout.addWidget(self.choose_model_button)
        detection.addLayout(model_layout)
        layout.addLayout(detection)

        layout.addStretch(1)
        container.setLayout(layout)
        container.setMinimumSize(250, 250)
        container.setMaximumSize(300, 300)
        self.setWidget(container)

    @Slot()
    def update_stage_state(self, stage_state):
        self.x_label.setText(f"X: {stage_state.x}")
        self.y_label.setText(f"Y: {stage_state.y}")
        self.z_label.setText(f"Z: {stage_state.z}")

    def bind_controls(self, worker):
        """
        Signals are bound outside of construction for easier testing
        """
        slots = [
            worker.scanner.start,
            worker.scanner.cancel,
            worker.stage_move_up,
            worker.stage_move_down,
            worker.stage_move_left,
            worker.stage_move_right,
            worker.stage_move_forward,
            worker.stage_move_back,
            worker.stage_set_home,
            worker.stage_home,
        ]
        buttons = self.get_buttons()
        for button, slot in zip(buttons, slots):
            button.clicked.connect(slot)
        self.recalibrate_button.stateChanged.connect(worker.scanner.set_autocalibration)
        self.dws_button.stateChanged.connect(worker.scanner.set_dws)
        self.skip_stacks_button.stateChanged.connect(worker.scanner.set_skip_stack)
        self.save_detection_frame_button.stateChanged.connect(worker.scanner.set_save_detection_frames)
        self.choose_model_button.clicked.connect(self.dialog_model_path)
        self.user_changed_cnn.connect(worker.scanner.set_current_model_path)

    def get_buttons(self):
        return [self.start_button,
                self.stop_button,
                self.button_up,
                self.button_down,
                self.button_left,
                self.button_right,
                self.button_forward,
                self.button_backward,
                self.button_set_home,
                self.button_go_home]

    @Slot()
    def dialog_model_path(self):
        CLEAR = "\033[0m"
        ORANGE = "\033[33m"
        dialog_box = CNNDirectoryDialog()
        new_path = dialog_box.getExistingDirectory(None, caption="New image detection model directory", dir=self.current_cnn_model)
        print(f"new path is \"{new_path}\" ({type(new_path)})")
        elements = [elem.name for elem in Path(new_path).iterdir()]
        valid_dir = True
        if "labels.txt" not in elements:
            print(ORANGE + "WARNING: the given directory is invalid as the file labels.txt is missing" + CLEAR)
            valid_dir = False
        if "model.pt" not in elements and "model.pth" not in elements:
            print(ORANGE + "WARNING: the given directory is invalid as the file model.pt or model.pth is missing" + CLEAR)
            valid_dir = False
        if valid_dir:
            self.user_changed_cnn.emit(new_path)
            self.current_cnn_model = new_path
            self.current_model_txtbox.setText(
                "Current model:\n" + Path(new_path).name)

    @Slot(str)
    def update_model_path(self, path_str):
        """
        Sets the directory from which the model directory selection dialog starts
        It is necessary after the init of the scanner to update all of the UI data
        """
        self.current_cnn_model = path_str
        self.current_model_txtbox.setText(
            "Current model:\n" + Path(path_str).name)


class CameraSettingsWidget(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        title = QLabel("Image Settings", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: lightgrey;")
        self.setTitleBarWidget(title)

        # A container for the content of the QDockWidget
        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        layout.addWidget(QLabel("Color Channels:", container))

        color_buttons = QHBoxLayout()
        # TODO: Use QRadioButton instead
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
        layout.addStretch(1)

        container.setLayout(layout)
        container.setMinimumSize(200, 160)
        self.setWidget(container)

    def bind_controls(self, worker):
        """
        Signals are bound outside of construction for easier testing
        """
        self.button_BGR.clicked.connect(worker.camera_bgr)
        self.button_B.clicked.connect(worker.camera_blue)
        self.button_G.clicked.connect(worker.camera_green)
        self.button_R.clicked.connect(worker.camera_red)
        self.exposure_spinbox.valueChanged.connect(worker.camera_exposure)
        self.gain_spinbox.valueChanged.connect(worker.camera_gain)


class ZonesSettingsWidget(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        title = QLabel("Scan Settings", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: lightgrey;")
        self.setTitleBarWidget(title)

        # A container for the content of the QDockWidget
        container = QWidget(self)
        layout = QVBoxLayout()

        spinboxes = QGridLayout()
        spinboxes.addWidget(QLabel("Stack height (µm):", container), 0, 0)
        sh_spinbox = QSpinBox(container)
        sh_spinbox.setRange(100, 10_000)
        sh_spinbox.setSingleStep(200)
        spinboxes.addWidget(sh_spinbox, 0, 1)
        spinboxes.addWidget(QLabel("Stack step (µm):", container), 1, 0)
        ss_spinbox = QSpinBox(container)
        ss_spinbox.setRange(20, 200)
        ss_spinbox.setSingleStep(20)
        spinboxes.addWidget(ss_spinbox, 1, 1)
        layout.addLayout(spinboxes)
        self.stack_height_spinbox = sh_spinbox
        self.stack_step_spinbox = ss_spinbox

        layout.addWidget(QLabel("Set zone borders:", container))
        grid = QGridLayout()
        button_names = [("Back Left", 0, 0),
                        ("Back Right", 0, 1),
                        ("Front Left", 1, 0)]
        corner_buttons = []
        for name, col, row in button_names:
            btn = QPushButton(name, container)
            grid.addWidget(btn, col, row)
            corner_buttons.append(btn)
        self.z_corr_button = corner_buttons[0]
        self.br_button = corner_buttons[1]
        self.fl_button = corner_buttons[2]
        layout.addLayout(grid)

        hint_text = ("Press Shift when clicking to move\n"
                     "quickly to the designated corner.")
        hint = QLabel(hint_text, container)
        hint.setStyleSheet("color: grey; font-style: italic;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        zone_select = QGridLayout()
        label1 = QLabel("Select Zone:", container)
        label2 = QLabel("Manage Zones:", container)
        zone_select.addWidget(label1, 0, 0, 1, 2)
        zone_select.addWidget(label2, 0, 2, 1, 2)
        button_names = [("Prev", 1, 0),
                        ("Next", 1, 1),
                        ("Add", 1, 2),
                        ("Rmv", 1, 3)]
        selection_buttons = []
        for name, col, row in button_names:
            btn = QPushButton(name, container)
            zone_select.addWidget(btn, col, row)
            selection_buttons.append(btn)
        self.previous_button = selection_buttons[0]
        self.next_button = selection_buttons[1]
        self.add_button = selection_buttons[2]
        self.remove_button = selection_buttons[3]
        layout.addLayout(zone_select)

        layout.addWidget(QLabel("Zones Configuration:", container))
        self.zones_config_box = QTextEdit()
        self.zones_config_box.setReadOnly(True)
        layout.addWidget(self.zones_config_box)

        layout.addStretch(1)
        container.setLayout(layout)
        container.setMinimumSize(250, 400)
        self.setWidget(container)

    def bind_controls(self, worker):
        """
        Signals are bound outside of construction for easier testing
        """
        self.z_corr_button.clicked.connect(worker.scan_set_z_correction)
        self.fl_button.clicked.connect(worker.scan_set_FL)
        self.br_button.clicked.connect(worker.scan_set_BR)
        self.previous_button.clicked.connect(worker.scan_select_previous_zone)
        self.next_button.clicked.connect(worker.scan_select_next_zone)
        self.add_button.clicked.connect(worker.scan_add_zone)
        self.remove_button.clicked.connect(worker.scan_delete_zone)
        self.stack_step_spinbox.valueChanged.connect(worker.stack_set_step)
        self.stack_height_spinbox.valueChanged.connect(worker.stack_set_height)

    @Slot(int, Configuration)
    def update_config_info(self, selected_scan_zone: int, config: Configuration):
        message = ""
        message += f"Home:\t{config.stage.home_offset[0]}\t{config.stage.home_offset[1]}\t{config.stage.home_offset[2]}"
        message += "\n"
        for i, zone in enumerate(config.scanner.zones):
            if selected_scan_zone == i:
                message += f"\n\nScan {i}: [SELECTED]"
            else:
                message += f"\n\nScan {i}:"
            message += (f"\nFL\t{zone.FL[0]}\t{zone.FL[1]}\t{zone.FL[2]}"
                        f"\nBR\t{zone.BR[0]}\t{zone.BR[1]}\t{zone.BR[2]}"
                        f"\nBL_Z\t{zone.BL_Z}"
                        f"\nZ_corrections\t{zone.Z_corrections[0]}\t{zone.Z_corrections[1]}")
        self.zones_config_box.setText(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = QMainWindow()
    center_area = QWidget(mw)
    center_layout = QVBoxLayout()
    center_area.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    movements = MovementsWidget()
    cam = CameraSettingsWidget()
    zones = ZonesSettingsWidget()
    helo = QLabel("HELLO", center_area)
    center_layout.addWidget(helo)
    center_layout.setAlignment(Qt.AlignCenter)
    helo.setAlignment(Qt.AlignCenter)
    center_area.setLayout(center_layout)

    mw.setCentralWidget(center_area)
    mw.addDockWidget(Qt.LeftDockWidgetArea, movements)
    mw.addDockWidget(Qt.LeftDockWidgetArea, cam)
    mw.addDockWidget(Qt.RightDockWidgetArea, zones)
    mw.resize(1200, 800)
    mw.show()

    sys.exit(app.exec())
