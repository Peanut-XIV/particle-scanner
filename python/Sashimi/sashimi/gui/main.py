import sys
from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
        QApplication, QMainWindow, QPushButton, QVBoxLayout,
        QLabel, QWidget, QHBoxLayout, QDoubleSpinBox,
        QSpinBox, QGridLayout, QLayout, QTextEdit
)
from PySide6.QtGui import QImage, QPixmap
import numpy as np

from sashimi.configuration.configuration import Configuration
from sashimi.gui.controller import ControllerWorker


class MainWindow(QMainWindow):
    def __init__(self, **kwargs):
        super().__init__()

        # self.controller = controller
        self.setWindowTitle("Sashimi Controller Interface")

        # Controller
        self.worker = ControllerWorker(**kwargs)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.camera_image_changed.connect(self.update_image)

        # Connect key press signal to worker's slot
        # Assuming you have a mechanism in your GUI to capture key presses
        # self.keyPressedSignal.connect(self.worker.on_key_pressed)


        self.thread.start()

        # self.stage_timer = QTimer()
        # self.stage_timer.timeout.connect(self.worker.stage_poll_position)
        # self.stage_timer.start(500)

        self.image_label = QLabel("Camera Feed")
        self.init_ui()

    def init_ui(self):
        # Main
        layout = QHBoxLayout()

        controls_layout = QVBoxLayout()
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)

        camera_layout = QVBoxLayout()
        image_widget = QWidget()
        image_widget.setLayout(camera_layout)

        layout.addWidget(controls_widget)
        layout.addWidget(image_widget)

        self.init_camera_ui(camera_layout)
        self.init_controls_ui(controls_layout)

        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_camera_ui(self, camera_layout):
        colour_mode_layout = QHBoxLayout()
        button_bgr = QPushButton("BGR")
        button_bgr.clicked.connect(self.worker.camera_bgr)
        button_blue = QPushButton("Blue")
        button_blue.clicked.connect(self.worker.camera_blue)
        button_green = QPushButton("Green")
        button_green.clicked.connect(self.worker.camera_green)
        button_red = QPushButton("Red")
        button_red.clicked.connect(self.worker.camera_red)

        colour_mode_layout.addWidget(button_bgr)
        colour_mode_layout.addWidget(button_blue)
        colour_mode_layout.addWidget(button_green)
        colour_mode_layout.addWidget(button_red)

        colour_mode_widget = QWidget()
        colour_mode_widget.setLayout(colour_mode_layout)
        camera_layout.addWidget(colour_mode_widget)

        exposure_and_gain_layout = QHBoxLayout()
        self.add_exposure_control(exposure_and_gain_layout)
        self.add_gain_control(exposure_and_gain_layout)
        self.add_layout_as_widget(exposure_and_gain_layout, camera_layout)

        camera_layout.addWidget(self.image_label)

    def add_layout_as_widget(self, layout, parent_layout):
        widget = QWidget()
        widget.setLayout(layout)
        parent_layout.addWidget(widget)

    def add_exposure_control(self, parent_layout):
        layout = QVBoxLayout()
        exposureLabel = QLabel("Exposure Time (ms):", self)
        layout.addWidget(exposureLabel)
        exposureSpinBox = QSpinBox(self)
        exposureSpinBox.setRange(100, 20000)
        exposureSpinBox.setSingleStep(500)
        exposureSpinBox.setValue(self.worker.config.camera.exposure_time)
        layout.addWidget(exposureSpinBox)
        exposureSpinBox.valueChanged.connect(self.worker.camera_exposure)
        self.worker.camera_exposure_changed.connect(exposureSpinBox.setValue)
        widget = QWidget()
        widget.setLayout(layout)
        parent_layout.addWidget(widget)

    def add_gain_control(self, parent_layout):
        layout = QVBoxLayout()
        label = QLabel("Gain:", self)
        layout.addWidget(label)
        spinBox = QDoubleSpinBox(self)
        spinBox.setRange(0.0, 20.0)
        spinBox.setSingleStep(0.5)
        spinBox.setValue(self.worker.config.camera.gain)
        layout.addWidget(spinBox)
        spinBox.valueChanged.connect(self.worker.camera_gain)
        self.worker.camera_gain_changed.connect(spinBox.setValue)
        widget = QWidget()
        widget.setLayout(layout)
        parent_layout.addWidget(widget)

    def init_controls_ui(self, controls_layout):

        self.add_state_information(controls_layout)
        self.add_movement_controls(controls_layout)
        self.add_scan_controls(controls_layout)
        self.add_config_information(controls_layout)

    def add_state_information(self, parent_layout):
        label = QLabel("State")
        parent_layout.addWidget(label)

        layout = QHBoxLayout()
        x_label = QLabel("X:")
        layout.addWidget(x_label)
        y_label = QLabel("Y:")
        layout.addWidget(y_label)
        z_label = QLabel("Z:")
        layout.addWidget(z_label)

        def update_stage_state(stage_state):
            x_label.setText(f"X: {stage_state.x}")
            y_label.setText(f"Y: {stage_state.y}")
            z_label.setText(f"Z: {stage_state.z}")

        self.worker.stage_state_changed.connect(update_stage_state)
        # update_stage_state(self.worker.stage.state)

        parent_layout.addLayout(layout)

    def add_config_information(self, parent_layout):
        displayBox = QTextEdit()
        displayBox.setReadOnly(True)
        def update_config_information(config: Configuration):
            message = ""
            message += f"Home:\t{config.stage.home_offset[0]}\t{config.stage.home_offset[1]}\t{config.stage.home_offset[2]}"
            message += "\n"
            for i, zone in enumerate(config.scanner.zones):
                if self.worker.selected_scan_zone == i:
                    message += f"\n\nScan {i}: [SELECTED]"
                else:
                    message += f"\n\nScan {i}:"
                message += (f"\nFL\t{zone.FL[0]}\t{zone.FL[1]}\t{zone.FL[2]}"
                            f"\nBR\t{zone.BR[0]}\t{zone.BR[1]}\t{zone.BR[2]}"
                            f"\nBL_Z\t{zone.BL_Z}"
                            f"\nZ_corrections\t{zone.Z_corrections[0]}\t{zone.Z_corrections[1]}")
            displayBox.setText(message)

        self.worker.config_changed.connect(update_config_information)
        update_config_information(self.worker.config)

        parent_layout.addWidget(displayBox)

    def add_scan_controls(self, parent_layout):
        layout = QVBoxLayout()

        label = QLabel()
        label.setText("Scanning")
        layout.addWidget(label)

        # Row 1: Select Previous, Select Next
        row1Layout = QHBoxLayout()
        btnPrevZone = QPushButton("Previous Zone")
        btnPrevZone.clicked.connect(self.worker.scan_select_previous_zone)
        row1Layout.addWidget(btnPrevZone)

        btnNextZone = QPushButton("Next Zone")
        btnNextZone.clicked.connect(self.worker.scan_select_next_zone)
        row1Layout.addWidget(btnNextZone)

        btnAddZone = QPushButton("New Zone")
        btnAddZone.clicked.connect(self.worker.scan_add_zone)
        row1Layout.addWidget(btnAddZone)

        btnDeleteZone = QPushButton("Delete Zone")
        btnDeleteZone.clicked.connect(self.worker.scan_delete_zone)
        row1Layout.addWidget(btnDeleteZone)
        layout.addLayout(row1Layout)

        # Row 2: Add New, Set FL, Set BR, Set Z Correction
        row2Layout = QHBoxLayout()

        btnSetFL = QPushButton("Set Front Left")
        btnSetFL.clicked.connect(self.worker.scan_set_FL)
        row2Layout.addWidget(btnSetFL)

        btnSetBR = QPushButton("Set Back Right")
        btnSetBR.clicked.connect(self.worker.scan_set_BR)
        row2Layout.addWidget(btnSetBR)

        btnSetZCor = QPushButton("Set Z Correction")
        btnSetZCor.clicked.connect(self.worker.scan_set_z_correction)
        row2Layout.addWidget(btnSetZCor)
        layout.addLayout(row2Layout)

        # Row 3: Delete Selected, Delete All
        row3Layout = QHBoxLayout()
        self.add_stack_height_control(row3Layout)
        self.add_stack_step_control(row3Layout)
        layout.addLayout(row3Layout)

        row_4_layout = QHBoxLayout()
        start_scan_button = QPushButton("Start Scanning")
        start_scan_button.clicked.connect(self.worker.scanner.start)
        row_4_layout.addWidget(start_scan_button)

        button = QPushButton("Stop Scanning")
        button.clicked.connect(self.worker.scanner.cancel)
        row_4_layout.addWidget(button)

        layout.addLayout(row_4_layout)

        parent_layout.addLayout(layout)

    def add_movement_controls(self, parent_layout):
        # Label
        label = QLabel()
        label.setText("Home")
        parent_layout.addWidget(label)

        # Home buttons
        home_layout = QHBoxLayout()
        # zero_button = QPushButton("Zero")
        # zero_button.clicked.connect(self.worker.stage_zero)
        # home_layout.addWidget(zero_button)
        home_button = QPushButton("Home")
        home_button.clicked.connect(self.worker.stage_home)
        home_layout.addWidget(home_button)
        set_home_button = QPushButton("Set Home")
        set_home_button.clicked.connect(self.worker.stage_set_home)
        home_layout.addWidget(set_home_button)
        self.add_layout_as_widget(home_layout, parent_layout)

        # Label
        label = QLabel()
        label.setText("Movement")
        parent_layout.addWidget(label)

        # Movement buttons
        grid = QGridLayout()
        grid.setSizeConstraint(QLayout.SetFixedSize)
        # Create and place buttons in a 5x6 grid
        for row in range(5):
            for col in range(6):
                # Place "Left" and "Right" in the middle axis (column 2 and 3)
                if row == 2 and col == 1:
                    button = QPushButton("Left")
                    button.clicked.connect(self.worker.stage_move_left)
                elif row == 2 and col == 3:
                    button = QPushButton("Right")
                    button.clicked.connect(self.worker.stage_move_right)
                # "X Left" and "X Right" next to "Left" and "Right" (columns 0 and 5)
                elif row == 2 and col == 0:
                    button = QPushButton("X Left")
                    button.clicked.connect(self.worker.stage_move_x_left)
                elif row == 2 and col == 4:
                    button = QPushButton("X Right")
                    button.clicked.connect(self.worker.stage_move_x_right)


                # Place "Forward" and "Back" in the middle axis (row 2 and 3)
                elif col == 2 and row == 1:
                    button = QPushButton("Forward")
                    button.clicked.connect(self.worker.stage_move_forward)
                elif col == 2 and row == 3:
                    button = QPushButton("Back")
                    button.clicked.connect(self.worker.stage_move_back)
                # "X Forward" and "X Back" next to "Forward" and "Back" (row 0 and 4, column 2)
                elif col == 2 and row == 0:
                    button = QPushButton("X Forward")
                    button.clicked.connect(self.worker.stage_move_x_forward)
                elif col == 2 and row == 4:
                    button = QPushButton("X Back")
                    button.clicked.connect(self.worker.stage_move_x_back)

                # Place "Up" and "Down" in the last column (row 2 and 3)
                elif col == 5 and row == 1:
                    button = QPushButton("Up")
                    button.clicked.connect(self.worker.stage_move_up)
                elif col == 5 and row == 3:
                    button = QPushButton("Down")
                    button.clicked.connect(self.worker.stage_move_down)
                # "X Up" and "X Down" below "Up" and "Down" (last column, row 2 and 4)
                elif col == 5 and row == 0:
                    button = QPushButton("X Up")
                    button.clicked.connect(self.worker.stage_move_x_up)
                elif col == 5 and row == 4:
                    button = QPushButton("X Down")
                    button.clicked.connect(self.worker.stage_move_x_down)

                else:
                    continue
                grid.addWidget(button, row, col)

        self.add_layout_as_widget(grid, parent_layout)

    def add_stack_height_control(self, parent_layout):
        layout = QVBoxLayout()
        label = QLabel("Stack Height (um):", self)
        layout.addWidget(label)
        spinBox = QSpinBox(self)
        spinBox.setRange(100, 10000)
        spinBox.setSingleStep(200)
        spinBox.setValue(self.worker.config.scanner.stack_height)
        layout.addWidget(spinBox)
        spinBox.valueChanged.connect(self.worker.stack_set_height)
        def update(config: Configuration):
            spinBox.setValue(config.scanner.stack_height)
        self.worker.config_changed.connect(update)
        widget = QWidget()
        widget.setLayout(layout)
        parent_layout.addWidget(widget)

    def add_stack_step_control(self, parent_layout):
        layout = QVBoxLayout()
        label = QLabel("Stack Step (um):", self)
        layout.addWidget(label)
        spinBox = QSpinBox(self)
        spinBox.setRange(20, 200)
        spinBox.setSingleStep(20)
        spinBox.setValue(self.worker.config.scanner.stack_step)
        layout.addWidget(spinBox)
        spinBox.valueChanged.connect(self.worker.stack_set_step)
        def update(config: Configuration):
            spinBox.setValue(config.scanner.stack_step)
        self.worker.config_changed.connect(update)
        widget = QWidget()
        widget.setLayout(layout)
        parent_layout.addWidget(widget)

    def update_image(self, image: np.ndarray):
        # Convert image to QImage and display
        q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        # Signal the worker to stop, if you have a mechanism in the worker to do so
        self.worker.stop()

        # Wait for the thread to finish
        self.thread.quit()
        self.thread.wait()

        # Proceed with the rest of the shutdown process
        super().closeEvent(event)


def main(**kwargs):
    # Assuming 'controller' is an instance of your Controller class
    app = QApplication(sys.argv)

    window = MainWindow(**kwargs)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
