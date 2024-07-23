import sys

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap
import numpy as np

from sashimi.gui.controller import ControllerWorker
from sashimi.gui.dock_widgets import CameraSettingsWidget, MovementsWidget, ZonesSettingsWidget


class MainWindow(QMainWindow):
    disable_ctrl_buttons = Signal(bool)
    request_update = Signal()

    def __init__(self, **kwargs):
        super().__init__()

        # self.controller = controller
        self.setWindowTitle("Sashimi Controller Interface")

        # Controller
        self.worker = ControllerWorker(self.disable_ctrl_buttons, self.request_update, **kwargs)
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
        central_widget = QWidget()
        layout = QHBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.image_label)
        layout.addStretch(1)
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.get_movements_dock())
        self.addDockWidget(Qt.LeftDockWidgetArea, self.get_camera_settings_dock())
        self.addDockWidget(Qt.RightDockWidgetArea, self.get_zones_settings_dock())
        # Once the main Window is properly initalized, ask for the UI to update its info
        self.request_update.emit()

    def get_camera_settings_dock(self):
        widget = CameraSettingsWidget()
        widget.bind_controls(self.worker)
        self.disable_ctrl_buttons.connect(widget.exposure_spinbox.setDisabled)
        self.worker.camera_exposure_changed.connect(widget.exposure_spinbox.setValue)
        self.disable_ctrl_buttons.connect(widget.gain_spinbox.setDisabled)
        self.worker.camera_gain_changed.connect(widget.gain_spinbox.setValue)
        return widget

    def get_movements_dock(self):
        widget = MovementsWidget()
        widget.bind_controls(self.worker)
        for button in widget.get_buttons():
            if button is widget.stop_button:
                self.disable_ctrl_buttons.connect(button.setEnabled)
            else:
                self.disable_ctrl_buttons.connect(button.setDisabled)
        self.disable_ctrl_buttons.connect(widget.recalibrate_button.setDisabled)
        self.disable_ctrl_buttons.connect(widget.dws_button.setDisabled)
        self.disable_ctrl_buttons.connect(widget.choose_model_button.setDisabled)
        self.disable_ctrl_buttons.connect(widget.skip_stacks_button.setDisabled)
        self.disable_ctrl_buttons.connect(widget.save_detection_frame_button.setDisabled)
        self.worker.stage_state_changed.connect(widget.update_stage_state)
        self.worker.model_path_changed.connect(widget.update_model_path)
        return widget

    def get_zones_settings_dock(self):
        widget = ZonesSettingsWidget()
        widget.bind_controls(self.worker)
        buttons = [
                widget.z_corr_button,
                widget.fl_button,
                widget.br_button,
                widget.add_button,
                widget.remove_button,
        ]
        for button in buttons:
            self.disable_ctrl_buttons.connect(button.setDisabled)
        self.worker.stack_step_changed.connect(widget.stack_step_spinbox.setValue)
        self.worker.stack_height_changed.connect(widget.stack_height_spinbox.setValue)
        self.worker.zones_changed.connect(widget.update_config_info)
        return widget

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
