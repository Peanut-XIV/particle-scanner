from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QTimer

from sashimi.hardware.camera import Camera
from sashimi.configuration.configuration import Configuration
from sashimi.hardware.scanner import ScannerConfiguration, Scanner, ScanZone
from sashimi.hardware.stage import Stage


class CameraMode(Enum):
    BGR = auto()
    GRAY = auto()
    BLUE = auto()
    GREEN = auto()
    RED = auto()


@dataclass
class StageState:
    x: int
    y: int
    z: int
    reported_x: int
    reported_y: int
    reported_z: int


class ControllerWorker(QObject):

    # Image
    camera_image_changed = Signal(object)

    # Camera settings
    camera_exposure_changed = Signal(int)  # Signal to send exposure time to the main thread
    camera_gain_changed = Signal(float)  # Signal to send gain to the main thread

    # Extra camera settings
    camera_blue_balance_changed = Signal(float)  # Sign
    camera_red_balance_changed = Signal(float)  # Sign
    camera_green_balance_changed = Signal(float)  # Sign

    # Configuration
    config_changed = Signal(Configuration)  # Signal to send the configuration to the main thread

    # Stage
    stage_state_changed = Signal(StageState)

    def __init__(self):
        super().__init__()

        # Configuration
        self.config = Configuration.load_default()

        # Stage
        self.stage = Stage(self.config.stage)
        self.stage.start()

        # Camera
        self.camera = Camera(self.config.camera)
        self.img_mode = CameraMode.BGR
        self.camera.start()

        # Scanner
        self.scanner = Scanner(self.config.scanner, self.camera, self.stage)

        # Zones
        self.selected_scan_zone = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loop)
        self.timer.start(30)

        # Update camera
        self.camera.set_exposure(self.config.camera.exposure_time)
        self.camera.set_gain(self.config.camera.gain)

        # Update UI with config values
        self.camera_exposure_changed.emit(self.config.camera.exposure_time)
        self.camera_gain_changed.emit(self.config.camera.gain)

    def _config_has_changed(self):
        self.config_changed.emit(self.config)

    def _camera_image_has_changed(self, img):
        self.camera_image_changed.emit(img)

    @Slot()
    def stop(self):
        self.config.save_default()
        self.camera.stop()
        self.timer.stop()

    @Slot()
    def loop(self):

        # Camera
        frame = self.camera.latest_image(with_exposure=True)
        if frame is not None:
            img, exposure = frame
            display_img = cv2.resize(img, (640, 480))
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            if self.img_mode == CameraMode.GRAY:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                # Convert grayscale back to BGR
            elif self.img_mode == CameraMode.BLUE:
                # Keep only the blue channel
                # and set others to 0 to maintain BGR format
                b_channel = display_img[:, :, 2]
                display_img = np.zeros_like(display_img)
                display_img[:, :, 2] = b_channel
            elif self.img_mode == CameraMode.GREEN:
                # Keep only the green channel
                # and set others to 0 to maintain BGR format
                g_channel = display_img[:, :, 1]
                display_img = np.zeros_like(display_img)
                display_img[:, :, 1] = g_channel
            elif self.img_mode == CameraMode.RED:
                # Keep only the red channel
                # and set others to 0 to maintain BGR format
                r_channel = display_img[:, :, 0]
                display_img = np.zeros_like(display_img)
                display_img[:, :, 0] = r_channel
            self.camera_image_changed.emit(display_img)
            # Emit signal with the processed BGR image

            if self.camera.rescale != 1.0:
                img = cv2.resize(frame[0], (0, 0),
                                 fx=self.camera.rescale,
                                 fy=self.camera.rescale)
                frame = (img, exposure)

        # Scanner
        self.scanner.loop(frame)

        # Stage
        state = StageState(self.stage.x,
                           self.stage.y,
                           self.stage.z,
                           self.stage.reported_x,
                           self.stage.reported_y,
                           self.stage.reported_z)
        self.stage_state_changed.emit(state)

    @Slot()
    def camera_bgr(self):
        print("camera_bgr slot called")
        self.img_mode = CameraMode.BGR

    @Slot()
    def camera_gray(self):
        self.img_mode = CameraMode.GRAY

    @Slot()
    def camera_blue(self):
        self.img_mode = CameraMode.BLUE

    @Slot()
    def camera_green(self):
        self.img_mode = CameraMode.GREEN

    @Slot()
    def camera_red(self):
        self.img_mode = CameraMode.RED

    @Slot(int)
    def camera_exposure(self, value):
        self.config.camera.exposure_time = value
        self.camera.set_exposure(value)
        self._config_has_changed()

    @Slot(float)
    def camera_gain(self, value):
        self.config.camera.gain = value
        self.camera.set_gain(value)
        self._config_has_changed()

    @Slot()
    def stage_zero(self):
        self.stage.zero(self.config.stage.home_offset)
        print("STAGE: zero")

    @Slot()
    def stage_home(self):
        self.stage.home(self.config.stage.home_offset)
        print("STAGE: move home")

    @Slot()
    def stage_set_home(self):
        self.config.stage.home_offset = self.stage.position
        print(f"STAGE: set home to {self.config.stage.home_offset}")
        self._config_has_changed()

    @Slot()
    def stage_move_forward(self):
        self.stage.move_y(1000)
        print("STAGE: move y 1mm")

    @Slot()
    def stage_move_back(self):
        self.stage.move_y(-1000)
        print("STAGE: move y -1mm")

    @Slot()
    def stage_move_left(self):
        self.stage.move_x(-1000)
        print("STAGE: move x -1mm")

    @Slot()
    def stage_move_right(self):
        self.stage.move_x(1000)
        print("STAGE: move x 1mm")

    @Slot()
    def stage_move_x_forward(self):
        self.stage.move_y(10000)
        print("STAGE: move y 10mm")

    @Slot()
    def stage_move_x_back(self):
        self.stage.move_y(-10000)
        print("STAGE: move y -10mm")

    @Slot()
    def stage_move_x_left(self):
        self.stage.move_x(-10000)
        print("STAGE: move x -10mm")

    @Slot()
    def stage_move_x_right(self):
        self.stage.move_x(10000)
        print("STAGE: move x 10mm")

    @Slot()
    def stage_move_up(self):
        self.stage.move_z(20)
        print("STAGE: move z 20um")

    @Slot()
    def stage_move_down(self):
        self.stage.move_z(-20)
        print("STAGE: move z -20um")

    @Slot()
    def stage_move_x_up(self):
        self.stage.move_z(200)
        print("STAGE: move z 200um")

    @Slot()
    def stage_move_x_down(self):
        self.stage.move_z(-200)
        print("STAGE: move z -200um")

    @Slot()
    def stage_poll_position(self):
        self.stage.poll()
        print("STAGE: poll position")

    @Slot()
    def scan_select_previous_zone(self):
        if self.selected_scan_zone > 0:
            self.selected_scan_zone -= 1
            print("SCAN: Selected previous scan zone")
            self._config_has_changed()

    @Slot()
    def scan_select_next_zone(self):
        if self.selected_scan_zone < len(self.config.scans) - 1:
            self.selected_scan_zone += 1
            print("SCAN: Selected next scan zone")
            self._config_has_changed()

    @Slot()
    def scan_add_zone(self):
        self.config.scanner.zones.append(
            ScanZone(
                FL=[10000, 50000, 2000],
                BR=[11000, 51000, 2000],
                BL_Z=2000,
                Z_corrections=[0, 0]
            )
        )
        self.selected_scan_zone = len(self.config.scanner.zones) - 1
        print("SCAN: Added a new scan zone")
        self._config_has_changed()

    @Slot()
    def scan_delete_zone(self):
        if len(self.config.scanner.zones) >= 1:
            self.config.scanner.zones.pop(self.selected_scan_zone)
            self.selected_scan_zone = max(0, self.selected_scan_zone)
            print("SCAN: Deleted the currently selected scan zone")
            self._config_has_changed()

    @Slot()
    def scan_delete_all_zones(self):
        self.selected_scan_zone = 0
        self.config.scans = []
        print("SCAN: Deleted all scan zones")
        self._config_has_changed()

    @Slot()
    def scan_set_FL(self):
        if len(self.config.scanner.zones) == 0:
            self.scan_add_zone()
        zone = self.config.scanner.zones[self.selected_scan_zone]
        if not (self.stage.x == zone.BR[0] or self.stage.y == zone.BR[1]):
            zone.FL = [self.stage.x, self.stage.y, self.stage.z]
            self.config.update_z_correction_terms(self.selected_scan_zone)
            print("SCAN: Set front left corner of zone")
            self._config_has_changed()

    @Slot()
    def scan_set_BR(self):
        if len(self.config.scanner.zones) == 0:
            self.scan_add_zone()
        zone = self.config.scanner.zones[self.selected_scan_zone]
        if not (self.stage.x == zone.FL[0] or self.stage.y == zone.FL[1]):
            zone.BR = [self.stage.x, self.stage.y, self.stage.z]
            self.config.update_z_correction_terms(self.selected_scan_zone)
            print("SCAN: Set back right corner zone")
            self._config_has_changed()

    @Slot()
    def scan_set_z_correction(self):
        self.config.update_z_correction_terms(self.selected_scan_zone,
                                              self.stage.z)
        print("SCAN: Updated Z correction terms")
        self._config_has_changed()

    @Slot()
    def stack_set_height(self, value):
        self.config.scanner.stack_height = value
        print("STACK: Set stack height")
        self._config_has_changed()

    @Slot()
    def stack_set_step(self, value):
        self.config.scanner.stack_step = value
        print("STACK: Set stack step")
        self._config_has_changed()
