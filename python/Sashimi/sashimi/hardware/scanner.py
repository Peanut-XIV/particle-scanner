"""

State machine ticks over at the camera frame rate

It reads the current image and states

"""
import os
from dataclasses import field, dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional
import skimage.io as skio

import multiprocessing as mp

from PySide6.QtCore import QObject, Signal, QTimer, Slot

from sashimi import utils
from sashimi.configuration.base import BaseModel
from sashimi.hardware.camera import Camera
from sashimi.hardware.stage import Stage
from sashimi.stacking import helicon_stack


@dataclass
class ScanZone(BaseModel):
    FL: List[int]
    BR: List[int]
    BL_Z: int
    Z_corrections: List[int]


@dataclass
class ScannerConfiguration(BaseModel):
    """
    Configuration for the state machine
    """
    save_dir: str = str(Path.home().joinpath("sashimi_scans"))
    scan_name: str = "test"
    overwrite: bool = False

    zones: List[ScanZone] = field(default_factory=list)
    exposure_times: Optional[List[int]] = None
    z_margin: int = 200
    lowest_z: int = None

    stack_height: int = 2000
    stack_step: int = 60

    overlap_x: float = 0.25
    overlap_y: float = 0.25

    remove_raw_images: bool = True


@dataclass
class ScannerState:
    """
    Current state of the state machine
    """
    # Global state
    state: str = "idle"

    # Wait function
    wait_fn: Optional[callable] = None
    wait_ticks: Optional[int] = None
    wait_state: Optional[str] = None

    num_zones: int = 0
    num_exposures: int = 0
    num_stacks: int = 0
    num_focus: int = 0

    step_x: int = 0
    step_y: int = 0

    num_steps_x: int = 0
    num_steps_y: int = 0

    zone_idx: int = 0
    stack_idx: int = 0
    exposure_idx: int = 0
    focus_idx: int = 0

    image_idx: int = 0

    z_orig: int = 0
    stack_x: int = 0
    stack_y: int = 0
    stack_z: int = 0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class Scanner(QObject):
    # State has changed
    state_changed = Signal(ScannerState)

    def __init__(self,
                 config: ScannerConfiguration,
                 camera: Camera,
                 stage: Stage,
                 disable_ctrl: Signal(bool),
                 **kwargs):

        super().__init__()
        # Configuration
        self.config = config
        self.stack_method = kwargs["stack_method"]
        # TODO: implement stack method as a config option
        #       instead of a kwarg passed down from main()

        # State
        self.state = ScannerState()

        # Devices
        self.camera = camera
        self.stage = stage

        # Queue for parallel processing
        self.queue = None
        self.parallel_process = None

        # Camera
        self.camera_exposure = 0
        self.camera_img = None

        # Cancelled
        self.scan_cancelled = False

        # Update step
        self.state.step_x = int(self.camera.width * (1 - self.config.overlap_x))
        self.state.step_y = int(self.camera.height * (1 - self.config.overlap_y))

        self.disable_ctrl = disable_ctrl

    @Slot()
    def start(self):
        self.state.start_time = datetime.now()
        self.scan_cancelled = False
        self._transition_to("init")

    @Slot()
    def cancel(self):
        self.scan_cancelled = True

    def _state_has_changed(self):
        self.state_changed.emit(self.state)

    def _transition_to(self, new_state):
        self.state.state = new_state
        self._state_has_changed()

    def _wait_for_move_then_transition_to(self, state, ms):
        self.state.wait_fn = self.stage.is_ready
        self.state.wait_ticks = ms // 30
        self.state.wait_state = state
        self._transition_to("wait")
        self._state_has_changed()

    def _wait_for_exposure_then_transition_to(self, state, exposure_time):
        self.state.wait_fn = lambda: self.camera_exposure == exposure_time
        self.state.wait_ticks = 300 // 30
        self.state.wait_state = state
        self._transition_to("wait")
        self._state_has_changed()

    def _get_scan_path(self):
        # Get current date and time as string
        datetime_str = self.state.start_time.strftime("%Y%m%d_%H%M%S")
        return (Path(self.config.save_dir)
                / f"{datetime_str}_{self.config.scan_name}")

    def _get_exposure_path(self):
        # Get current date and time as string
        datetime_str = self.state.start_time.strftime("%Y%m%d_%H%M%S")
        return (Path(self.config.save_dir)
                / f"{datetime_str}_{self.config.scan_name}"
                / f"Zone{self.state.zone_idx:03d}"
                / f"Exp{self.config.exposure_times[self.state.exposure_idx]:05d}")

    def _get_stack_path(self):
        return self._get_exposure_path() / r"images"

    def _get_raw_path(self):
        return self._get_exposure_path() / "raw" / f"Yi{self.state.stack_y:06d}_Xi{self.state.stack_x:06d}"

    def _get_stack_filename(self, x, y):
        return self._get_stack_path() / (f"{self.config.scan_name}"
                                         f"_Zone{self.state.zone_idx:03d}"
                                         f"_Exp{self.config.exposure_times[self.state.exposure_idx]:05d}"
                                         f"_Xi{self.state.stack_x:06d}_Yi{self.state.stack_y:06d}"
                                         f"_X{x:06d}_Y{y:06d}")

    def _get_stack_offset(self):
        zone = self.config.zones[self.state.zone_idx]
        steps_x, steps_y = self.state.step_x, self.state.step_y
        xi, yi = self.state.stack_x, self.state.stack_y
        fl = zone.FL
        dx, dy = steps_x * xi, steps_y * yi
        xy = [fl[0] + dx, fl[1] + dy, fl[2]]
        return xy, dx, dy

    def _lowest_corner_of_zone(self) -> int:
        zone = self.config.zones[self.state.zone_idx]
        fl = zone.FL
        br = zone.BR
        blz = zone.BL_Z
        flz = fl[2]
        brz = br[2]
        frz = flz - brz + blz
        mini = min((blz, brz, flz, frz))
        mini = max(0, mini)
        return mini

    def _get_corrected_z(self, dx, dy):
        zone = self.config.zones[self.state.zone_idx]
        if self.config.lowest_z:
            # 'Dumb-but-works' correction
            new_z = self._lowest_corner_of_zone()
        else:
            # 'Smart' correction
            dz_dx, dz_dy = zone.Z_corrections
            z_correction = int(dz_dx * dx + dz_dy * dy)
            new_z = zone.FL[2] + z_correction
        return max(new_z - self.config.z_margin, 0)

    def _get_steps_xy(self, scan) -> (int, int):
        x_steps = 1 + (scan.BR[0] - scan.FL[0]) // self.state.step_x
        y_steps = 1 + (scan.BR[1] - scan.FL[1]) // self.state.step_y
        return x_steps, y_steps

    def wait_state(self):
        if self.state.wait_fn():
            self._transition_to(self.state.wait_state)
            self._state_has_changed()
        else:
            self.state.wait_ticks -= 1
            if self.state.wait_ticks <= 0:
                self._transition_to(self.state.wait_state)
                print("ERROR: timeout waiting")
                self._state_has_changed()

    def loop(self, frame):

        # If cancelled, goto done
        if self.scan_cancelled:
            self._transition_to("done")

        # The current state
        state = self.state.state

        # Get image every loop
        # frame = self.camera.latest_image(with_exposure=True)
        if frame is not None:
            self.camera_img, self.camera_exposure = frame

        # print(f"SCANNER: {state}")

        # --------------------------------------------------------------------
        if state == "idle":
            self.disable_ctrl.emit(False)
            return
        # --------------------------------------------------------------------
        elif state == "wait":
            if self.state.wait_fn():
                self._transition_to(self.state.wait_state)
                self._state_has_changed()
            else:
                self.state.wait_ticks -= 1
                if self.state.wait_ticks <= 0:
                    self._transition_to(self.state.wait_state)
                    print("ERROR: timeout waiting")
                    self._state_has_changed()
        # --------------------------------------------------------------------
        elif state == "init":
            # Return to idea state if no zones
            self.disable_ctrl.emit(True)
            if len(self.config.zones) == 0:
                self._transition_to("idle")
                return

            # If no exposure times
            if self.config.exposure_times is None or len(self.config.exposure_times) <= 1:
                self.config.exposure_times = [self.camera_exposure]

            # Initialise
            self.state.is_scanning = True

            # Create scan, removing existing if necessary
            scan_dir = self._get_scan_path()
            if scan_dir.exists() and self.config.overwrite:
                utils.remove_folder(scan_dir)
            Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

            # Stack errors
            self.error_logs = scan_dir / 'error_logs.txt'
            if self.error_logs.exists():
                os.remove(self.error_logs)

            # Parallel stack command queue
            self.queue = mp.Queue()
            arguments = (self.queue,
                         self.error_logs,
                         self.stack_method,
                         self.config.remove_raw_images)
            self.parallel_process = mp.Process(target=helicon_stack.parallel_stack, args=arguments)
            self.parallel_process.start()

            # Update state
            self.state.num_exposures = len(self.config.exposure_times)
            self.state.num_zones = len(self.config.zones)
            self.state.zone_idx = 0
            self._transition_to("zone")
        # --------------------------------------------------------------------
        elif state == "zone":
            # If we have finished all the zones
            if self.state.zone_idx == self.state.num_zones:
                self._transition_to("done")
                return

            # Get the current zone
            zone = self.config.zones[self.state.zone_idx]

            # Get the zone steps etc
            self.state.num_steps_x, self.state.num_steps_y = self._get_steps_xy(zone)
            self.stack_idx = 0
            self.num_stacks = self.state.num_steps_x * self.state.num_steps_y * len(self.config.exposure_times)
            self.state.stack_x = 0
            self.state.stack_y = 0
            self.state.image_idx = 0

            # Move to zone start
            self.stage.goto(zone.FL, busy=True)
            self._wait_for_move_then_transition_to("stack_init", 10000)
        # --------------------------------------------------------------------
        elif state == "stack_init":
            # If we have finished all the stacks
            if self.state.stack_x >= self.state.num_steps_x and self.state.stack_y >= self.state.num_steps_y:
                self.state.zone_idx += 1
                self._transition_to("start_zone")
                return

            # First exposure
            self.state.exposure_idx = 0

            # Move to start
            du, _, _ = self._get_stack_offset()
            print(du)
            self.stage.goto(du, busy=True)
            self._wait_for_move_then_transition_to("stack_exposure", 10000)
        # --------------------------------------------------------------------
        elif state == "stack_exposure":
            # All exposures done?
            if self.state.exposure_idx == self.state.num_exposures:
                print("STACK: All exposures done")
                self.state.stack_x += 1
                if self.state.stack_x >= self.state.num_steps_x:
                    self.state.stack_y += 1
                    if self.state.stack_y >= self.state.num_steps_y:
                        self.state.zone_idx += 1
                        self._transition_to("zone")
                        return
                    self.state.stack_x = 0
                self._transition_to("stack_init")
                return
            # Get the current exposure time
            exposure_time = self.config.exposure_times[self.state.exposure_idx]
            # Set the exposure time
            self.camera.set_exposure(exposure_time)
            self._wait_for_exposure_then_transition_to("stack_z", exposure_time)
        # --------------------------------------------------------------------
        elif state == "stack_z":
            # Reset images
            self.state.focus_idx = 0
            self.state.num_focus = (self.config.stack_height + self.config.z_margin) // self.config.stack_step
            # Make path
            self._get_raw_path().mkdir(parents=True, exist_ok=True)
            # Move to start
            self.state.z_orig = self.stage.z
            _, dx, dy = self._get_stack_offset()
            self.stage.goto_z(self._get_corrected_z(dx, dy), busy=True)
            self._wait_for_move_then_transition_to("stack_image", 10000)
        # --------------------------------------------------------------------
        elif state == "stack_image":
            save_path = (self._get_raw_path()
                         / f"X{self.stage.x:06d}_"
                           f"Y{self.stage.y:06d}_"
                           f"Z{self.stage.z:06d}.jpg")
            skio.imsave(str(save_path), self.camera_img[..., ::-1], check_contrast=False, quality=90)
            self.state.focus_idx += 1
            # Stack done?
            print(self.state.focus_idx, self.state.num_focus, self.state.image_idx, self.state.num_exposures)
            if self.state.focus_idx > self.state.num_focus:
                self.queue.put(
                    (
                        str(self._get_raw_path()),
                        str(self._get_stack_filename(self.stage.x, self.stage.y))
                    )
                )
                self.state.image_idx += 1
                self.state.exposure_idx += 1
                self.stage.goto_z(self.state.z_orig, busy=True)
                self._wait_for_move_then_transition_to("stack_exposure", 50 * self.state.num_focus)
            # Move and take next
            else:
                self.stage.move_z(self.config.stack_step, busy=True)
                self._wait_for_move_then_transition_to("stack_image", 100)
        # --------------------------------------------------------------------
        elif state == "done":
            self.state.end_time = datetime.now()
            self.queue.put("terminate")
            self.parallel_process.join()
            self._transition_to("idle")
            return
