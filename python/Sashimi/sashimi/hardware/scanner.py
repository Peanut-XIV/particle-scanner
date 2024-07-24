"""

State machine ticks over at the camera frame rate

It reads the current image and states

"""
import os
from dataclasses import field, dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import List, Optional
import skimage.io as skio

import multiprocessing as mp

from PySide6.QtCore import QObject, Signal, QTimer, Slot, Qt
import numpy as np
import cv2
import torch

from sashimi.utils import Style, remove_folder
from sashimi.configuration.base import BaseModel
from sashimi.hardware.camera import Camera
from sashimi.hardware.stage import Stage
from sashimi.stacking import helicon_stack
from sashimi.detection.cnn import Detector


class States(IntEnum):
    IDLE = 0
    WAIT = 1
    INIT = 2
    ZONE = 3
    STACK_INIT = 4
    SHARPNESS_INIT_A = 5
    SHARPNESS_INIT_B = 6
    SHARPNESS_STEP = 7
    SHARPNESS_END = 8
    STACK_EXPOSURE = 9
    STACK_Z = 10
    STACK_IMAGE = 11
    DONE = 12



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

    cnn_model_dir: str = ""


@dataclass
class ScannerState:
    """
    Current state of the state machine
    """
    # Global state
    state: States = States.IDLE

    initial_Z = None
    prev_pos = None
    cur_pos = None
    prev_sharpness = None
    cur_sharpness = None
    direction = None
    threshold = 0
    wide_step = 120  # µm
    is_stack_valid = True

    detection_boxes = []

    # Wait function
    wait_fn: Optional[callable] = None
    wait_ticks: Optional[int] = None
    wait_state: Optional[str] = None

    stacks_since_calibration = 0

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
        self.stack_method = "helicon"
        # TODO: implement stack method as a config option
        #       instead of a kwarg passed down from main()

        # State
        self.state = ScannerState()

        # Devices
        self.camera = camera
        self.stage = stage

        self.verbosity = 9

        # Used in Detection While Scanning
        self.detector = None
        self.detect_while_scanning = False
        self.save_detection_frames = False
        self.skip_stacks = False
        self.z_min, self.z_max = self.stage.z_limits

        # Autocalibration
        self.does_autocalibrate = False

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

        self.error_logs = None

    def log(self, log_lvl: int, color, text):
        if self.verbosity >= log_lvl:
            print(color + text + Style.CLR)

    def is_out_of_bounds(self, position, log_error=True):
        if log_error and not self.z_min <= position <= self.z_max:
            self.log(0, Style.RED,
                     f"Error : Z={self.z_min} not between"
                     f" {self.z_min} and {self.z_max}!!!")
            return True
        else:
            return False


    @Slot()
    def start(self):
        self.state.start_time = datetime.now()
        self.scan_cancelled = False
        self._transition_to(States.INIT)

    @Slot(int)
    def set_dws(self, new_state):
        if new_state == Qt.CheckState.Checked.value:
            self.detect_while_scanning = True
        else:
            self.detect_while_scanning = False

    @Slot(int)
    def set_autocalibration(self, new_state):
        if new_state == Qt.CheckState.Checked.value:
            self.does_autocalibrate = True
        else:
            self.does_autocalibrate = False

    @Slot(int)
    def set_skip_stack(self, new_state):
        if new_state == Qt.CheckState.Checked.value:
            self.skip_stacks = True
        else:
            self.skip_stacks = False

    @Slot(int)
    def set_save_detection_frames(self, new_state):
        if new_state == Qt.CheckState.Checked.value:
            self.save_detection_frames = True
        else:
            self.save_detection_frames = False

    @Slot(str)
    def set_current_model_path(self, path_str):
        self.config.cnn_model_dir = path_str

    @Slot()
    def cancel(self):
        self.scan_cancelled = True

    def _state_has_changed(self):
        self.state_changed.emit(self.state)

    def _transition_to(self, new_state: int):
        self.state.state = new_state
        self._state_has_changed()

    def _wait_for_move_then_transition_to(self, state, ms):
        self.state.wait_fn = self.stage.is_ready
        self.state.wait_ticks = ms // 30
        self.state.wait_state = state
        self._transition_to(States.WAIT)
        self._state_has_changed()

    def _wait_for_exposure_then_transition_to(self, state, exposure_time):
        self.state.wait_fn = lambda: self.camera_exposure == exposure_time
        self.state.wait_ticks = 300 // 30
        self.state.wait_state = state
        self._transition_to(States.WAIT)
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
        zone = self.config.zones[self.state.zone_idx]  # Errors Here
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


    def _idle(self):
        """
        One of the behaviors of the scanner's state machine
        """
        self.disable_ctrl.emit(False)

    def _wait(self):
        """
        One of the behaviors of the scanner's state machine
        """
        if self.state.wait_fn():
            self._transition_to(self.state.wait_state)
            self._state_has_changed()
        else:
            self.state.wait_ticks -= 1
            if self.state.wait_ticks % 100 == 0:
                print(Style.YLW
                      +f"{self.state.wait_ticks} ticks before timeout"
                      +Style.CLR)
            if self.state.wait_ticks <= 0:
                self._transition_to(self.state.wait_state)
                print(Style.RED+"ERROR: timeout waiting"+Style.CLR)
                self._state_has_changed()

    def _init(self):
        """
        One of the behaviors of the scanner's state machine
        """
        # Return to idea state if no zones
        self.disable_ctrl.emit(True)
        if len(self.config.zones) == 0:
            self._transition_to(States.IDLE)
            return

        # If no exposure times
        if self.config.exposure_times is None or len(self.config.exposure_times) <= 1:
            self.config.exposure_times = [self.camera_exposure]

        # Create scan, removing existing if necessary
        scan_dir = self._get_scan_path()
        if scan_dir.exists() and self.config.overwrite:
            remove_folder(scan_dir)
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

        # Stack errors
        self.error_logs = scan_dir / 'error_logs.txt'
        if self.error_logs.exists():
            os.remove(self.error_logs)

        # load the cnn for foram detection while scanning
        if self.detect_while_scanning:
            self.detector = Detector(self.config.cnn_model_dir)

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
        self._transition_to(States.ZONE)

    def _zone(self):
        """
        One of the behaviors of the scanner's state machine
        """
        # This flag is modified in the different sharpness states
        self.state.is_stack_valid = True
        # If we have finished all the zones
        if self.state.zone_idx == self.state.num_zones:
            self._transition_to(States.DONE)
            return

        # Get the current zone
        zone = self.config.zones[self.state.zone_idx]

        # Get the zone steps etc
        self.state.num_steps_x, self.state.num_steps_y = self._get_steps_xy(zone)
        self.state.stack_idx = 0
        self.state.num_stacks = self.state.num_steps_x * self.state.num_steps_y * len(self.config.exposure_times)
        self.state.stack_x = 0
        self.state.stack_y = 0
        self.state.image_idx = 0

        # Move to zone start
        self.stage.goto(zone.FL, busy=True)
        self._wait_for_move_then_transition_to(States.STACK_INIT, 10000)

    def _stack_init(self):
        """
        One of the behaviors of the scanner's state machine
        """
        self.state.is_stack_valid = True
        # First exposure
        self.state.exposure_idx = 0
        # Move to start
        du, _, _ = self._get_stack_offset()
        # Check wether calibration is necessary or not
        self.log(4, Style.MGTA, f"stacks since calibration: {self.state.stacks_since_calibration}")
        if self.does_autocalibrate and self.state.stack_x == 0 and self.state.stacks_since_calibration >= 30:
            self.log(2, Style.MGTA, "CALIBRATING...")
            self.state.stacks_since_calibration = 0
            self.stage.home(du, busy=True)
        else:
            self.log(4, Style.MGTA, "No need to calibrate.")
            self.state.stacks_since_calibration += 1
        self.stage.goto(du, busy=True)
        if self.detector is not None and self.detect_while_scanning:
            self._wait_for_move_then_transition_to(States.SHARPNESS_INIT_A, 30000)
        else:
            self._wait_for_move_then_transition_to(States.STACK_EXPOSURE, 30000)

    def _sharpness_init_a(self):
        """
        One of the behaviors of the scanner's state machine.
        ---
        The first step in looking for the local sharpness maximum of the stack.
        Measures sharpness of the camera feed at current height, then changes
        state and moves accordingly. Goes to state SHARPNESS_END and doesn't
        move if the next step is Out of Bounds.
        """
        self.log(1, Style.YLW, "init sharpness A")
        self.state.cur_pos = self.stage.z
        self.state.cur_sharpness = sharpness(self.camera_img)
        self.log(3, Style.YLW, f"Sharp={self.state.cur_sharpness}\tZ={self.state.cur_pos}")

        # Avoid iterating until the maximum height :
        self.state.initial_Z = self.stage.z
        self.z_max = self.state.initial_Z + self.config.stack_height

        # Behaves differenstate.tly ial_Z + self.config.stack_height
        if self.state.cur_sharpness < self.state.threshold:
            # sharpness is to low : The sharpness maximum is too far away.
            # It would be slow to reach it stack step by st
            self.log(0, Style.YLW, "sharpness too low")
            # use of wide_step (hard coded for now)
            next_pos = self.state.cur_pos + self.state.wide_step
            if self.is_out_of_bounds(next_pos):
                # check first where we land
                self.state.is_stack_valid = False
                self._transition_to(States.SHARPNESS_END)
                self.log(4, Style.YLW, "Next state = sharpness END")
            else:
                self.stage.goto_z(next_pos, busy=True)
                self._wait_for_move_then_transition_to(States.SHARPNESS_INIT_A, 10_000)
                self.log(4, Style.YLW, "Next state = INIT_A")
        else:
            # sharpness is good enough
            # we measure the gradient to find the direction of the sharpness spike
            next_pos = self.state.cur_pos + self.config.stack_step
            if self.is_out_of_bounds(next_pos):
                # check first where we land
                self.state.is_stack_valid = False
                self._transition_to(States.SHARPNESS_END)
                self.log(4, Style.YLW, "Next state = sharpness END")
            else:
                self.stage.goto_z(next_pos, busy=True)
                self._wait_for_move_then_transition_to(States.SHARPNESS_INIT_B, 10_000)
                self.log(4, Style.YLW, "Next state = INIT_B")

    def _sharpness_init_b(self):
        """
        One of the behaviors of the scanner's state machine.
        ---
        Compares the sharpness of the current image with the sharpess of the
        previous one. Identify the direction of the sharpness gradient. Changes
        state to SHARPNESS_STEP and moves in the direction of the gradient.
        However, it goes to state SHARPNESS_END and doesn't move if the next
        step is Out of Bounds.
        """
        self.log(1, Style.GRN, "init sharpness B")
        # update values
        self.state.prev_pos, self.state.cur_pos = self.state.cur_pos, self.stage.z
        self.state.prev_sharpness, self.state.cur_sharpness = self.state.cur_sharpness, sharpness(self.camera_img)
        self.log(3, Style.GRN, f"Sharp={self.state.cur_sharpness}\tZ={self.state.cur_pos}")
        if self.state.cur_sharpness < self.state.prev_sharpness :
            self.log(2, Style.GRN, "going down")
            # swap values
            self.state.cur_sharpness, self.state.prev_sharpness = self.state.prev_sharpness, self.state.cur_sharpness
            self.state.cur_pos, self.state.prev_pos = self.state.prev_pos, self.state.cur_pos
            self.state.direction = -1
        else:
            self.log(2, Style.GRN, "going up")
            self.state.direction = 1
        # move in the direction of the sharpness gradient
        next_pos = self.state.cur_pos + self.state.direction * self.config.stack_step
        if self.is_out_of_bounds(next_pos):
            # Bad Ending: No, we can't find it :(
            self.state.is_stack_valid = False
            self.log(4, Style.GRN, "Next state = sharpness END")
        else:
            self.stage.goto_z(next_pos, busy=True)
            self.log(4, Style.GRN, "Next state = sharpness STEP")
            self.stage.goto_z(next_pos, busy=True)
            self._wait_for_move_then_transition_to(States.SHARPNESS_STEP, 10000)

    def _sharpness_step(self):
        """
        One of the behaviors of the scanner's state machine.
        ---
        If the current image's sharpness is lower than the previous one's then
        the later is a local maximum.  In this case, move to the local maximum
        and end the search, otherwise, step in the same direction as usual.
        Changes state accordingly.  Goes to state SHARPNESS_END and doesn't
        move if the next step is Out of Bound
        """
        # update pos and sharpness values
        self.log(1, Style.CYAN, "init sharpness STEP")
        self.state.prev_pos, self.state.cur_pos = self.state.cur_pos, self.stage.z
        self.state.prev_sharpness, self.state.cur_sharpness = self.state.cur_sharpness, sharpness(self.camera_img)
        self.log(3, Style.CYAN, f"Sharp={self.state.cur_sharpness}\tZ={self.state.cur_pos}")

        # Have we found the maximum yet ?
        if self.state.cur_sharpness <= self.state.prev_sharpness:
            # Good ending: YES :)
            self.stage.goto_z(self.state.prev_pos, busy=True)
            self._wait_for_move_then_transition_to(States.SHARPNESS_END, 10_000)
            self.log(2, Style.CYAN, "Local maximum found")
            self.log(4, Style.CYAN, "Next state = sharpness END")
            return

        next_pos = self.state.cur_pos + self.state.direction * self.config.stack_step
        if self.is_out_of_bounds(next_pos):
            # Bad Ending: No, we can't find the maximum :(
            self.state.is_stack_valid = False
            self.log(4, Style.CYAN, "Next state = sharpness END")
            self._transition_to(States.SHARPNESS_END)
            return
        # Neutral Ending: No, but let's keep searching
        self.stage.goto_z(next_pos, busy=True)
        self._wait_for_move_then_transition_to(States.SHARPNESS_STEP, 10_000)

    def _sharpness_end(self):
        """
        One of the behaviors of the scanner's state machine
        """
        # skip invalid stack
        self.log(3, Style.BLU, "Maximum reached, writing img file.")
        if self.save_detection_frames:
            imdir = self._get_scan_path() / "detection_frames"
            os.makedirs(imdir, exist_ok=True)
            impath = imdir.joinpath(f"X{self.stage.x:06d}_"
                                    f"Y{self.stage.y:06d}_"
                                    f"Z{self.stage.z:06d}_"
                                    f"S{self.state.prev_sharpness:f}.jpg")
            cv2.imwrite(str(impath), self.camera_img)
        if not self.state.is_stack_valid:
            # transition directly to stack_exposure
            self._transition_to(States.STACK_EXPOSURE)
            return
        if isinstance(self.detector, Detector):
            unfiltered = self.detector.detect(self.camera_img)
            objects = [obj for obj in unfiltered if obj[2] > 0.5]
            # TODO: set the threshold as its own setting     ↑↑↑
        else:
            objects = detect_white_objects(self.camera_img, 3)


        if objects:
            # if there are objects of interest, take the stack
            self.log(3, Style.BLU, f"{len(objects)} OBJECTS DETECTED AT"
                                   f" {self.stage.x, self.stage.y}")
            self.state.detection_boxes = objects
        else:
            self.log(3, Style.BLU, f"nothing at {self.stage.x, self.stage.y}")
            self.state.is_stack_valid = False
        self.stage.goto_z(self.state.initial_Z, busy=True)
        self._wait_for_move_then_transition_to(States.STACK_EXPOSURE, 10_000)

    def _stack_exposure(self):
        """
        One of the behaviors of the scanner's state machine
        """
        # All exposures done? (or need to skip the stack ?)
        skip = self.skip_stacks or not self.state.is_stack_valid
        if skip or self.state.exposure_idx == self.state.num_exposures:
            print("STACK: Skipped" if skip else "STACK: All exposures done")
            self.state.stack_x += 1
            if self.state.stack_x >= self.state.num_steps_x:
                self.state.stack_y += 1
                if self.state.stack_y >= self.state.num_steps_y:
                    self.state.zone_idx += 1
                    self._transition_to(States.ZONE)
                    return
                self.state.stack_x = 0
            self._transition_to(States.STACK_INIT)
            return
        # Get the current exposure time
        exposure_time = self.config.exposure_times[self.state.exposure_idx]
        # Set the exposure time
        self.camera.set_exposure(exposure_time)
        self._wait_for_exposure_then_transition_to(States.STACK_Z, exposure_time)

    def _stack_z(self):
        """
        One of the behaviors of the scanner's state machine
        """
        # Reset images
        self.state.focus_idx = 0
        self.state.num_focus = (self.config.stack_height + self.config.z_margin) // self.config.stack_step
        # Make path
        self._get_raw_path().mkdir(parents=True, exist_ok=True)
        # Move to start
        self.state.z_orig = self.stage.z
        _, dx, dy = self._get_stack_offset()
        self.stage.goto_z(self._get_corrected_z(dx, dy), busy=True)
        self._wait_for_move_then_transition_to(States.STACK_IMAGE, 10000)

    def _stack_image(self):
        """
        One of the behaviors of the scanner's state machine
        """
        save_path = (self._get_raw_path()
                     / f"X{self.stage.x:06d}_"
                       f"Y{self.stage.y:06d}_"
                       f"Z{self.stage.z:06d}.jpg")
        skio.imsave(str(save_path), self.camera_img[..., ::-1], check_contrast=False, quality=90)
        self.state.focus_idx += 1
        # Stack done?
        # print(self.state.focus_idx, self.state.num_focus, self.state.image_idx, self.state.num_exposures)
        if self.state.focus_idx > self.state.num_focus:
            self.queue.put(
                (
                    str(self._get_raw_path()),
                    str(self._get_stack_filename(self.stage.x, self.stage.y)),
                    self.state.detection_boxes
                )
            )
            self.state.image_idx += 1
            self.state.exposure_idx += 1
            self.stage.goto_z(self.state.z_orig, busy=True)
            self._wait_for_move_then_transition_to(States.STACK_EXPOSURE, 50 * self.state.num_focus)
        # Move and take next
        else:
            self.stage.move_z(self.config.stack_step, busy=True)
            self._wait_for_move_then_transition_to(States.STACK_IMAGE, 1000)

    def _done(self):
        """
        One of the behaviors of the scanner's state machine
        """
        self.state.end_time = datetime.now()
        self.queue.put("terminate")
        self.parallel_process.join()
        self._transition_to(States.IDLE)

    def loop(self, frame):
        # If cancelled, goto done
        if self.scan_cancelled:
            print("\033[31m===============CANCELED===============\033[0m\n")
            self._transition_to(States.DONE)
            self.scan_cancelled = False
        # The current state
        state = self.state.state
        # Get image every loop
        # frame = self.camera.latest_image(with_exposure=True)
        if frame is not None:
            self.camera_img, self.camera_exposure = frame

        self.log(10, "", f"SCANNER: {state}")
        # TODO: Add bounds check to prevent the camera head from coliding with
        #       the bed (Possible if the sample tray is very thick)
        state = self.state.state
        # Overall the states are in logical order
        # but a graph of the state machine can help
        if state == States.IDLE:
            self._idle()
        elif state == States.WAIT:
            self._wait()
        elif state == States.INIT:
            self._init()
        elif state == States.ZONE:
            self._zone()
        elif state == States.STACK_INIT:
            self._stack_init()
        elif state == States.SHARPNESS_INIT_A:
            self._sharpness_init_a()
        elif state == States.SHARPNESS_INIT_B:
            self._sharpness_init_b()
        elif state == States.SHARPNESS_STEP:
            self._sharpness_step()
        elif state == States.SHARPNESS_END:
            self._sharpness_end()
        elif state == States.STACK_EXPOSURE:
            self._stack_exposure()
        elif state == States.STACK_Z:
            self._stack_z()
        elif state == States.STACK_IMAGE:
            self._stack_image()
        elif state == States.DONE:
            self._done()

def sharpness(img) -> int:
    return np.sum(cv2.Laplacian(img[::4, ::4, ...], -1))

def create_detector(sample_step):
    params = cv2.SimpleBlobDetector.Params()
    # Color
    params.filterByColor = True
    params.blobColor = 255
    params.thresholdStep = 8
    # Area
    params.filterByArea = True
    params.minArea = 20 * 20 * pow(sample_step, -2)
    params.maxArea = 300 * 300 * pow(sample_step, -2)
    # Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.3
    return cv2.SimpleBlobDetector.create(params)

def detect_white_objects(img, sample_step):
    detector = create_detector(sample_step)
    down_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[::sample_step,::sample_step]
    kp = detector.detect(down_sample)
    return len(kp)
