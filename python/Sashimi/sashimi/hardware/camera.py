import os.path
import threading
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import cv2
from pypylon import pylon

import sashimi.resources

# TODO: look into pylon.ImageFileFormat

capture_lock = threading.Lock()


@dataclass
class CameraConfiguration:
    exposure_time: int = 2000  # µs
    gain: float = 0.0
    blue: float = 0.0
    red: float = 0.0
    green: float = 0.0
    rescale: float = 0.5
    width: int = 2100  # in µm  # TODO: include this value in a config file
    height: int = 1760  # in µm


class CaptureThread(threading.Thread):
    def __init__(self,
                 camera: pylon.InstantCamera,
                 converter: pylon.ImageFormatConverter,
                 cancel_event: threading.Event,
                 rescale: float = 1.0,
                 group=None,
                 target=None,
                 name=None,
                 args=(),
                 kwargs=None,
                 verbose=None):
        super().__init__()
        self.camera = camera
        self.converter = converter
        self.cancel_event = cancel_event
        self.rescale = rescale
        self.target = target
        self.name = name
        self.image = None
        self.exposure = None

    # noinspection PyUnresolvedReferences
    def run(self):
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result is False:
                continue
            try:
                if grab_result.GrabSucceeded():
                    with capture_lock:
                        self.image = cv2.rotate(self.converter.Convert(grab_result).Array, cv2.ROTATE_180)
                        # if self.rescale != 1.0:
                        #     self.image = cv2.resize(self.image, (0, 0), fx=self.rescale, fy=self.rescale)
                        self.exposure = grab_result.ChunkExposureTime.Value
                    # Check if quit
                    if self.cancel_event.is_set():
                        grab_result.Release()
                        break
            except:
                print("Error: Failed to grab image")
            grab_result.Release()


class Camera:
    def __init__(self):
        self.pixel_size = 3.45
        self.objective_magnification = 4
        self.width = 2448
        self.height = 2048

        self.image = None
        self.camera = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.capture_thread = None
        self.cancel_event = threading.Event()

    def start(self):
        # Open camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.load_camera_settings()
        self.cancel_event.clear()

        self.capture_thread = CaptureThread(self.camera, self.converter, self.cancel_event)
        self.capture_thread.start()

    def load_camera_settings(self):
        n_map = self.camera.GetNodeMap()
        n_map.GetNode("ExposureMode").SetValue("Timed")
        node_file = "nodeFile.pfs"
        if os.path.exists(node_file):
            pylon.FeaturePersistence.Load(node_file, n_map)
        else:
            pylon.FeaturePersistence.Save(node_file, n_map)
        self.camera.StaticChunkNodeMapPoolSize = self.camera.MaxNumBuffer.GetValue()
        self.camera.ChunkModeActive = True
        self.camera.ChunkSelector = "ExposureTime"
        self.camera.ChunkEnable = True

    def stop(self):
        self.cancel_event.set()
        self.camera.StopGrabbing()
        self.camera.ChunkModeActive = False
        self.camera.Close()

    def latest_image(self, with_exposure=False):
        img = None
        with capture_lock:
            if self.cancel_event.is_set():
                return None
            if self.capture_thread.image is not None:
                img = self.capture_thread.image.copy()
                exp = int(self.capture_thread.exposure)
                if with_exposure:
                    return img, exp
                else:
                    return img
        return img

    def set_exposure(self, value):
        self.camera.ExposureTime.SetValue(value)

    def set_gain(self, value):
        self.camera.Gain.SetValue(value)


class DummyCamera(Camera):
    """
    a dummy camera class for development and testing purposes.
    """
    def __init__(self):
        super().__init__()
        self.exposure = 2000
        self.gain = 20
        im_path = str(files(sashimi.resources).joinpath("testscreen.png"))
        self.test_screen = get_test_img(im_path)

    def start(self):
        return

    def load_camera_settings(self):
        return

    def stop(self):
        return

    def latest_image(self, with_exposure=False):
        img = self.test_screen
        if with_exposure:
            return img, self.exposure
        return img

    def set_exposure(self, value):
        self.exposure = value

    def set_gain(self, value):
        self.gain = value


def get_test_img(im_path: str):
    img_mat = cv2.imread(im_path)
    return img_mat