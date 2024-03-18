import threading

import cv2
from pypylon import pylon

# TODO: look into pylon.ImageFileFormat

capture_lock = threading.Lock()


class CaptureThread(threading.Thread):
    def __init__(self,
                 camera: pylon.InstantCamera,
                 converter: pylon.ImageFormatConverter,
                 controller,
                 group=None,
                 target=None,
                 name=None,
                 args=(),
                 kwargs=None,
                 verbose=None):
        super(CaptureThread, self).__init__()
        self.camera = camera
        self.converter = converter
        self.controller = controller
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
            if grab_result.GrabSucceeded():
                with capture_lock:
                    self.image = cv2.rotate(self.converter.Convert(grab_result).Array, cv2.ROTATE_180)
                    self.exposure = grab_result.ChunkExposureTime.Value
                # Check if quit
                if self.controller.quit_requested:
                    grab_result.Release()
                    break
            grab_result.Release()
        return


class Camera(object):
    def __init__(self, controller):
        self.image = None
        self.camera = None
        self.controller = controller
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.capture_thread = None

    def start(self):
        # Open camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.load_camera_settings()
        self.capture_thread = CaptureThread(self.camera, self.converter, self.controller)
        self.capture_thread.start()
    
    def load_camera_settings(self):
        n_map = self.camera.GetNodeMap()
        n_map.GetNode("ExposureMode").SetValue("Timed")
        pylon.FeaturePersistence.Load("nodeFile.pfs", n_map)
        self.camera.StaticChunkNodeMapPoolSize = self.camera.MaxNumBuffer.GetValue()
        self.camera.ChunkModeActive = True
        self.camera.ChunkSelector = "ExposureTime"
        self.camera.ChunkEnable = True
    
    def stop(self):
        self.camera.StopGrabbing()
        self.camera.ChunkModeActive = False
        self.camera.Close()

    def latest_image(self, with_exposure=False):
        img = None
        with capture_lock:
            if self.capture_thread.image is not None:
                img = self.capture_thread.image.copy()
                exp = int(self.capture_thread.exposure)
        if with_exposure:
            return img, exp
        else:
            return img

    def set_exposure(self, value):
        self.camera.ExposureTime.SetValue(value)

    def set_gain(self, value):
        self.camera.Gain.SetValue(value)


