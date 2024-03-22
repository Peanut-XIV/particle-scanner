import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sashimi.configuration.base import BaseModel
from sashimi.hardware.camera import CameraConfiguration
from sashimi.hardware.scanner import ScannerConfiguration
from sashimi.hardware.stage import StageConfiguration


@dataclass
class Configuration(BaseModel):
    camera: CameraConfiguration = field(default_factory=CameraConfiguration)
    stage: StageConfiguration = field(default_factory=StageConfiguration)
    scanner: ScannerConfiguration = field(default_factory=ScannerConfiguration)

    def save_default(self):
        self.save(os.path.join(os.path.expanduser("~"), ".sashimi", "config.json"), 4)

    @staticmethod
    def load_default():
        config_file = os.path.join(os.path.expanduser("~"), ".sashimi", "config.json")
        if Path(config_file).exists():
            return Configuration.open(config_file)
        else:
            return Configuration()
