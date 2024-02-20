import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from marshmallow import EXCLUDE
from marshmallow_dataclass import class_schema


@dataclass(kw_only=True)
class BaseModel:

    class Meta:
        ordered = True
        unknown = EXCLUDE

    @property
    @classmethod
    def schema(cls, *args, **kwargs):
        return class_schema(cls)(*args, **kwargs)

    def dump(self):
        return class_schema(self.__class__)().dump(self)

    def dumps(self, indent=4):
        return class_schema(self.__class__)().dumps(self, indent=indent)

    @classmethod
    def load(cls, data):
        return class_schema(cls)().load(data)

    @classmethod
    def loads(cls, json):
        return class_schema(cls)().loads(json)

    def save(self, path: Union[Path, str], indent=4):
        path = Path(path)
        if not path.parent.exists():
            print(f"Creating configuration folder: {path.parent}")
            os.mkdir(path.parent)
        with open(path, "w") as fp:
            metadata = class_schema(self.__class__)().dumps(self, indent=indent)
            fp.write(metadata)

    @classmethod
    def open(cls, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        instance = class_schema(cls)().loads(path.read_text())
        return instance
