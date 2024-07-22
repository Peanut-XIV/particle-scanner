from pathlib import Path
import csv
import torch
from torch import Tensor
import cv2


def labels_from_file(path: Path) -> list[str]:
    indices = []
    names = []
    with open(path, "r", encoding="utf8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            idx, name = row
            indices.append(int(idx))
            names.append(name)
    output = [f"label{i}" for i in range(max(indices) + 1)]
    if 0 not in indices:
        output[0] = "background"
    for idx, name in zip(indices, names):
        output[idx] = name
    return output

def load_model(model_dir: Path, device) -> torch.nn.Module:
    if not model_dir.exists():
        raise NotADirectoryError(f"{str(model_dir)} does not exist")
    model_path_candidates = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
    if len(model_path_candidates) != 1:
        raise FileNotFoundError(f"expected to find 1 .pt or .pth file but found {len(model_path_candidates)}")
    model_path = model_path_candidates[0]
    model = torch.load(str(model_path), map_location=device)
    # model.to(device=device)
    model.eval()
    return model


class Detector:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.model = load_model(model_dir, self.device)
        labels_path = model_dir.joinpath("labels.txt")
        self.classes: list[str] = labels_from_file(labels_path)

    def _convert_img(self, ndarray, from_bgr: bool) -> Tensor:
        """
        Converts an image from an ndarray of type int or float and shape
        [Y , X, 3] to a torch tensor of type float and shape [3, Y, X], and
        stores it in the same device as the model.
        """
        try:
            assert ndarray.shape[2] == 3
            assert len(ndarray.shape) == 3
        except AssertionError:
            raise ValueError("Expected an ndarray of shape (_, _, 3) but got shape = {ndarray.shape}")
        if from_bgr:
            ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(ndarray)
        if img.dtype.is_complex:
            # how did you manage that ??
            raise TypeError("img datatype is 'complex'")
        if not img.dtype.is_floating_point:
            img = img.to(dtype=torch.float) / 255
        img = torch.transpose(img, 2, 0)
        return img

    def detect(self, img, from_bgr=False) -> list[Tensor, str, float]:
        # convert the image to a shape that torch can handle
        img = self._convert_img(img, from_bgr)
        img.to(device=self.device)
        with torch.inference_mode():
            logits = self.model([img])[0]  # model takes a list of tensors as input. This list is of length one.
        boxes = logits[0]["boxes"]
        labels = map(self.classes.__getitem__, logits[0]["labels"])
        scores = logits[0]["scores"]
        return list(zip(boxes, labels, scores))
