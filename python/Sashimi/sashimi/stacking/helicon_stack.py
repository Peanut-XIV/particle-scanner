import os
import shutil
import sys
import subprocess
import traceback
from pathlib import Path
from glob import glob
from typing import Union, Optional
from time import sleep
from typing import Optional, Iterable

from sashimi import utils

from PIL import Image
import numpy as np


def clip(x, a: float | int | list[float | int] | tuple[float | int], b: None | float | int = None) -> float | int:
    if b is None:
        if not isinstance(a, (list, tuple)):
            mini = 0
            maxi = a
        else:
            if len(a) != 2:
                raise ValueError(f"argument `a` is not of length 2: {a}")
            mini = a[0]
            maxi = a[1]
    else:
        mini = a
        maxi = b
    return min(max(x, mini), maxi)


def crop_picture(img, box):
    y_max, x_max, _ = img.shape
    x1, x2, y1, y2 = box
    x1 = clip(x1, 0, x_max)
    x2 = clip(x2, 0, x_max)
    y1 = clip(y1, 0, y_max)
    y2 = clip(y2, 0, y_max)
    crop = img[y1:y2, x1:x2, :]
    return crop


def get_helicon_focus():
    # TODO: Add Helicon stack location in config file
    possible_paths = [
        "/Applications/HeliconFocus.app/Contents/MacOS/HeliconFocus",
        r"C:\\Program Files\\Helicon Software\\Helicon Focus 7\\HeliconFocus.exe",
        r"C:\Program Files\Helicon Software\Helicon Focus 8\HeliconFocus.exe",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise ResourceWarning("Helicon Focus was not found")


def get_focus_stack():
    return r"C:\Users\ross.marchant\bin\focus-stack\focus-stack.exe"


def stack(_dir):
    helicon_focus = get_helicon_focus()

    dirs = sorted([d for d in glob(os.path.join(_dir, "*")) if os.path.isdir(d)])
    for d in dirs:
        print(d)
        fns = sorted(glob(os.path.join(d, "*.jpg")))
        if len(fns) == 0:
            continue
        command = [helicon_focus,
                   "-silent",
                   f"{d}",
                   "-dmap",
                   "-rp:4"]
        print(command)
        subprocess.run(command)


def radius_test(stacks_dir, save_dir):
    helicon_focus = get_helicon_focus()
    dirs = sorted([d for d in glob(os.path.join(stacks_dir, "*")) if os.path.isdir(d)])
    for radius in range(0, 11):
        for d in dirs:
            folder_name = f"radius_{radius}px"
            save_name = save_dir.joinpath(folder_name, Path(d).stem)
            images = sorted(glob(os.path.join(d, "*.jpg")))
            if len(images) == 0:
                continue
            command = [helicon_focus,
                       "-silent",
                       f"{d}",
                       "-mp:0",
                       f"-rp:{radius}",
                       f"-save:{save_name}.jpg"]
            print(command)
            subprocess.run(command)


def stack_from_to(stacks_dir, save_dir):
    """
    :param stacks_dir:
    :param save_dir: (must include sub folder if multiple scans)
    """
    helicon_focus = get_helicon_focus()

    dirs = sorted([d for d in glob(os.path.join(stacks_dir, "*")) if os.path.isdir(d)])
    for d in dirs:
        print(d)
        save_subdir = save_dir.joinpath(Path(d).stem)
        images = sorted(glob(os.path.join(d, "*.jpg")))
        if len(images) == 0:
            continue

        command = [helicon_focus,
                   "-silent",
                   f"{d}",
                   "-mp:0",
                   "-rp:4",
                   f"-save:{save_subdir}.jpg"]
        print(command)
        subprocess.run(command)


def stack_for_multiple_exp(scan_path: Path, f_stacks_path: Path, exp_values: list, do_overwrite=False):
    helicon_focus = get_helicon_focus()

    scan_name = scan_path.stem
    for exp in exp_values:
        output_folder = f_stacks_path.joinpath(f"E{exp}", scan_name)
        os.makedirs(output_folder, exist_ok=do_overwrite)

        xy_folders = sorted([d for d in glob(os.path.join(scan_path, "*")) if os.path.isdir(d)])
        for folder in xy_folders:
            stack_name = Path(folder).stem
            stacked_file_path = output_folder.joinpath(stack_name)
            dd = Path(folder).joinpath(f"E{exp}")
            images = sorted(glob(os.path.join(dd, "*.jpg")))
            if len(images) == 0:
                continue

            command = [
                helicon_focus,
                "-silent",
                f"{dd}",
                "-mp:0",
                "-rp:4",
                f"-save:{stacked_file_path}.jpg"
            ]
            print(command)
            subprocess.run(command, shell=True)

def parallel_stack(queue, error_logs, stack_method="focus_stack", remove_raw=False):
    if not Path(error_logs).exists():
        write_mode = "x"
    else:
        write_mode = "w"
    os.makedirs(Path(error_logs).parent, exist_ok=True)
    with open(error_logs, mode=write_mode, encoding='utf_8') as file:
        sys.stderr = file
        sys.stdout = file
        while True:
            if queue.empty():
                sleep(0.5)
                continue
            msg = queue.get()
            if msg == "terminate":
                break

            raw_folder = Path(msg[0])
            image_path = msg[1]

            # the boxes in which objects of iterest were detected
            if len(msg) > 2:
                detection_boxes = msg[2]
            else:
                detection_boxes = None
            try:
                if stack_method == "helicon":
                    stack_with_helicon(raw_folder, image_path, detection_boxes)
                elif stack_method == "focus_stack":
                    stack_with_focus_stack(raw_folder, image_path)
                else:
                    raise ValueError(f"Invalid stack method name: {stack_method}")
            except:
                print(f"Error processing {raw_folder}")
                traceback.print_exc()
            if remove_raw:
                shutil.rmtree(raw_folder)


def stack_with_helicon(raw_images_path: Union[str, Path],
                       image_path: Union[str, Path],
                       boxes: list[list[int], str, float] | None = None):
    if isinstance(raw_images_path, str):
        raw_images_path = Path(raw_images_path)
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    tiff_path = f"{image_path}.tiff"
    png_path = f"{image_path}.png"
    command = [get_helicon_focus(), "-silent", f"{str(raw_images_path)}", "-mp:0", "-rp:4", f"-save:{tiff_path}"]
    subprocess.run(command)
    img = Image.open(tiff_path)
    img.load()
    if boxes is not None:
        pixels = np.array(img)
        for box, label, _ in boxes:
            crop = crop_picture(pixels, box)
            # save crop to the correct dir
            crop_name = image_path.stem + f"_[{x1},{x2},{y1},{y2}].png"
            crop_dir = image_path.parent.parent.joinpath(label or "unknown_label")
            crop_path = crop_dir.joinpath(crop_name)
            Image.fromarray(crop).save(crop_path)
    img.save(png_path)
    os.remove(tiff_path)
    return


def stack_with_focus_stack(raw_images_path: Union[str, Path], image_path: Union[str, Path]):
    if isinstance(raw_images_path, str):
        raw_images_path = Path(raw_images_path)
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Get images
    images = [str(f) for f in raw_images_path.glob("*.jpg") if f.is_file()]
    # File names
    png_path = f"{image_path}.png"
    depthmap_path = image_path.parent.parent / "depthmap" / f"{image_path.stem}.png"
    depthmap_path.parent.mkdir(parents=True, exist_ok=True)
    command = [get_focus_stack(), *images, f"--output={png_path}", "--no-whitebalance", "--no-contrast", "--nocrop", f"--depthmap={depthmap_path}"]
    subprocess.run(command)
    return

if __name__ == "__main__":
    stack_with_helicon(Path(r"F:\Sashimi\test\Zone000\Exp02000\raw\Yi000001_Xi000000"),
              r"F:\Sashimi\test\Zone000\Exp02000\raw\Yi000001_Xi000000")
