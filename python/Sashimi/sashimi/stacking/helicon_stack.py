import os
import shutil
import sys
import subprocess
from pathlib import Path
from glob import glob
from typing import Union

from PIL import Image
from time import sleep
from sashimi import utils
import traceback


def get_helicon_focus():
    helicon_focus = r"C:\\Program Files\\Helicon Software\\Helicon Focus 7\\HeliconFocus.exe"
    if not os.path.exists(helicon_focus):
        helicon_focus = r"C:\Program Files\Helicon Software\Helicon Focus 8\HeliconFocus.exe"
        if not os.path.exists(helicon_focus):
            raise ResourceWarning("Helicon Focus was not found")
    return helicon_focus


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
            

def parallel_stack(queue, error_logs, remove_raw=False):
    with open(error_logs, mode='w', encoding='UTF-8') as file:
        sys.stderr = sys.stdout = file
        while True:
            if queue.empty():
                sleep(0.5)
                continue
            msg = queue.get()
            if msg == "terminate":
                break

            raw_folder = Path(msg[0])
            image_path = msg[1]

            try:
                # stack_with_helicon(raw_folder, image_path)
                stack_with_focus_stack(raw_folder, image_path)
            except:
                print(f"Error processing {raw_folder}")
                traceback.print_exc()

            shutil.rmtree(raw_folder)


def stack_with_helicon(raw_images_path: Union[str, Path], image_path: Union[str, Path]):
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
    depthmap_path = f"{image_path}_depthmap.png"
    command = [get_focus_stack(), *images, f"--output={png_path}", "--no-whitebalance", "--no-contrast", "--nocrop"]
    subprocess.run(command)
    return

if __name__ == "__main__":
    stack_with_helicon(Path(r"F:\Sashimi\test\Zone000\Exp02000\raw\Yi000001_Xi000000"),
              r"F:\Sashimi\test\Zone000\Exp02000\raw\Yi000001_Xi000000")