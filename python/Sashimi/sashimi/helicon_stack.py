import os
import sys
import subprocess
from pathlib import Path
from glob import glob
from PIL import Image
from time import sleep
from sashimi import utils


def get_helicon_focus():
    helicon_focus = r"C:\\Program Files\\Helicon Software\\Helicon Focus 7\\HeliconFocus.exe"
    if not os.path.exists(helicon_focus):
        helicon_focus = r"C:\Program Files\Helicon Software\Helicon Focus 8\HeliconFocus.exe"
        if not os.path.exists(helicon_focus):
            raise ResourceWarning("Helicon Focus was not found")
    return helicon_focus


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
            

def parallel_stack(queue, error_logs, exposures, remove_raw=False):
    with open(error_logs, mode='w', encoding='UTF-8') as file:
        sys.stderr = sys.stdout = file
        while True:
            if queue.empty():
                sleep(0.5)
                continue
    
            msg = queue.get()
            print(msg)
            if msg == "terminate":
                break
                
            xy_folder = Path(msg[0])  # save_dir/scanX/X__Y__/   (...X__Y__Z__.jpg)    or (...E__/X__Y__Z__.jpg)
            output_folder = Path(msg[1])  # save_dir/f_stacks/   (...scanX/X__Y__.jpg) or (...E__/scanX/X__Y__.jpg)
            scan_name = xy_folder.parent.stem
            img_name = xy_folder.stem
            
            if exposures is None:
                gen_stack(xy_folder, output_folder, scan_name, img_name)
            else:
                for exp in exposures:
                    from_ = xy_folder.joinpath(f"E{exp}")
                    to_ = xy_folder.joinpath(f"E{exp}")
                    gen_stack(from_, to_, scan_name, img_name)

            if remove_raw:
                utils.remove_folder(xy_folder)


def gen_stack(from_: Path, to_: Path, scan_name: str, img_name: str):
    tiff_path = to_.joinpath(scan_name, f"{img_name}.tiff")
    png_path = to_.joinpath(scan_name, f"{img_name}.png")
    command = [get_helicon_focus(), "-silent", f"{from_}", "-mp:0", "-rp:4", f"-save:{tiff_path}"]
    subprocess.run(command)
    img = Image.open(tiff_path)
    img.load()
    img.save(png_path)
    os.remove(tiff_path)
    return
