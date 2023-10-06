import os
import multiprocessing as mp
import skimage.io as skio
import numpy as np
import datetime as dt
from shutil import rmtree
from pathlib import Path
from sashimi import utils, helicon_stack

# TODO: make an ETA function


def clip(n, mini=0, maxi=None):
    if n < mini:
        return n
    if maxi is None or n <= maxi:
        return n
    return n
    
    
def measure_sharpness(img):
    img = img[::4, ::4, ...]
    sharpness = []
    for i in range(3):
        dx = np.diff(img, axis=1)[1:, :, i]  # remove the first row
        dy = np.diff(img, axis=0)[:, 1:, i]  # remove the first column
        dnorm = np.sqrt(dx ** 2 + dy ** 2)
        sharpness.append(np.average(dnorm))
    return sharpness


def remove_folder(folder):
    for files in os.listdir(folder):
        path = os.path.join(folder, files)
        try:
            rmtree(path)
        except OSError:
            os.remove(path)


def s2hms(s: int) -> [int, int, int]:
    m = s // 60
    s = s - 60 * m
    h = m // 60
    m = m - 60 * h
    return h, m, s


class Scanner(object):
    def __init__(self, controller):
        # instances
        self.controller = controller
        self.stage = self.controller.stage
        self.camera = self.controller.camera
        self.config = self.controller.config

        self.frame_duration_ms = self.controller.frame_duration_ms
        self.auto_f_stack = self.controller.auto_f_stack
        self.remove_raw = self.controller.remove_raw
        self.auto_quit = self.controller.auto_quit
        self.scan_dir = self.controller.save_dir
        self.selected_scan = self.controller.selected_scan
        self.multi_exp = self.controller.multi_exp
        self.fs_folder = self.controller.save_dir.joinpath("f_stacks")
        
        # parameters and variables
        if self.multi_exp:
            self.fs_exp_folders = [self.fs_folder.joinpath(f"E{exp}") for exp in self.multi_exp]
        self.X_STEP = 1700
        self.Y_STEP = 1700
        self.stack_count = None
        self.current_stack = 0
        self.total_stacks = 0
        self.current_pic_count = 0
        self.total_pic_count = 0
        self.is_multi_scanning = False
        self.parallel_process = None
        self.queue = None
        self.update_stack_count()
        self.update_total_pic_count()

        self.summary = {
            'save_dir': self.controller.save_dir,
            'language': self.controller.lang,
            'layout': self.controller.layout,
            'auto_f_stack': self.auto_f_stack,
            'remove_raw': self.remove_raw,
            'auto_quit': self.auto_quit,
            'lowest_z': self.controller.lowest_z,
            'exposure (µs)': self.config.exposure_time if self.multi_exp is None else self.multi_exp,
            'stack height (µm)': self.config.stack_height,
            'XY_step (µm)': (self.X_STEP, self.Y_STEP),
            'stack_step (µm)': self.config.stack_step
        }
        
    def lowest_corner(self) -> int:
        current_scan = self.selected_scan()
        fl = current_scan['FL']
        br = current_scan['BR']
        blz = current_scan['BL_Z']
        flz = fl[2]
        brz = br[2]
        frz = flz - brz + blz
        mini = min((blz, brz, flz, frz))
        if mini < 0:
            mini = 0
        return mini
    
    def get_corrected_z(self, dx, dy):
        if self.controller.lowest_z:
            # 'Dumb-but-works' correction
            new_z = self.lowest_corner()
        else:
            # 'Smart' correction
            dz_dx, dz_dy = self.selected_scan()['Z_corrections']
            z_correction = int(dz_dx * dx + dz_dy * dy)
            new_z = self.selected_scan()['FL'][2] + z_correction
        return clip(new_z - self.config.z_margin)
        
    def update_stack_count(self):
        self.stack_count = self.config.stack_height // self.config.stack_step

    def update_total_pic_count(self):
        if self.multi_exp:
            pps = self.stack_count * len(self.multi_exp)
        else:
            pps = self.stack_count
        
        total_stacks = 0
        for scan in self.config.scans:
            x_steps, y_steps = self.step_nbr_xy(scan)
            total_stacks += x_steps * y_steps
        
        self.total_pic_count = pps * total_stacks
    
    def step_nbr_xy(self, scan) -> (int, int):
        x_steps = 1 + (scan['BR'][0] - scan['FL'][0]) // self.X_STEP
        y_steps = 1 + (scan['BR'][1] - scan['FL'][1]) // self.Y_STEP
        return x_steps, y_steps
    
    def multi_scan(self):
        self.is_multi_scanning = True
        self.current_pic_count = 0
        self.update_total_pic_count()
        self.summary['scan_dates'] = []
        self.controller.selected_scan_number = 1
        self.controller.interrupt_flag = False

        if self.controller.save_dir.exists() and len(list(self.controller.save_dir.iterdir())) > 0:
            if self.controller.do_overwrite:
                utils.remove_folder(self.controller.save_dir)
                os.makedirs(self.controller.save_dir)
            else:
                self.controller.save_dir = utils.make_unique_subdir(self.controller.save_dir.parent)
        
        if self.auto_f_stack:
            os.makedirs(self.fs_folder)
            mp.set_start_method("spawn")
            self.queue = mp.Queue()
            error_logs = self.controller.save_dir.joinpath('error_logs.txt')
            if error_logs.exists():
                os.remove(error_logs)
            arguments = (self.queue, error_logs, self.multi_exp, self.controller.remove_raw)
            self.parallel_process = mp.Process(target=helicon_stack.parallel_stack, args=arguments)
            self.parallel_process.start()

        for n in range(len(self.config.scans)):
            if not self.is_multi_scanning:
                break
            scan_name = f"scan{n + 1}"
            scan_dir = Path(self.controller.save_dir).joinpath(scan_name)
            os.makedirs(scan_dir)
            self.controller.selected_scan_number = n + 1
            self.summary['scan_dates'].append(dt.datetime.now(tz=dt.timezone(dt.timedelta(hours=2))))
            self.scan(scan_dir)
        self.summary['scan_dates'].append(dt.datetime.now(tz=dt.timezone(dt.timedelta(hours=2))))
        self.make_scan_summary()
        
        if self.auto_f_stack:
            self.queue.put('terminate')
            self.parallel_process.join()

        self.is_multi_scanning = False
        if self.auto_quit:
            # in case of user interruption:
            self.controller.interrupt_flag = True if self.controller.quit_requested else False
            self.controller.quit_requested = True

    def scan(self, scan_dir):
        selected_scan = self.controller.selected_scan()
        fl = selected_scan['FL']
        br = selected_scan['BR']
        try :
            assert (br[0] > fl[0])
            assert (br[1] > fl[1])
        except AssertionError:
            print("INVALID SCAN COORDINATES")
            return
        
        os.makedirs(scan_dir, exist_ok=True)
        
        self.stage.goto(fl)
        self.stage.wait_until_position(10000)
        x_steps, y_steps = self.step_nbr_xy(selected_scan)
        self.total_stacks = x_steps * y_steps
        
        # Start scanning
        self.current_stack = 0
        for yi in range(y_steps):
            for xi in range(x_steps):
                self.current_stack += 1

                if self.check_for_escape():
                    print('escaping scan()')
                    return
                
                dx, dy = self.X_STEP * xi, self.Y_STEP * yi
                du = [fl[0] + dx, fl[1] + dy, fl[2]]
                self.stage.goto(du)
                self.stage.wait_until_position(1000)
                self.take_stack(dx, dy, scan_dir)

    def take_stack(self, dx, dy, scan_dir):
        # Create directory to save stack
        xy_folder = Path(scan_dir).joinpath(f"X{self.stage.x//10:05d}_Y{self.stage.y//10:05d}")
        os.makedirs(xy_folder, exist_ok=True)
        if self.multi_exp:
            for exp in self.multi_exp:
                os.makedirs(xy_folder.joinpath(f"E{exp}"), exist_ok=True)
        
        z_orig = self.stage.z
        self.stage.goto_z(self.get_corrected_z(dx, dy))
        self.stage.wait_until_position(1000)
        
        exp_values = self.multi_exp if self.multi_exp else (self.config.exposure_time,)
        for i in range(self.stack_count):
            for exp in exp_values:
                self.current_pic_count += 1
                if self.check_for_escape():
                    print('escaping take_stack()')
                    return
                # set exposure and take a picture
                self.camera.set_exposure(exp)
                img = self.wait_until_exposure(exp, 300)
                self.show_image(img)
                
                # save the picture
                if self.multi_exp is not None:
                    save_path = xy_folder.joinpath(f"E{exp}")
                else:
                    save_path = xy_folder
                save_path = save_path.joinpath(f"X{self.stage.x//10:05d}_"
                                               f"Y{self.stage.y//10:05d}_"
                                               f"Z{self.stage.z//10:05d}.jpg")
                skio.imsave(str(save_path), img[..., ::-1], check_contrast=False, quality=90)
            
            self.stage.move_z(self.config.stack_step)
            self.stage.wait_until_position(100)
        
        if self.auto_f_stack:
            self.queue.put((str(xy_folder), str(self.fs_folder)))

        self.camera.set_exposure(exp_values[0])
        self.stage.goto_z(z_orig)
        self.stage.wait_until_position(50 * self.stack_count)
        
    def find_floor(self):
        z_orig = self.stage.z
        self.stage.goto_z(100)
        self.stage.wait_until_position(500)
        sharpness = []
        for i in range(100):
            img = self.camera.latest_image()
            # self.show_image(img)
            sh = measure_sharpness(img)
            sharpness.append(sh)
            print(sh)
            self.stage.move_z(20)
            self.stage.wait_until_position(200)
        sharpness = np.asarray(sharpness)
        print(np.max(sharpness, axis=0))
        print(np.argmax(sharpness, axis=0) * 20 + 100)
        self.stage.goto_z(z_orig)
    
    def wait_until_exposure(self, exp, ms):
        img = None
        for i in range(ms//self.frame_duration_ms):
            img, img_exp = self.camera.latest_image(with_exposure=True)
            if exp == img_exp:
                
                return img
            else:
                self.controller.display(img)
                self.controller.wait(display=False)
        print(f'desired exposure was not reached in {ms}ms')
        return img

    def show_image(self, img):
        if img is None:
            return
        self.controller.display(img)
    
    def check_for_escape(self):
        if self.is_multi_scanning and not self.controller.quit_requested:
            return False
        if self.auto_quit:
            self.controller.quit_requested = True
        self.is_multi_scanning = False
        self.controller.interrupt_flag = True
        return True

    def make_scan_summary(self):
        summary_path = self.controller.save_dir.joinpath('summary.txt')
        with open(summary_path, mode='x') as summary:
            if self.controller.interrupt_flag:
                summary.write('///////////THE SCANS WERE INTERRUPTED BEFORE FINISHING!!!///////////\n\n')
            summary.write('This is the summary of this multi-scan folder.\n'
                          'Here are some parameters :')
            param_list = [
                'save_dir',
                'language',
                'layout',
                'auto_f_stack',
                'remove_raw',
                'auto_quit',
                'lowest_z',
                'exposure (µs)',
                'stack height (µm)',
                'XY_step (µm)',
                'stack_step (µm)'
            ]
            for param in param_list:
                summary.write(f"{param} = {self.summary[param]}\n")
            dates = self.summary['scan_dates']
            deltas = [dates[n+1] - dates[n] for n in range(len(dates) - 1)]
            total_pics = 0
            
            if self.multi_exp is None:
                exp_count = 1
            else:
                exp_count = len(self.multi_exp)
            summary.write('Here are statistics about the scans.')
            for n in range(len(deltas)):
                scan = self.config.scans[n]
                steps_x, steps_y = self.step_nbr_xy(scan)
                stack_nbr = steps_x * steps_y
                pic_nbr = stack_nbr * self.stack_count
                total_pics += pic_nbr
                h, m, s = s2hms(deltas[n].seconds)
                summary.write(
                    f"scan{n + 1} started at {dates[n]}, lasted {h}h {m}min {s}s and took :\n"
                    f"{stack_nbr} stacks x {exp_count} exposures x {self.stack_count} heights = {pic_nbr} pictures.\n\n"
                )
            delta = dates[-1] - dates[0]
            h, m, s = s2hms(int(delta.total_seconds()))
            summary.write(f'Overall, the task ended at {dates[-1]} and lasted {h}h {m}min and {s}s.\n')
