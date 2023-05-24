import os
import time
from shutil import rmtree
from pathlib import Path
import skimage.io as skio
import numpy as np
from sashimi.helicon_stack import stack_from_to, stack_for_multiple_exp

# TODO: auto z_correction

# TODO: add an update_total_stacks_nbr() function
# TODO: measure the time needed to :
#  - take all the pictures of a stack
#  - stack these pictures together with Helicon Focus
# TODO: make an ETA function


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


class Scanner(object):
    def __init__(self, controller):
        self.controller = controller
        self.stage = self.controller.stage
        self.camera = self.controller.camera
        self.config = self.controller.config
        
        self.X_STEP = 1700
        self.Y_STEP = 1700
        
        self.auto_f_stack = self.controller.auto_f_stack
        self.remove_pics = self.controller.remove_pics
        
        self.auto_quit = self.controller.auto_quit
        self.save_dir = self.controller.save_dir
        self.scan_dir = self.save_dir
        self.scans = self.config.scans
        
        self.selected_scan = self.controller.selected_scan
        self.stack_count = None
        
        self.current_stack = 0
        self.total_stacks = 0
        self.current_pic_count = 0
        self.total_pic_count = 0
        
        self.is_scanning = False
        self.is_multi_scanning = False
        self.multi_exp = self.controller.multi_exp
        self.fs_folder = self.save_dir.joinpath("f_stacks")
        
        self.fs_exp_folders = [self.fs_folder.joinpath(f"E{exp}") for exp in self.multi_exp]
        self.reposition_offset = self.controller.reposition_offset

        self.update_stack_count()
        self.update_total_pic_count()
        
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
    
    def get_corrected_z(self):
        if self.controller.lowest_z:
            # 'Dumb-but-works' correction
            new_z = self.lowest_corner()
        else:
            # 'Smart' correction
            dz_dx, dz_dy = self.selected_scan()['Z_corrections']
            z_correction = int(dz_dx * dx + dz_dy * dy)
            new_z = self.selected_scan['FL'][2] + z_correction
        return new_z
        
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
        self.controller.selected_scan_number = 1
        self.current_pic_count = 0
        self.update_total_pic_count()
        os.makedirs(self.save_dir, exist_ok=True)

        for n, path in enumerate(self.scans):
            scan_name = f"scan{n + 1}"
            if not self.is_multi_scanning:
                return

            self.controller.selected_scan_number = n + 1
            self.stage.goto(self.selected_scan()['FL'])
            self.wait_ms_check_input(5000)
            self.scan_dir = Path(self.save_dir).joinpath(scan_name)
            self.scan()
            
            if self.auto_f_stack:
                self.focus_stack(scan_name)
                if self.remove_pics:
                    remove_folder(self.scan_dir)

        self.is_multi_scanning = False
        if self.auto_quit:
            # in case of user interruption:
            self.controller.interrupt_flag = True if self.controller.quit_requested else False
            self.controller.quit_requested = True

    def scan(self):
        # Reset the stack
        self.current_stack = 0
        
        # Create directory to store images
        os.makedirs(self.scan_dir, exist_ok=True)
        selected_scan = self.controller.selected_scan()
        
        # Move to the starting position
        self.stage.goto(selected_scan['FL'])
        self.stage.wait_until_position(10000)
        
        # Calculate the number of steps needed
        x_steps, y_steps = self.step_nbr_xy(self.selected_scan())
        self.total_stacks = (x_steps + 1) * (y_steps + 1)
        
        # Start scanning
        self.is_scanning = True
        for yi in range(y_steps):
            for xi in range(x_steps):
                self.current_stack += 1
                if self.check_for_escape():
                    print('escaping scan()')
                    return
                
                dx, dy = self.X_STEP * xi, self.Y_STEP * yi
                self.stage.goto_z(selected_scan['FL'][2])
                self.stage.goto_x(selected_scan['FL'][0] + dx)
                self.stage.goto_y(selected_scan['FL'][1] + dy)
                self.stage.wait_until_position(1000)
                self.wait_ms_check_input(300)
                self.take_stack(dx, dy)
        self.is_scanning = False

    def take_stack(self, dx, dy):
        # Create directory to save stack
        stack_folder = Path(self.scan_dir).joinpath(f"X{self.stage.x:06d}_Y{self.stage.y:06d}")
        os.makedirs(stack_folder, exist_ok=True)
        if self.multi_exp:
            for exp in self.multi_exp:
                os.makedirs(stack_folder.joinpath(f"E{exp}"), exist_ok=True)
        
        z_orig = self.stage.z
        self.stage.goto_z(self.get_corrected_z())
        self.wait_ms_check_input(100)
        stack_order = +1
        if self.config.top_down:  # Reposition the camera with downward travel to reduce the tilting of the camera
            self.stage.move_z(self.config.stack_height + self.reposition_offset)
            self.stage.move_z(-self.reposition_offset)
            stack_order = -1
        
        exp_values = self.multi_exp if self.multi_exp else (self.config.exposure_time,)
        for i in range(self.stack_count):
            for exp in exp_values:
                self.current_pic_count += 1
                if self.check_for_escape():
                    print('escaping take_stack()')
                    return
                
                # set exposure and take a picture
                self.camera.set_exposure(exp)
                self.wait_ms_check_input(300)
                img = self.camera.latest_image()
                self.show_image(img)
                
                # save the picture
                sub_folder = f"E{exp}/" if self.multi_exp else ""
                save_path = stack_folder.joinpath(f"{sub_folder}"
                                                  f"X{self.stage.x:06d}_"
                                                  f"Y{self.stage.y:06d}_"
                                                  f"Z{self.stage.z:06d}.jpg")
                skio.imsave(str(save_path), img[..., ::-1], check_contrast=False, quality=90)
            self.stage.move_z(self.config.stack_step * stack_order)

        # Return to base Z coordinate
        img = self.camera.latest_image()
        self.show_image(img)
        self.wait_ms_check_input(100)
        self.stage.goto_z(z_orig)
        self.wait_ms_check_input(50 * self.stack_count)

    def focus_stack(self, scan_name):
        if not self.is_multi_scanning:
            return
        scan_fs_dir = self.fs_folder.joinpath(scan_name)
        if self.multi_exp is None:
            stack_from_to(self.scan_dir, scan_fs_dir)
        else:
            stack_for_multiple_exp(self.scan_dir, self.fs_folder, self.multi_exp)
            
    def find_floor(self):
        z_orig = self.stage.z
        self.stage.goto_z(100)
        self.wait_ms_check_input(500)
        sharpness = []
        for i in range(100):
            img = self.camera.latest_image()
            self.show_image(img)
            sh = measure_sharpness(img)
            sharpness.append(sh)
            print(sh)
            self.stage.move_z(20)
            self.wait_ms_check_input(200)
        sharpness = np.asarray(sharpness)
        print(np.max(sharpness, axis=0))
        print(np.argmax(sharpness, axis=0) * 20 + 100)
        self.stage.goto_z(z_orig)

    def wait_ms_check_input(self, ms):
        self.controller.check_for_command(ms)

    def show_image(self, img):
        if img is None:
            return
        self.controller.display(img)
    
    def check_for_escape(self):
        if (not self.is_scanning) or (not self.is_multi_scanning) or self.controller.quit_requested:
            self.is_multi_scanning = False
            self.is_scanning = False
            if self.auto_quit:
                self.controller.quit_requested = True
            return True
        else:
            return False
        
        
