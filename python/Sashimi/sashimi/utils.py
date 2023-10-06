import os
import shutil
import datetime as dt
from pathlib import Path


def remove_folder(path):
	for file in os.listdir(path):
		subdir = path.joinpath(file)
		try:
			shutil.rmtree(subdir)
		except OSError:
			os.remove(subdir)


def make_unique_subdir(directory: str | Path = None):
	if directory is None:
		directory = Path.home().joinpath("Desktop", "sashimi")
	else:
		if type(directory) == str:
			directory = Path(directory)

	d = dt.datetime.now(tz=dt.timezone(dt.timedelta(hours=2)))
	subdir = f"{d.day}{d.month}{d.year}_{d.hour}{d.minute}"
	if directory.joinpath(subdir).exists():
		subdir = subdir + str(d.second)
	n = 0
	subdir_ = subdir
	while directory.joinpath(subdir_).exists():
		n += 1
		subdir_ = subdir + f"_{n}"
	output = directory.joinpath(subdir_)
	os.makedirs(output)
	return output


def is_valid_path(_path):
	try:
		_path = Path(_path).resolve()
	except(OSError, RuntimeError):
		print("ERROR: invalid syntax and/or forbidden characters")
		return False
	
	if _path.is_dir():
		print("Directory found.")
		return True
	elif _path.is_file():
		print("ERROR: The user_path links to a file")
		return False
	else:
		print("Directory not found. It will be created.")
		os.makedirs(_path)
		return True


def is_valid_range(user_range):
	mini = 1
	maxi = 5000
	
	try:
		user_range = [float(val) for val in user_range]
	except (RuntimeError, TypeError):
		print("ERROR: Inputs could not be converted to integers.")
		return False
	
	if user_range[0] >= user_range[1]:
		print("ERROR: Lower bound greater than higher bound.")
		return False
	elif not (mini < user_range[0] < maxi) or not (mini < user_range[1] < maxi):
		print("ERROR: inputs are too high or too low. (Expected 1 < inputs < 10,000)")
		return False
	elif user_range[0] % 1 or user_range[1] % 1:
		print("ERROR: Non-integer inputs.")
	else:
		print("Range is valid.")
		return True


def is_valid_step_nbr(user_step_nbr, valid_bounds):
	try:
		user_step_nbr = float(user_step_nbr)
	except(RuntimeError, TypeError):
		print("ERROR: Input must be an integer greater than 1.")
		return False
	
	if user_step_nbr % 1 or user_step_nbr <= 1:
		print("ERROR: Input must be an integer greater than 1.")
		return False
	if (valid_bounds[1] - valid_bounds[0]) < user_step_nbr:
		print("ERROR: Step size too big.")
		return False
	else:
		return True


def ask_for_path():
	path = input("Enter a directory to use.\nPath = ")
	while not is_valid_path(path):
		print("Please try again.")
		path = input("Enter a directory to use:\nPath = ")
	path = Path(path).resolve()
	return path


def ask_for_interval():
	bounds = [None, None]
	text_prompt = "Enter the range of exposition values to test (µs, positive integers only).\nFrom: "
	bounds[0] = input(text_prompt)
	bounds[1] = input("To: ")
	while not is_valid_range(bounds):
		print("Please try again")
		bounds[0] = input(text_prompt)
		bounds[1] = input("To: ")
	bounds = [int(bound) for bound in bounds]
	return bounds


def ask_for_step_nbr(valid_bounds):
	steps = input("Enter a step number (integer > 1):\n")
	while not is_valid_step_nbr(steps, valid_bounds):
		print("Please try again.")
		steps = input("Please enter a step number (integer > 1):\n")
	steps = int(steps)
	return steps


class Keyboard(object):
	def __init__(self, layout='QWERTY'):
		self.layout = layout
		self.HOME = ord('H')
		self.SET_HOME = ord('h')
		
		self.FORWARD = ord('w')
		self.BACK = ord('s')
		self.LEFT = ord('a')
		self.RIGHT = ord('d')
		self.UP = ord('q')
		self.DOWN = ord('e')
		
		self.X_FORWARD = ord('W')
		self.X_BACK = ord('S')
		self.X_LEFT = ord('A')
		self.X_RIGHT = ord('D')
		self.X_UP = ord('Q')
		self.X_DOWN = ord('E')
		
		self.EXPOSURE_UP = ord('t')
		self.EXPOSURE_DOWN = ord('g')
		
		self.SCAN_FL = ord('j')
		self.SCAN_BR = ord('i')
		self.SET_Z_COR = ord('u')
		
		self.MOVE_SCAN_FL = ord('J')
		self.MOVE_SCAN_BL = ord('U')
		self.MOVE_SCAN_BR = ord('I')
		self.MOVE_SCAN_FR = ord('K')
		self.SCAN = ord('p')
		
		self.HELP1 = ord('?')
		self.HELP2 = ord('/')
		
		self.PREV_SCAN = ord('z')
		self.NEXT_SCAN = ord('x')
		self.ADD_ZONE = ord('v')
		self.DEL_ZONE = ord('B')
		self.DEL_ALL_ZONES = ord('N')
		
		self.TAKE_STACK1 = ord('\n')
		self.TAKE_STACK2 = ord('\r')
		
		self.SAVE_TO_CFG1 = ord('5')  # NOT IMPLEMENTED YET
		self.SAVE_TO_CFG2 = ord('6')  # NOT IMPLEMENTED YET
		self.SAVE_TO_CFG3 = ord('7')  # NOT IMPLEMENTED YET
		
		self.LOAD_CFG1 = ord('8')  # NOT IMPLEMENTED YET
		self.LOAD_CFG2 = ord('9')  # NOT IMPLEMENTED YET
		self.LOAD_CFG3 = ord('0')  # NOT IMPLEMENTED YET
		
		if self.layout == 'AZERTY':
			self.FORWARD = ord('z')
			self.BACK = ord('s')
			self.LEFT = ord('q')
			self.RIGHT = ord('d')
			self.UP = ord('a')
			self.DOWN = ord('e')
			
			self.X_FORWARD = ord('Z')
			self.X_BACK = ord('S')
			self.X_LEFT = ord('Q')
			self.X_RIGHT = ord('D')
			self.X_UP = ord('A')
			self.X_DOWN = ord('E')
			self.PREV_SCAN = ord('w')
