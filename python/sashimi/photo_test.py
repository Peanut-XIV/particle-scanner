import os

from sashimi.controller import Controller
from pathlib import Path


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
    elif not (mini<user_range[0]<maxi) or not (mini<user_range[1]<maxi):
        print("ERROR: inputs are too high or too low. (Expected 1 < inputs < 10,000)")
        return False
    elif user_range[0]%1 or user_range[1]%1:
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


if __name__ == "__main__":
    user_path = ask_for_path()
    interval = ask_for_interval()
    step_nbr = ask_for_step_nbr(interval)

    step_size = (interval[1] - interval[0]) // (step_nbr - 1)
    exposition_values = [interval[0] + i * step_size for i in range(step_nbr)]
    print("Input collection finished, the scanning program will start.")

    controller = Controller(user_path, 'COM5', photo_test=exposition_values, lang='fr')
    # On pressing [p], the program will scan repeatedly with different settings of camera exposition.
    # function returns automatically after scanning
    interrupt_flag = controller.start()

    if interrupt_flag:
        quit("program interrupted by the user")
