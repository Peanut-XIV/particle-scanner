from sashimi.controller import Controller
from sashimi.utils import make_unique_subdir


if __name__ == "__main__":
    my_dir = make_unique_subdir(r"C:\Users\christine\sashimi_scans")
    controller = Controller(my_dir,
                            "COM4",
                            lang="en",
                            layout='AZERTY',
                            auto_f_stack=True,
                            lowest_z=True,
                            remove_raw=True,
                            horizontal_stack_offset=(1600, 1260),
                            # 500 µm of overlap in both direction
                            )
    controller.start()
