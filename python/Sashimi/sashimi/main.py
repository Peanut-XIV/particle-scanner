from sashimi.controller import Controller
from sashimi.utils import make_unique_subdir


if __name__ == "__main__":
    my_dir = make_unique_subdir("C:\\Users\\utilisateur.SF_GEOL_STAG_IN\\Desktop\\Data")
    controller = Controller(my_dir, "COM5", lang="en", layout='AZERTY', auto_f_stack=True, lowest_z=True)
    controller.start()
