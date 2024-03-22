from pathlib import Path
from sashimi.gui.main_window import main

if __name__ == "__main__":
    dummy_cam = bool(input("dummy_cam ? [y/n] ") in ["y","Y","yes","Yes","YES"])
    dummy_stage = bool(input("dummy_stage ? [y/n] ") in ["y","Y","yes","Yes","YES"])
    main(
        dummy_camera=dummy_cam,
        dummy_stage=dummy_stage,
        stack_method="helicon"
    )
