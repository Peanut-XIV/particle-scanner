from pathlib import Path
from sashimi.gui.main import main

if __name__ == "__main__":
    dummy_cam = bool(input("dummy_cam ? [y/n] ") in ["y","Y","yes","Yes","YES"])
    dummy_stage = bool(input("dummy_stage ? [y/n] ") in ["y","Y","yes","Yes","YES"])
    image_path = input("camera image : ")
    if not (Path(image_path).exists() and Path(image_path).is_file()):
        print("Invalid image")
    else:
        main(
            dummy_camera=dummy_cam,
            dummy_stage=dummy_stage,
            test_screen=image_path,
            stack_method="helicon"
        )
