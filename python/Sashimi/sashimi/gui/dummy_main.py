from sashimi.gui.main_window import main

if __name__ == "__main__":
    YES = ["y","Y","yes","Yes","YES"]
    dummy_cam = bool(input("dummy_cam ? [y/n] ") in YES)
    dummy_stage = bool(input("dummy_stage ? [y/n] ") in YES)
    if not (dummy_stage or dummy_cam) :
        debug_dws = {
                "skip_stacks": True,
                "detector": "SimpleBlobDetector",
                "verbose": True
                }
    main(
        dummy_camera=dummy_cam,
        dummy_stage=dummy_stage,
        stack_method="helicon",
        debug_dws=debug_dws
    )

