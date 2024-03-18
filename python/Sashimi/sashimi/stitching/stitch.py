from pathlib import Path

import cv2

# from stitch2d import StructuredMosaic
#
# mosaic = StructuredMosaic(
#     r"F:\Sashimi\test\zone_00000",
#     dim=3,                  # number of tiles in primary axis
#     origin="lower left",     # position of first tile
#     direction="vertical",  # primary axis (i.e., the direction to traverse first)
#     pattern="raster"          # snake or raster
#   )
#
#
# # mosaic = create_mosaic(r"F:\Sashimi\test\zone_00000")
#
# mosaic.align()
# mosaic.reset_tiles()
# mosaic.save_params()
#
# mosaic.smooth_seams()
# mosaic.save(r"F:\Sashimi\test\zone_00000\mosaic\mosaic.jpg")

path = Path(r"F:\Sashimi\test\ZONE00000\E02000")

images = [cv2.imread(str(im)) for im in path.glob("*.png")]

sitcher = cv2.Stitcher_create()

status, stitched_image = sitcher.stitch(images)

if status == 0:
    mosaic_path = path / "mosaic" / "mosaic.jpg"
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    # Save or display the stitched image
    cv2.imwrite(str(mosaic_path), stitched_image)
else:
    print("Image stitching failed:", status)
