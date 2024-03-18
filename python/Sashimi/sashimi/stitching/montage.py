import itk
import os

from itk.itkImagePython import itkImageUC3, itkImageUC2
from tqdm import tqdm

# Directory containing the images
path = "/Users/louis/Desktop/dossier2"
scale = 2

# List all image files in the directory and sort them according to your custom logic
# This example assumes image filenames contain row and column indices
image_files = [f for f in os.listdir(path) if f.endswith('.png')]
# sort the image files increase by column and then by row and store all in a list
image_files.sort(key=lambda f: (-int(f.split('_')[-2][1:]), int(f.split('_')[-1][1:-4])))

print(image_files)

coords = []
with open("TileConfiguration.txt", "w") as f:
    f.write(
        "# Tile configuration according to https://github.com/InsightSoftwareConsortium/ITKMontage/blob/master/examples/SampleData_CMUrun2/TileConfiguration.registered.txt\n")
    f.write("dim = 2\n\n")
    for image_file in image_files:
        parts = image_file.split('_')
        col = int(parts[-2][1:])
        row = int(parts[-1][1:-4])
        coords.append((row, col))
        f.write(f"{image_file};;({row}, {-col})\n")

# Get number of unique rows and columns
num_rows = len(set([c[0] for c in coords]))
num_cols = len(set([c[1] for c in coords]))

import sys
import os
import itk
from pathlib import Path


input_path = Path(path)
output_path = input_path / ".." / "mosaic"
output_path.mkdir(exist_ok=True)
out_file = output_path / "mosaic.tif"

if not out_file.is_absolute():
    out_file = (output_path / out_file).resolve()

dimension = 2

stage_tiles = itk.TileConfiguration[dimension]()
stage_tiles.Parse("TileConfiguration.txt")



color_images = []  # for mosaic creation
grayscale_images = []  # for registration


for i, t in enumerate(range(stage_tiles.LinearSize())):
    origin = stage_tiles.GetTile(t).GetPosition()
    filename = str(input_path / stage_tiles.GetTile(t).GetFileName())
    image = itk.imread(filename)
    spacing = image.GetSpacing()


    # tile configurations are in pixel (index) coordinates
    # so we convert them into physical ones
    # for d in range(dimension):
    #     origin[d] *= spacing[d]

    image.SetOrigin(origin)
    image.SetSpacing((3.45 / scale, 3.45 / scale))
    color_images.append(image)

    image = itk.imread(filename, itk.F)
    image.SetSpacing((3.45 / scale, 3.45 / scale))# read as grayscale
    image.SetOrigin(origin)
    grayscale_images.append(image)

# only float is wrapped as coordinate representation type in TileMontage
montage = itk.TileMontage[type(grayscale_images[0]), itk.F].New()
montage.SetMontageSize(stage_tiles.GetAxisSizes())

# Set registration parameters
# montage.SetRegistrationMethod(itk.RegistrationMethodType.FeatureBased)
# montage.SetFeatureDetectorType(itk.FeatureDetectorType.SIFT)
# montage.SetOptimizerType(itk.OptimizerType.GradientDescent)
# montage.SetMetricType(itk.MetricType.MutualInformation)
#
# # Set transformation model
# montage.SetTransformationType(itk.TransformationType.Affine)
#
# # Set interpolator for resampling
# montage.SetInterpolator(itk.InterpolatorType.Linear)


for t in range(stage_tiles.LinearSize()):
    montage.SetInputTile(t, grayscale_images[t])

print("Computing tile registration transforms")
montage.Update()

print("Writing tile transforms")
actual_tiles = stage_tiles  # we will update it later
for t in range(stage_tiles.LinearSize()):
    index = stage_tiles.LinearIndexToNDIndex(t)
    regTr = montage.GetOutputTransform(index)
    tile = stage_tiles.GetTile(t)
    itk.transformwrite([regTr], str(output_path / (tile.GetFileName() + ".tfm")))

    # calculate updated positions - transform physical into index shift
    pos = tile.GetPosition()
    for d in range(dimension):
        pos[d] -= regTr.GetOffset()[d] / spacing[d]
    tile.SetPosition(pos)
    actual_tiles.SetTile(t, tile)
actual_tiles.Write(str(output_path / "TileConfiguration.registered.txt"))

print("Producing the mosaic")
input_pixel_type = itk.template(color_images[0])[1][0]
try:
    input_rgb_type = itk.template(input_pixel_type)[0]
    accum_type = input_rgb_type[itk.F]  # RGB or RGBA input/output images
except KeyError:
    accum_type = itk.D  # scalar input / output images

resampleF = itk.TileMergeImageFilter[type(color_images[0]), accum_type].New()
resampleF.SetMontageSize(stage_tiles.GetAxisSizes())
for t in range(stage_tiles.LinearSize()):
    resampleF.SetInputTile(t, color_images[t])
    index = stage_tiles.LinearIndexToNDIndex(t)
    resampleF.SetTileTransform(index, montage.GetOutputTransform(index))
resampleF.Update()
itk.imwrite(resampleF.GetOutput(), str(out_file))
print("Resampling complete")
