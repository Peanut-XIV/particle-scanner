import numpy as np
from numpy import ndarray
from numpy.linalg import inv
from pathlib import Path
from math import ceil
from skimage.measure import ransac
from skimage.transform import warp, AffineTransform
from skimage.feature import corner_peaks, corner_harris
import cv2.imread as imread
import cv2.imwrite as imwrite

# In order to combine a set of neighboring images into a larger one, each image needs to be deformed a bit to fit
# exactly with each other. Then, they are stitched together by doing an average where they overlap.
# /!\ Caution /!\ The vertical axis comes first in an image converted to a numpy ndarray.
# However, it's the x coordinate first in a homogeneous transformation matrix!


class ImageTransform:
	"""
	This Class links an image (as a path or ndarray) to its transformation matrix, two neighboring images from which
	the transformation matrix is estimated and the image's corners.Corners are easily identifiable points of an image.
	They are used to find out how two images should superpose.
	"""
	def __init__(self,
	             image: Path | str | ndarray,
	             left_ref: object,  # type should be ImageTransform but it cannot be used recursively
	             top_ref: object,   # Same
	             matrix=None,
	             corners=None):
		if image is Path or image is str:
			self.path = image
		else:
			self.path = None
			self._image = image
		self.left = left_ref
		self.top = top_ref
		if matrix is None:
			self.matrix = np.identity(3)
		else:
			self.matrix = matrix
		if corners is None:
			image_for_corners = imread(self.path) if self.path is not None else self._image
			self.corners = corner_peaks(corner_harris(image_for_corners), min_distance=5)
		else:
			self.corners = corners
	
	@property
	def image(self) -> ndarray:
		# Reduces RAM usage
		# if path is not None, the image is loaded on demand instead of loaded from initiation until deletion
		if self.path is None:
			return self._image
		else:
			return imread(self.path)


def image_dir_2_grid(path) -> ndarray:
	path = Path(path)
	images = path.iterdir()
	x_values = []
	y_values = []
	for img in images:
		x, _, y = img.stem.partition("_")
		x = x.lstrip()
		y = y.lstrip()
		if x not in x_values:
			x_values.append(x)
		if y not in y_values:
			y_values.append(y)
	grid = ndarray([len(x_values), len(y_values)])
	x_values.sort()  # number of x values == number of columns == width
	y_values.sort()  # number of y values == number of rows    == height
	for i, x in enumerate(y_values):
		for j, y in enumerate(x_values):
			grid[i, j] = path.joinpath(f"X{x}_Y{y}.jpg")
	return grid


def make_image_list(grid: ndarray) -> [ImageTransform]:
	# flattens the ndarray by splitting it in vertical slices
	height, width = grid.shape
	output = []
	for j in range(width):
		for i in range(height):
			path = grid[i, j]
			top = None if i == 0 else output[-1]
			left = None if j == 0 else output[- height]
			output.append(ImageTransform(path, left, top))
	return output


def compute_transforms(_image_list: [ImageTransform]) -> None:
	for img in _image_list[1:]:
		if img.left is not None and img.top is not None:
			ref = merge_references(img.left, img.top)
		elif img.left is None:
			ref = img.top
		else:
			ref = img.left
		matched_corners = match_locations(ref.image, img.image, ref.corners, img.corners)
		estimated_transform = ransac(
			matched_corners,
			AffineTransform,
			min_samples=3,
			residual_threshold=2,
			max_trials=100
		)
		img.matrix = np.matmul(ref.matrix, np.asmatrix(estimated_transform[0].params))
	

def merge_references(image1: ImageTransform, image2: ImageTransform) -> ImageTransform:
	# transform image2 to the reference frame of untransformed image1
	matrix = np.matmul(inv(image1.matrix), image2.matrix)
	# find the bounds of possible x and y values to find the correct offset and image size
	img1 = image1.image
	img2 = image2.image
	y1, x1, _ = img1.shape
	y2, x2, _ = img2.shape
	angles = cartesian([0, x1], [0, y1]) + repeat_transform(matrix, cartesian([0, x2], [0, y2]))
	(x_min, y_min), (x_max, y_max) = min_max_ab(angles)
	delta = np.asmatrix([[1, 0, - x_min], [0, 1, - y_min], [0, 0, 1]])
	im_shape = (ceil(y_max - y_min), ceil(x_max - x_min))
	# needs the inverse matrix of the transformation to apply
	img1_transformed = warp(img1, inv(delta), output_shape=im_shape, cval=np.NaN)
	img2_transformed = warp(img2, inv(np.matmul(delta, matrix)), output_shape=im_shape, cval=np.NaN)
	# the reference image
	output_image = np.nanmean(np.stack((img1_transformed, img2_transformed)))
	# the transform to apply to have it look like it's in the final micro panorama
	output_matrix = np.matmul(image1.matrix, inv(delta))
	return ImageTransform(output_image, None, None, output_matrix)


def repeat_transform(matrix, points_list) -> [(float, float)]:
	output = []
	for point in points_list:
		new_point = np.matmul(matrix, np.array((point[0], point[1], 1)))
		output.append((new_point[0], new_point[1]))
	return output


def cartesian(set1: [any], set2: [any]) -> [(any, any)]:
	output = []
	for e1 in set1:
		for e2 in set2:
			output.append((e1, e2))
	return output


def min_max_ab(points_list: [(any, any)]) -> ((any, any), (any, any)):
	x_min = min(points_list, key=lambda v: v[0])[0]
	x_max = max(points_list, key=lambda v: v[0])[0]
	y_min = min(points_list, key=lambda v: v[1])[1]
	y_max = max(points_list, key=lambda v: v[1])[1]
	return (x_min, y_min), (x_max, y_max)


def match_locations(img0, img1, coords0, coords1, radius=5, sigma=3) -> ndarray:
	y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
	weights = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
	weights /= 2 * np.pi * sigma * sigma
	match_list = []
	for r0, c0 in coords0:
		roi0 = img0[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1]
		roi1_list = [img1[r1 - radius:r1 + radius + 1, c1 - radius:c1 + radius + 1] for r1, c1 in coords1]
		# sum of squared differences
		ssd_list = [np.sum(weights * (roi0 - roi1) ** 2) for roi1 in roi1_list]
		match_list.append(coords1[np.argmin(ssd_list)])
	return np.array(match_list)


def compose_panorama(_image_list: [ImageTransform]) -> ndarray:
	# compute the dimensions of the panorama
	# for each image, get coords of corners after transform
	corners = []
	for img in _image_list:
		image = img.image
		y, x, _ = image.shape
		del image
		# TODO: getting image dimensions by reading the header of the file
		#  would be faster than loading it in memory (don't know how though)
		corners += repeat_transform(img.matrix, cartesian((0, x), (0, y)))
	(x_min, y_min), (x_max, y_max) = min_max_ab(corners)
	off_mat = np.identity(3)
	off_mat[:2, 2] = -x_min, -y_min
	dimensions = (y_max - y_min, x_max - x_min)
	
	# create an accumulator image
	img0 = _image_list[0]
	panorama = warp(img0, inv(np.matmul(off_mat, img0.matrix)), output_shape=dimensions, cval=np.NaN)
	for img in _image_list[1:]:
		img_tf = warp(img, inv(np.matmul(off_mat, img.matrix)), output_shape=dimensions, cval=np.NaN)
		np.nanmean(np.stack((img_tf, panorama)), out=panorama)
	return panorama


if __name__ == "__main__":
	dir_path = input("image folder: ")
	image_name = input("image name:")
	image_grid = image_dir_2_grid(dir_path)
	image_list = make_image_list(image_grid)
	compute_transforms(image_list)
	out_image = compose_panorama(image_list)
	imwrite(image_name, out_image)
