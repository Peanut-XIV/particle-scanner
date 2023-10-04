import numpy as np
from pathlib import Path
from skimage.measure import ransac
from skimage.transform import warp
from skimage.feature import corner_peaks, corner_harris
import cv2.imread as imread

class ImageTransform:
	def __init__(self, image_path: Path | str, left_image: object, top_image: object):
		self.path = image_path
		self.left = left_image
		self.top = top_image
		self.matrix = np.identity(3)
		img = imread(self.path)
		self.corners = corner_peaks(corner_harris(img), min_distance=5)
	
	def get_image(self):
		return imread(self.path)


def image_dir_2_grid(path) -> np.ndarray:
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
	grid = np.ndarray([len(x_values), len(y_values)])
	x_values.sort()
	y_values.sort()
	for i, x in enumerate(x_values):
		for j, y in enumerate(y_values):
			grid[i, j] = path.joinpath(f"X{x}_Y{y}.jpg")
	return grid


def make_image_list(grid: np.ndarray) -> [ImageTransform]:
	width, height = grid.shape
	output = []
	for i in range(width):
		for j in range(height):
			path = grid[i, j]
			left = None if i == 0 else output[- height]
			top = None if j == 0 else output[-1]
			output.append(ImageTransform(path, left, top))
	return output


def calculate_transforms(ref_list: [ImageTransform]) -> None:
	for img in ref_list[1:]:
		if img.left is not None and img.top is not None:
			# Deal wih both left and top reference images
			left = img.left.get_image()
			l_corners = img.left.corners
			top = img.top.get_image()
			t_corners = img.top.corners
			
			ref = "hello :)"
		
		# Now, deal with the only ref
		print("deal with only ref")


def merge_left_top(left: ImageTransform, top: ImageTransform) -> np.ndarray:
	# transform top image to the reference frame of untransformed left image
	rm_trans = np.ndarray([[1, 1, 0], [1, 1, 0], [1, 1, 1]])
	kp_trans = np.ndarray([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
	inv_left_warp = np.linalg.inv(left.matrix * rm_trans)
	matrix = np.matmul(top.matrix - left.matrix * kp_trans, inv_left_warp)
	
	# find the bounds of possible x and y values to find the correct offset and image size
	l_img = left.get_image()
	t_img = top.get_image()
	lx, ly, _ = l_img.shape
	tx, ty, _ = t_img.shape
	angles = [(0, 0), (0, ly), (lx, 0), (lx, ly)] + repeat_transform(matrix, [(0, 0), (0, ty), (tx, 0), (tx, ty)])
	x_min = min(angles, key=lambda v: v[0])[0]
	x_max = max(angles, key=lambda v: v[0])[0]
	y_min = min(angles, key=lambda v: v[1])[1]
	y_max = max(angles, key=lambda v: v[1])[1]
	offset = (-x_min, -y_min)
	im_shape = (round(x_max - x_min), round(y_max - y_min), 3)
	
	# blend output with offset left img
	output1 = warp(l_img, np.linalg.inv())
	# blend output with transformed then offset top img
	return output

	
def repeat_transform(matrix, points_list) -> [(float, float)]:
	output = []
	for point in points_list:
		transform = np.matmul(matrix, np.array((point[0], point[1], 0)))
		output.append((transform[0], transform[1]))
	return output
	

def match_locations(img0, img1, coords0, coords1, radius=5, sigma=3):
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
	
	
	
