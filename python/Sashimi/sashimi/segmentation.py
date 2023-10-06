"""
Segment directory containing stacked images
"""
import os
from glob import glob

import cv2
import skimage.io as skio
import skimage.color as skc
import numpy as np

import matplotlib.pyplot as plt


def segment(dir, threshold, resolution=1159.42):
    images_dir = [d for d in sorted(glob(os.path.join(dir, "*"))) if os.path.isdir(d)]
    if len(images_dir) == 0:
        raise ValueError("No image directories found")

    for d in images_dir:
        filename = None

        if os.path.exists(os.path.join(d, "Focused")):
            filename = glob(os.path.join(d, "Focused", "*.jpg"))[0]

        bn = os.path.basename(d)
        bn_parts = bn.split("_")
        x = int(bn_parts[0][1:])
        y = int(bn_parts[1][1:])

        print(x)
        print(y)

        if filename is not None:
            im = skio.imread(filename)
            im_grey = skc.rgb2grey(im) * 255
            _, th = cv2.threshold(im_grey, threshold, 255, cv2.THRESH_BINARY)
            th = th.astype(np.uint8)
            elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [5, 5])
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, elem)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 30 * 30:
                    continue
                M = cv2.moments(contour)
                minx = np.min(contour[..., 0])
                miny = np.min(contour[..., 1])
                maxx = np.max(contour[..., 0])
                maxy = np.max(contour[..., 1])
                width = maxx - minx
                height = maxy - miny
                w = int(0.1 * width)
                h = int(0.1 * height)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                print(f"cx: {cx}, cy: {cy}, minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy}")
                sx = np.max((minx - w, 0))
                sy = np.max((miny - h, 0))
                ex = np.min((maxx + w, im.shape[1]))
                ey = np.min((maxy + h, im.shape[0]))
                crop = im[sy:ey, sx:ex, :]

                posx = cx / resolution * 1000 + x
                posy = (im.shape[1] - cy) / resolution * 1000 + y

                plt.imshow(crop)
                plt.title(f"x: {posx}, y: {posy}")
                plt.show()

                segment_dir = os.path.join(dir, "segments")
                os.makedirs(segment_dir, exist_ok=True)
                skio.imsave(os.path.join(segment_dir, f"X{posx:06.0f}_Y{posy:06.0f}.jpg"), crop, quality=90)
            plt.imshow(th)
            plt.show()

if __name__ == "__main__":
    segment(r"D:\Sashimi\Test4", 50)
