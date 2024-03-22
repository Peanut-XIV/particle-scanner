"""
Focus stacking with laplacian pyramids

Taken from https://github.com/sjawhar/focus-stacking

Use https://github.com/PetteriAimonen/focus-stack instead, which is much faster as it is written in C++ with OpenCL
"""
import argparse
import os
import numpy as np
from scipy import ndimage
import cv2
import skimage.transform as skt
import skimage.util as sku
from glob import glob
import skimage.io as skio
import multiprocessing


def generating_kernel(a):
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(layer, kernel=generating_kernel(0.4)):
    if len(layer.shape) == 2:
        convolution = convolve(layer, kernel)
        return convolution[::2, ::2]

    ch_layer = reduce_layer(layer[:, :, 0])
    next_layer = np.zeros(list(ch_layer.shape) + [layer.shape[2]], dtype=ch_layer.dtype)
    next_layer[:, :, 0] = ch_layer

    for channel in range(1, layer.shape[2]):
        next_layer[:, :, channel] = reduce_layer(layer[:, :, channel])

    return next_layer


def expand_layer(layer, kernel=generating_kernel(0.4)):
    if len(layer.shape) == 2:
        expand = np.zeros((2 * layer.shape[0], 2 * layer.shape[1]), dtype=np.float64)
        expand[::2, ::2] = layer
        convolution = convolve(expand, kernel)
        return 4. * convolution

    ch_layer = expand_layer(layer[:, :, 0])
    next_layer = np.zeros(list(ch_layer.shape) + [layer.shape[2]], dtype=ch_layer.dtype)
    next_layer[:, :, 0] = ch_layer

    for channel in range(1, layer.shape[2]):
        next_layer[:, :, channel] = expand_layer(layer[:, :, channel])

    return next_layer


def convolve(image, kernel=generating_kernel(0.4)):
    return ndimage.convolve(image.astype(np.float64), kernel, mode='mirror')


def gaussian_pyramid(images, levels):
    pyramid = [images.astype(np.float64)]
    num_images = images.shape[0]

    while levels > 0:
        next_layer = reduce_layer(pyramid[-1][0])
        next_layer_size = [num_images] + list(next_layer.shape)
        pyramid.append(np.zeros(next_layer_size, dtype=next_layer.dtype))
        pyramid[-1][0] = next_layer
        for layer in range(1, images.shape[0]):
            pyramid[-1][layer] = reduce_layer(pyramid[-2][layer])
        levels = levels - 1

    return pyramid


def laplacian_pyramid(images, levels):
    gaussian = gaussian_pyramid(images, levels)

    pyramid = [gaussian[-1]]
    for level in range(len(gaussian) - 1, 0, -1):
        gauss = gaussian[level - 1]
        pyramid.append(np.zeros(gauss.shape, dtype=gauss.dtype))
        for layer in range(images.shape[0]):
            gauss_layer = gauss[layer]
            expanded = expand_layer(gaussian[level][layer])
            if expanded.shape != gauss_layer.shape:
                expanded = expanded[:gauss_layer.shape[0], :gauss_layer.shape[1]]
            pyramid[-1][layer] = gauss_layer - expanded

    return pyramid[::-1]


def collapse(pyramid):
    image = pyramid[-1]
    for layer in pyramid[-2::-1]:
        expanded = expand_layer(image)
        if expanded.shape != layer.shape:
            expanded = expanded[:layer.shape[0], :layer.shape[1]]
        image = expanded + layer
    return image


def get_probabilities(gray_image):
    levels, counts = np.unique(gray_image.astype(np.uint8), return_counts=True)
    probabilities = np.zeros((256,), dtype=np.float64)
    probabilities[levels] = counts.astype(np.float64) / counts.sum()
    return probabilities


def entropy(image, kernel_size):
    def _area_entropy(area, probabilities):
        levels = area.flatten()
        return -1. * (levels * np.log(probabilities[levels])).sum()
    probabilities = get_probabilities(image)
    pad_amount = int((kernel_size - 1) / 2)
    padded_image = cv2.copyMakeBorder(image, pad_amount, pad_amount, pad_amount, pad_amount, cv2.BORDER_REFLECT101)
    entropies = np.zeros(image.shape[:2], dtype=np.float64)
    offset = np.arange(-pad_amount, pad_amount + 1)
    for row in range(entropies.shape[0]):
        for column in range(entropies.shape[1]):
            area = padded_image[row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset]
            entropies[row, column] = _area_entropy(area, probabilities)
    return entropies


def deviation(image, kernel_size):
    def _area_deviation(area):
        average = np.average(area).astype(np.float64)
        return np.square(area - average).sum() / area.size
    pad_amount = int((kernel_size - 1) / 2)
    padded_image = cv2.copyMakeBorder(image, pad_amount, pad_amount, pad_amount, pad_amount, cv2.BORDER_REFLECT101)
    deviations = np.zeros(image.shape[:2], dtype=np.float64)
    offset = np.arange(-pad_amount, pad_amount + 1)
    for row in range(deviations.shape[0]):
        for column in range(deviations.shape[1]):
            area = padded_image[row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset]
            deviations[row, column] = _area_deviation(area)
    return deviations


def get_fused_base(images, kernel_size):
    layers = images.shape[0]
    entropies = np.zeros(images.shape[:3], dtype=np.float64)
    deviations = np.copy(entropies)
    for layer in range(layers):
        gray_image = cv2.cvtColor(images[layer].astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.uint8)
        probabilities = get_probabilities(gray_image)
        entropies[layer] = entropy(gray_image, kernel_size)
        deviations[layer] = deviation(gray_image, kernel_size)
    best_e = np.argmax(entropies, axis=0)
    best_d = np.argmax(deviations, axis=0)
    fused = np.zeros(images.shape[1:], dtype=np.float64)
    # print("get_fused_base")
    # plt.matshow(best_e), plt.show()
    # plt.matshow(best_d), plt.show()
    for layer in range(layers):
        fused += np.where(best_e[:, :, np.newaxis] == layer, images[layer], 0)
        fused += np.where(best_d[:, :, np.newaxis] == layer, images[layer], 0)
    return (fused / 2).astype(images.dtype)


def fuse_pyramids(pyramids, kernel_size):
    fused = [get_fused_base(pyramids[-1], kernel_size)]
    energies = []
    for layer in range(len(pyramids) - 2, -1, -1):
        f, en = get_fused_laplacian(pyramids[layer])
        fused.append(f)
        energies.append(en)
    en_best = np.argmax(energies[-2], axis=0)
    en_best = ndimage.median_filter(en_best, [15,15]).astype(np.uint8)
    m = sku.img_as_ubyte(skt.resize(en_best, energies[-1].shape[1:3]))
    return fused[::-1], m


def get_fused_laplacian(laplacians):
    layers = laplacians.shape[0]
    region_energies = np.zeros(laplacians.shape[:3], dtype=np.float64)
    for layer in range(layers):
        gray_lap = cv2.cvtColor(laplacians[layer].astype(np.float32), cv2.COLOR_BGR2GRAY)
        region_energies[layer] = region_energy(gray_lap)
    best_re = np.argmax(region_energies, axis=0)
    # print("get_fused_laplacian")
    # plt.matshow(best_re), plt.colorbar(), plt.show()
    # plt.matshow(np.log(np.max(region_energies, axis=0))), plt.colorbar(), plt.show()
    fused = np.zeros(laplacians.shape[1:], dtype=laplacians.dtype)
    for layer in range(layers):
        fused += np.where(best_re[:, :, np.newaxis] == layer, laplacians[layer], 0)
    return fused, region_energies


def region_energy(laplacian):
    return convolve(np.square(laplacian))


def get_pyramid_fusion(images, min_size=32):
    smallest_side = min(images[0].shape[:2])
    depth = int(np.log2(smallest_side / min_size))
    kernel_size = 5
    pyramids = laplacian_pyramid(images, depth)
    fusion, depth = fuse_pyramids(pyramids, kernel_size)
    return collapse(fusion), depth


def phase_correction(images):
    offsets = []
    for i in range(len(images)-1):
        im1 = images[i][..., 1].astype(np.float32) / 255# Green
        im2 = images[i+1][..., 1].astype(np.float32) / 255 # Green
        f1 = np.fft.fft2(im1)
        f2 = np.fft.fft2(im2)
        f = f1 * np.conj(f2)
        imp = np.real(np.fft.ifft2(f))
        minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(imp)
        maxLoc = np.asarray(maxLoc)
        if maxLoc[0] > im1.shape[1] / 2:
            maxLoc[0] -= im1.shape[1]
        if maxLoc[1] > im1.shape[0] / 2:
            maxLoc[1] -= im1.shape[0]
        offsets.append(maxLoc)

    cox = 0
    coy = 0
    fixed_images = []
    offsets = np.asarray(offsets)
    cum_offsets = np.cumsum(offsets, axis=0)
    cmin = np.min(cum_offsets, axis=0)
    cmin[cmin > 0] = 0
    cmax = np.max(cum_offsets, axis=0)
    cmax[cmax < 0] = 0
    for i, offset in enumerate(offsets):
        cox += offset[0]
        coy += offset[1]
        M = np.asarray([[1, 0, cox], [0, 1, coy]], dtype=np.float32)
        im = cv2.warpAffine(images[i+1].astype(np.float32), M, dsize=images[i+1].shape[:2][::-1])
        fixed_images.append(im)
        # plt.matshow(images[i + 1] / 255)
        # plt.show()
        # plt.matshow(im / 255)
        # plt.show()
    fixed_images = np.asarray(fixed_images)

    return fixed_images[:,
           cmax[1]:fixed_images.shape[1] + cmin[1],
           cmax[0]:fixed_images.shape[2] + cmin[0],
           :]


def stack_process(process_name, tasks):
    print(f"{process_name}: stacking process starts")

    while True:
        d = tasks.get()
        print(d)
        if d is None:
            print(f"{process_name}: stacking process ends")
            break
        else:
            print(f"{process_name}: processing {d}")
            fns = sorted(glob(os.path.join(d, "*.jpg")))
            if len(fns) == 0:
                continue
            print(f"{process_name}: loading images {d}")
            images = []
            for fn in fns:
                im = skio.imread(fn, False)
                images.append(im)
            images = np.asarray(images)
            print(f"{process_name}: phase correction {d}")
            fixed_images = phase_correction(images)
            print(f"{process_name}: fusing {d}")
            fused_image, depth = get_pyramid_fusion(np.asarray(fixed_images))
            fused_image = (fused_image / fused_image.max() * 255).astype(np.uint8)

            save_dir = os.path.join(d, "fused_laplacian_pyramids")
            os.makedirs(save_dir, exist_ok=True)
            skio.imsave(os.path.join(save_dir, os.path.basename(d) + "_image.png"), fused_image, check_contrast=False)
            skio.imsave(os.path.join(save_dir, os.path.basename(d) + "_depth.png"), depth, check_contrast=False)
            skio.imsave(os.path.join(save_dir, os.path.basename(d) + "_depthx10.png"), depth * 10, check_contrast=False)
    return


def stack(dir):
    manager = multiprocessing.Manager()
    tasks = manager.Queue()

    num_processes = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []

    for i in range(num_processes):
        process_name = 'P%i' % i
        new_process = multiprocessing.Process(target=stack_process, args=(process_name, tasks))
        processes.append(new_process)
        new_process.start()

    dirs = sorted([d for d in glob(os.path.join(dir, "*")) if os.path.isdir(d)])
    for d in dirs:
        tasks.put(d)

    for job in processes:
        job.join()

