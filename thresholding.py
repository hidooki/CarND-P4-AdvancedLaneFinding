"""Gradient and color thresholding"""
import cv2
import numpy as np
from matplotlib import image as mpimage
from matplotlib import pyplot as plt

import utils


# Gradient thresholding

def abs_sobel_thresh(rgb_image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Calculate directional gradient. Apply threshold"""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    d = 1 if orient == 'x' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, d, 1 - d, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    mask = (scaled_sobel >= thresh[0]) * (scaled_sobel <= thresh[1])
    return mask.astype(np.int)


def mag_thresh(rgb_image, sobel_kernel=3, mag_thresh=(0, 255)):
    """Calculate gradient magnitude. Apply trehold"""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.sqrt(gradx**2 + grady ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    mask = (scaled_sobel >= mag_thresh[0]) * (scaled_sobel <= mag_thresh[1])
    return mask.astype(np.int)


def dir_threshold(rgb_image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """Calculate gradient direction. Apply threshold"""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.abs(gradx)
    abs_sobely = np.abs(grady)

    direction = np.arctan2(abs_sobely, abs_sobelx)
    mask = (direction >= thresh[0]) * (direction <= thresh[1])
    return mask.astype(int)


# Color Thresholding

def hls_threshold(
        rbg_image,
        h_thresh=(0, 180),
        l_thresh=(0, 255),
        s_thresh=(0, 255)):

    hls = cv2.cvtColor(rbg_image, cv2.COLOR_RGB2HLS)

    h_mask = (hls[:, :, 0] >= h_thresh[0]) * (hls[:, :, 0] <= h_thresh[1])
    l_mask = (hls[:, :, 1] >= l_thresh[0]) * (hls[:, :, 1] <= l_thresh[1])
    s_mask = (hls[:, :, 2] >= s_thresh[0]) * (hls[:, :, 2] <= s_thresh[1])
    return (h_mask * l_mask * s_mask).astype(np.int)


# Testing utility

def test_color_gradient(
        rgb_image,
        kernel=3,
        grad_thresh=(20, 100),
        h_thresh=(0, 255),
        l_thresh=(0, 255),
        s_thresh=(170, 255),
        plot_all=True):

    c_binary = hls_threshold(rgb_image, h_thresh, l_thresh, s_thresh)

    g_binary = abs_sobel_thresh(
        rgb_image,
        orient='x',
        sobel_kernel=kernel,
        thresh=grad_thresh)

    combo = np.logical_or(c_binary, g_binary)

    plt.interactive(True)

    if plot_all:
        fig = utils.plot_batch(
            [rgb_image, c_binary, g_binary, combo],
            ['Original', 'Color Threshold', 'Gradient Threshold', 'Combo'],
            figsize=(16, 6))

    else:
        fig = plt.figure()
        plt.imshow(combo, cmap='gray')

    fig.tight_layout()
    return fig


if __name__ == '__main__':

    # Example of color and gradient thresholding

    test_img = 'color-shadow-example.jpg'
    image = mpimage.imread('test_images/%s' % test_img)

    figure = test_color_gradient(
        image,
        kernel=3,
        grad_thresh=(40, 100),
        h_thresh=(0, 90),
        s_thresh=(170, 255),
        plot_all=False)

    plt.savefig('output_images/threshold_%s' % test_img)
