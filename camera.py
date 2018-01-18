import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import pickle

import utils


# Camera calibration

def calibrate_camera(image_names, nx, ny):
    """Calibrate camera distortion given test images

    Input
        images - images of chessboard, to calibrate to
        nx - number of inner corners on each row
        ny - number of inner conrers on each column

    Output
        ret, mtx, dist, rvecs, tvecs - calibration params
    """

    height, length, n_channels = cv2.imread(image_names[0]).shape

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, n_channels), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in image_names:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

        cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    img_size = (length, height)
    params = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # ret, mtx, dist, rvecs, tvecs = params

    return params


def undistort(image, params):
    ret, mtx, dist, rvecs, tvecs = params
    return cv2.undistort(image, mtx, dist, None, mtx)


class Camera():
    def __init__(self,
                 calib_images='camera_cal/calibration*.jpg',
                 params_file='camera_cal/params.p',
                 n_corners=(9, 6)
                 ):
        self.calib_images = calib_images
        self.params_file = params_file
        self.n_corners = n_corners
        self.params = None

    def calibrate(self):

        if os.path.isfile(self.params_file):
            print('Loading camera params from cache..')
            self.params = pickle.load(open(self.params_file, 'rb'))
        else:
            print('Running calibration..')
            image_names = glob.glob(self.calib_images)
            self.params = calibrate_camera(image_names, *self.n_corners)
            pickle.dump(self.params, open(self.params_file, 'wb'))

    def undistort(self, image):
        return undistort(image, self.params)


# Perspective transform

SRC = np.float32([[160, 720], [585, 450], [695, 450], [1120, 720]])
DST = np.float32([[300, 720], [300, 0], [980, 0], [980, 720]])


def warp_perspective(undist_image, src=SRC, dst=DST):
    """Warps an image as prescribed by the src -> dst mapping
    Input image should be undistorted"""
    M = cv2.getPerspectiveTransform(src, dst)
    img_sz = undist_image.shape[1:: -1]
    return cv2.warpPerspective(undist_image, M, img_sz, flags=cv2.INTER_LINEAR)


def plot_warp(image, src=SRC, dst=DST, color=[255, 0, 0], thickness=3):
    img = np.copy(image)
    for i in range(3):
        cv2.line(img, tuple(src[i]), tuple(src[i + 1]),
                 color=color, thickness=thickness)

    warped = warp_perspective(image, src, dst)
    for i in range(3):
        cv2.line(warped, tuple(dst[i]), tuple(dst[i + 1]),
                 color=color, thickness=thickness)

    return utils.plot_batch([img, warped], ['Original', 'Warped'], (16, 6))


if __name__ == '__main__':

    my_camera = Camera()
    my_camera.calibrate()

    # Example of undistorted image

    img = 'camera_cal/calibration2.jpg'
    image = mpimg.imread(img)
    undist = my_camera.undistort(image)
    utils.plot_batch([image, undist], ['Original', 'Undistorted'], (8, 3))
    plt.savefig('output_images/undistortion.jpg')

    # Example of perspective transform

    test_imgs = ['color-shadow-example.jpg']
    for img in test_imgs:
        test_image = mpimg.imread('test_images/' + img)
        undist = my_camera.undistort(test_image)
        fig = plot_warp(undist)
        fig.savefig('output_images/warped_' + img)
