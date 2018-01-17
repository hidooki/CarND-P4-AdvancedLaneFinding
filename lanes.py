"""Functions for lane finding """

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import image as mpimg


def boot_quad_fit(binary_warped,
                  base_xband,
                  nwindows=9,
                  margin=100,
                  minpix=50):
    """Start curve finding by a sliding window search

    Input:
        binary_warped: table of hot points representing candidate pixels
            for lane fitting. Typically a result of gradiend and/or color
            thresholding
        base_xband: search only the lanes crossing the bottom of the frame
            in the (x_min, x_max) band
        nwindows: number of windows (stacked) per lane
        margin: half-width of the search window
        minpix: low threshold for number of hot pixels required per window

    Output:
        quad_fit: coefficients of second degree polynomial fit
        sliding_windows: array of windows by level
    """

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[base_xband[0]:base_xband[1], :], axis=0)

    # Find the peak of histogram; this will be the starting point or the line
    x_base = np.argmax(histogram) + base_xband[0]

    window_height = np.int(binary_warped.shape[0] / nwindows)

    nonzeroy, nonzerox = binary_warped.nonzero()

    # Current positions to be updated for each window
    x_current = x_base

    # Create empty list to receive lane pixel indices
    line_inds = []

    sliding_windows = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_x_low = x_current - margin
        win_x_high = x_current + margin

        window = (win_x_low, win_y_low), (win_x_high, win_y_high)
        sliding_windows.append(window)

        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) &
                     (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_low) &
                     (nonzerox < win_x_high)).nonzero()[0]

        # Append these indices to the lists
        line_inds.append(good_inds)

        # If found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices
    line_inds = np.concatenate(line_inds)

    # Extract left and right line pixel positions
    x, y = nonzerox[line_inds], nonzeroy[line_inds]

    # Fit a second order polynomial to each
    quad_fit = np.polyfit(y, x, 2)

    return quad_fit, sliding_windows


def plot_sliding_windows(
        binary_warped,
        nwindows=9,
        margin=100,
        minpix=50):

    midpoint = binary_warped.shape[1] // 2

    left_fit, left_windows = boot_quad_fit(
        binary_warped,
        (0, midpoint),
        nwindows,
        margin,
        minpix)

    right_fit, right_windows = boot_quad_fit(
        binary_warped,
        (midpoint, binary_warped.shape[1]),
        nwindows,
        margin,
        minpix)

    plt.interactive(True)

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    for window in left_windows + right_windows:
        cv2.rectangle(out_img, window[0], window[1], (0, 255, 0), 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def fit_quad(binary_warped, curve_fit, margin=100):
    """Find quadratic given approximate starting points

    Input:
        binary_warped: table of hot points representing candidate pixels
            for lane fitting. Typically a result of gradiend and/or color
            thresholding
        curve_fit: starting points for curve searching, typically
            the fitted curve of the prior frame
        margin: half-width of the search window

    Output:
        quad_line_fit: line of the current frame
    """
    nonzeroy, nonzerox = binary_warped.nonzero()

    x = curve_fit[0] * (nonzeroy**2) + curve_fit[1] * nonzeroy + curve_fit[2]
    lane_inds = np.abs(nonzerox - x) < margin

    # Extract left and right line pixel positions
    x, y = nonzerox[lane_inds], nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    quad_fit = np.polyfit(y, x, 2)

    return quad_fit


def plot_lane_fit(binary_warped, left_fit, right_fit, margin=100):
    """Plot lane on an image given left and right edge curves"""

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    P_left = np.poly1d(left_fit)
    left_fitx = P_left(ploty)

    P_right = np.poly1d(right_fit)
    right_fitx = P_right(ploty)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    nonzeroy, nonzerox = binary_warped.nonzero()
    left_lane_inds = np.abs(nonzerox - P_left(nonzeroy)) < margin
    right_lane_inds = np.abs(nonzerox - P_right(nonzeroy)) < margin

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def curvature(y, coeffs, ym_per_pix=40 / 720, xm_per_pix=3.7 / 670):
    """Curvature of quadratic function at point y

    Input:
        coeffs: array of function coefficients, in decreasing degree order
        ym_per_pix: meters per pixel in y direction
        xm_per_pix: meters per pixel in x direction

    Output:
        curvature
    """
    a, b, _ = coeffs
    r = xm_per_pix / ym_per_pix
    A = a * r / ym_per_pix
    B = b * r
    return (1 + (2 * A * y + B)**2)**1.5 / np.abs(2 * A)


if __name__ == '__main__':

    binary_warped = mpimg.imread('examples/warped-example.jpg')

    margin = 100

    # Boot lane fit and visualize the results

    plot_sliding_windows(binary_warped, margin=margin)
    plt.savefig('output_images/sliding_windows.jpg')
    plt.title('Line detection by "sliding windows"')

    midpoint = binary_warped.shape[1] // 2
    left_fit, _ = boot_quad_fit(binary_warped, (0, midpoint))
    right_fit, _ = boot_quad_fit(
        binary_warped,
        (midpoint, binary_warped.shape[1]),
        margin=margin)

    # Subsequent lane fit using prior frame fit

    left_fit = fit_quad(binary_warped, left_fit)
    right_fit = fit_quad(binary_warped, right_fit)

    plt.figure()
    plot_lane_fit(binary_warped, left_fit, right_fit, margin=100)
    plt.savefig('output_images/lines_fit.jpg')
    plt.title('Line detection with given starting curve')
