"""Pipeline implementation"""
import cv2
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np

import camera
import thresholding
import lanes


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.current_fit = np.array([0, 0, 0])
        # radius of curvature of the line in some units
        self.radius_of_curvature = None


def make_pipeline(alpha=0.3):
    left_line = Line()
    right_line = Line()

    my_camera = camera.Camera()
    my_camera.calibrate()

    def pipeline(image):
        """Pipeline for frame proceesing

        Input: An image of the road in front

        Ouput: Image with lane detected
        """
        # correct for camera distortion
        undist = my_camera.undistort(image)

        # gradient and color thresholding
        c_binary = thresholding.hls_threshold(
            undist,
            h_thresh=(0, 90),
            s_thresh=(170, 255))

        g_binary = thresholding.abs_sobel_thresh(
            undist,
            orient='x',
            sobel_kernel=3,
            thresh=(40, 100))

        binary = np.logical_or(c_binary, g_binary).astype(np.uint8)

        # perspective transform
        SRC, DST = camera.SRC, camera.DST
        binary_warped = camera.warp_perspective(binary, SRC, DST)
        ny, nx = binary_warped.shape

        # fit lanes
        midpoint = nx // 2
        xbands = [(0, midpoint), (midpoint, nx)]
        for line, xband in zip([left_line, right_line], xbands):

            if not line.detected:
                # sliding windows to boot
                line_fit, _ = lanes.boot_quad_fit(binary_warped, xband)
                line.detected = True
                line.current_fit = line_fit
            else:
                # search near line of prior frame
                line_fit = lanes.fit_quad(
                    binary_warped,
                    line.current_fit,
                    margin=50)
                line.current_fit = alpha * line_fit + \
                    (1 - alpha) * line.current_fit

            # radius of curvature of the line in some units
            crv = lanes.curvature(ny - 100, line.current_fit)
            line.radius_of_curvature = crv

        # Draw lane on original image
        ploty = np.linspace(0, ny - 1, ny)

        left_fitx = np.poly1d(left_line.current_fit)(ploty)
        right_fitx = np.poly1d(right_line.current_fit)(ploty)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        def pts_stack(fitx): return np.transpose(np.vstack([fitx, ploty]))
        pts_left = np.array([pts_stack(left_fitx)])
        pts_right = np.array([np.flipud(pts_stack(right_fitx))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space
        newwarp = camera.warp_perspective(color_warp, DST, SRC)
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Annotate curvature
        mean_crv = (left_line.radius_of_curvature +
                    right_line.radius_of_curvature) / 2
        effective_crv = np.min((mean_crv, 10000))
        xxl = '+' if mean_crv >= 10000 else ''
        cv2.putText(
            result,
            'Radius of Curvature: {:,.0f}{} (m)'.format(effective_crv, xxl),
            (100, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 255, 255),
            thickness=cv2.LINE_4)

        # Annotate distance from center
        xm_per_pix = 3.7 / 670
        d_pix = (left_fitx[-1] + right_fitx[-1]) / 2 - midpoint
        d_m = xm_per_pix * d_pix
        lr = 'left' if d_m < 0 else 'right'
        cv2.putText(
            result,
            'Vehicle is {:.2f} m {} of center'.format(abs(d_m), lr),
            (100, 110),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 255, 255),
            thickness=cv2.LINE_4)

        return result

    return pipeline


if __name__ == '__main__':

    # Example of lane identified

    # test_img = 'straight_lines1.jpg'
    # image = mpimg.imread('test_images/%s' % test_img)
    # pipe = make_pipeline()
    # result = pipe(image)
    # plt.interactive(True)
    # plt.imshow(result)
    # plt.savefig('output_images/lanes_%s' % test_img)

    # Run pipeline on project video

    pipe = make_pipeline(alpha=0.3)
    clip_name = "harder_challenge_video.mp4"
    clip = VideoFileClip(clip_name)

    out_clip = clip.fl_image(pipe)
    out_clip.write_videofile(clip_name.replace('.', '_result.'), audio=False)
