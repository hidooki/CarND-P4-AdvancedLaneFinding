# **Advanced Lane Finding**


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

It is graded according to the following  [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

[//]: # (Image References)

[image1]: ./output_images/undistortion.jpg "Undistorted"
[image2]: ./test_images/color-shadow-example.jpg "Road Transformed"
[image3]: ./output_images/threshold_color-shadow-example.jpg "Binary Example"
[image4]: ./output_images/warped_color-shadow-example.jpg "Warp Example"
[image5]: ./output_images/sliding_windows.jpg "Sliding Windows"
[image6]: ./output_images/lanes_straight_lines1.jpg "Output"
[image7]: ./output_images/lines_fit.jpg "Fit Visual"
[video1]: ./project_video_result.mp4 "Video"


---

### Submitted Files

The following are included with the project:
- project writeup
- code: camera.py, thresholding.py, lanes.py, utils.py, runner.py
- images: included in output_images folder
- video result: project_video_result.mp4

### Camera Calibration

#### Camera matrix and distortion coefficients

The code for this step is contained in `camera` module. The calibration is implemented in the function `calibrate_camera()`. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

The image undistortion is implemented in the `undistort()` function within
the same module. The camera calibration parameters are required, which will
be passed in to `cv2.undistort()` which does the actual undistortion. Here is an example:

![alt text][image1]


Since I wanted to run camera calibration just once, and then use the same calibration parameters multiple times while working on the project, I created
a `Camera` class which remembers its parameters (by streaming them do disk) once calibrated.

### Pipeline (single images)

#### 1. Distortion-corrected image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Using color transforms, gradients to create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image (thresholding functions in `thresholding` module). For gradient thresholding, I applied a Sobel filter in the horizontal direction. For color
thresholding, I first converted the image to the HLS color space and then
I filtered for high saturation _and_ hue values lower than 90 (since lines are
always white or yellow). Here's an example of my binary output  

![alt text][image3]

#### 3. Perspective transform

The code for my perspective transform is included in the `camera` module. The `warp_perspective()` function takes as inputs an image, typically undistorted, as well as source and destination points.  After examining the
test image above, I chose to hardcode the source and destination points in the following manner:


| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 300, 720      |
| 595, 450      | 300, 0        |
| 695, 450      | 970, 0        |
| 1120, 720     | 970, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a the above test image to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Line fitting with a quadratic polynomial

The code for line fitting in contained in the `lanes` module. The main two
functions are `boot_quad_fit()` and `fit_quad()`. The first will attempt
a curve fit using the 'sliding windows' method described in the lecture. The warped binary image is divided into 9 layers and the first pair of windows is
placed on the bottom layer based on a poll (histogram) of the number of hot
pixels per column. As we scan the image upwards, the windows are re-centered to always follow the bulk of the hot pixels. In the end, a quadratic function
is fit and the result is similar to this:

![alt text][image5]


The latter fitting function starts from an 'initial value' curve and looks for the best fit in its immediate vicinity. A typical result is illustrated below. In both case, the line search is done one line at a time, so the caller will typically call them twice, one for the left lane and again for the right lane.


![alt text][image7]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center

The radius of curvature is calculated in the `curvature()` function of the `lanes` module. I used the formula for the curvature of a quadratic function. I examined the warped images to gauge the distance in meters represented by one pixel in both the x and the y direction. Using the standard specifications for the width of a highway lane, I determined the scale of the warped image and converted the curvature formula accordingly, so that the radius would be reported in meters.



#### 6. Result plotted back down onto the road

Putting everything together, the pipeline in the `runner` module will process a given image as follows


![alt text][image6]

---

### Pipeline (video)

The pipeline is run over an entire clip frame by frame, as illustrated in the 'main' section of the `runner` module. It performs reasonably well on the entire project video. The line detection hesitates ever so slightly in two places on the road where the lines' color washes away in the pavement, but it is barely perceptible and it recovers very quickly.

Here's a [link](./project_video_result.mp4) to my video result.

---

### Discussion

This was a relatively straightforward project, especially given the extensive code examples provided in the lectures.

In order ensure a smooth update of the lane from one frame to the next, I used
an exponential filter on the quadratic coefficients. I had to experiment with different values of alpha (the exponential decay parameter). However, the value I settled on might become inadequate if the car accelerates or the road starts to wind more (as the coefficients of the fit would change more rapidly). A possible improvement would be to account for the driving speed in selecting the value of this parameter.
