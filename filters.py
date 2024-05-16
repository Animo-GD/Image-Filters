import cv2
import numpy as np
from skimage.segmentation import slic
from skimage import color


def LPF(image):
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(cv2.convertScaleAbs(image), -1, kernel)
    return filtered_image


# Function to apply High Pass Filter (HPF)
def HPF(image):
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float32)
    filtered_image = cv2.filter2D(cv2.convertScaleAbs(image), -1, laplacian_kernel)
    return filtered_image


# Function to apply Mean Filter
def mean_filter(image):
    kernel_size = (5, 5)
    filtered_image = cv2.blur(cv2.convertScaleAbs(image), kernel_size)
    return filtered_image


# Function to apply Median Filter
def median_filter(image):
    kernel_size = 3
    filtered_image = cv2.medianBlur(cv2.convertScaleAbs(image), kernel_size)
    return filtered_image


# Function to apply Roberts Cross Operator
def roberts(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = np.float32(gray_image)
    # Define Roberts kernels
    roberts_kernel_x = np.array([[1, 0],
                                 [0, -1]], dtype=np.float32)
    roberts_kernel_y = np.array([[0, 1],
                                 [-1, 0]], dtype=np.float32)

    # Apply Roberts edge detection
    gradient_x = cv2.filter2D(gray_image, -1, roberts_kernel_x)
    gradient_y = cv2.filter2D(gray_image, -1, roberts_kernel_y)

    # Compute the magnitude of the gradients
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = np.uint8(gradient_magnitude)
    return gradient_magnitude


# Function to apply Prewitt Operator
def prewitt(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    prewitt_kernel_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=np.float32)
    prewitt_kernel_y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]], dtype=np.float32)
    gradient_x = cv2.filter2D(cv2.convertScaleAbs(gray_image), -1, prewitt_kernel_x)
    gradient_y = cv2.filter2D(cv2.convertScaleAbs(gray_image), -1, prewitt_kernel_y)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = np.uint8(gradient_magnitude)
    return gradient_magnitude


# Function to apply Sobel Operator
def sobel(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    gradient_x = cv2.Sobel(cv2.convertScaleAbs(gray_image), cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(cv2.convertScaleAbs(gray_image), cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = np.uint8(gradient_magnitude)
    return gradient_magnitude


# Function to apply Erosion
def erosion(image):
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(cv2.convertScaleAbs(image), kernel, iterations=1)
    return eroded_image


# Function to apply Dilation
def dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(cv2.convertScaleAbs(image), kernel, iterations=1)
    return dilated_image


# Function to apply Opening
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(cv2.convertScaleAbs(image), cv2.MORPH_OPEN, kernel)
    return opened_image


# Function to apply Closing
def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(cv2.convertScaleAbs(image), cv2.MORPH_CLOSE, kernel)
    return closed_image


# Function to apply Hough Transform for Circle Detection
def hough_circle(image):
    filtered_image = image.copy()

    sobel_edge = sobel(filtered_image)

    circles = cv2.HoughCircles(sobel_edge, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=300, param2=50, minRadius=10, maxRadius=70)

    if circles is not None:
        # Convert coordinates and radius to integers
        circles = np.uint16(np.around(circles))

        # Draw detected circles
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Draw the outer circle
            cv2.circle(filtered_image, center, radius, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(filtered_image, center, 2, (0, 0, 255), 3)

    return filtered_image


# Function to perform Segmentation using Region Split and Merge
def region_split_merge(image, min_region_size=64):
    segments = slic(image, n_segments=100, compactness=10.0)
    segmented_image = color.label2rgb(segments, image, kind='avg')
    return segmented_image


def threshold_segmentation(image):
    _, binary_image = cv2.threshold(cv2.convertScaleAbs(image), 127, 255, cv2.THRESH_BINARY)
    return binary_image
