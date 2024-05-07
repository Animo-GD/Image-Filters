import cv2
import numpy as np
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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cv2.convertScaleAbs(gray_image), cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        # Convert coordinates and radius to integers
        circles = np.uint16(np.around(circles))

        # Draw detected circles
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Draw the outer circle
            cv2.circle(image, center, radius, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, center, 2, (0, 0, 255), 3)

    return image

# Function to perform Segmentation using Region Split and Merge
def region_split_merge(image, min_region_size=64):
    # Split and merge segmentation algorithm

    # Define the split condition (e.g., variance threshold)
    def split_condition(region):
        # Compute the variance of the region
        variance = np.var(region)
        return variance > .5

    def compute_similarity(region1, region2):
        # Ensure that both regions have the same shape
        if region1.shape != region2.shape:
            # Resize or pad the regions to have the same shape
            # For simplicity, we'll resize the regions to the minimum common shape
            min_height = min(region1.shape[0], region2.shape[0])
            min_width = min(region1.shape[1], region2.shape[1])
            region1 = cv2.resize(region1, (min_width, min_height))
            region2 = cv2.resize(region2, (min_width, min_height))

        # Compute the mean squared error (MSE) between the two regions
        mse = np.mean((region1 - region2) ** 2)
        return mse
    # Define the merge condition (e.g., intensity similarity)

    def merge_condition(region1, region2):
        # Compute some measure of similarity between region1 and region2
        similarity = compute_similarity(region1, region2)
        return similarity > .5

    def split_into_subregions(region):
        # Split the region into four quadrants
        height, width = region.shape[:2]
        mid_height = height // 2
        mid_width = width // 2

        top_left = region[:mid_height, :mid_width]
        top_right = region[:mid_height, mid_width:]
        bottom_left = region[mid_height:, :mid_width]
        bottom_right = region[mid_height:, mid_width:]

        return [top_left, top_right, bottom_left, bottom_right]
    # Recursive function to split a region into smaller subregions
    def split_region(region):
        if region.size <= min_region_size:
            return [region]

        # Split the region into smaller subregions
        subregions = split_into_subregions(region)

        # Recursively split each subregion
        smaller_regions = []
        for subregion in subregions:
            smaller_regions.extend(split_region(subregion))

        return smaller_regions

    # Function to merge adjacent regions
    def merge_two_regions(region1, region2):
        # Ensure that both regions have the same shape
        if region1.shape != region2.shape:
            # Resize or pad the regions to have the same shape
            min_height = min(region1.shape[0], region2.shape[0])
            min_width = min(region1.shape[1], region2.shape[1])
            region1 = cv2.resize(region1, (min_width, min_height))
            region2 = cv2.resize(region2, (min_width, min_height))

        # Merge the two regions by averaging pixel values
        merged_region = (region1.astype(np.float32) + region2.astype(np.float32)) // 2  # Element-wise averaging
        return merged_region.astype(np.uint8)  # Convert back to uint8 format if needed
    def merge_regions(regions):
        merged_regions = []
        # Iterate through all pairs of adjacent regions
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if merge_condition(regions[i], regions[j]):
                    # Merge the adjacent regions
                    merged_regions.append(merge_two_regions(regions[i], regions[j]))
        return merged_regions

    # Initial split of the entire image
    regions = split_region(image)

    # Merge adjacent regions iteratively until convergence
    prev_num_regions = len(regions) + 1
    while len(regions) < prev_num_regions:
        prev_num_regions = len(regions)
        regions = merge_regions(regions)

    return regions


def threshold_segmentation(image):
    _, binary_image = cv2.threshold(cv2.convertScaleAbs(image), 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    return binary_image