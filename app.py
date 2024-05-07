import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from filters import *

# Function to perform Segmentation using Thresholding
def threshold_segmentation(image):
    _, binary_image = cv2.threshold(cv2.convertScaleAbs(image), 127, 255, cv2.THRESH_BINARY)
    return binary_image

# Function to apply the selected filter
def apply_filter():
    selected_filter = filter_choice.get()
    if selected_filter == 'LPF':
        filtered_image = LPF(original_image)
    elif selected_filter == 'HPF':
        filtered_image = HPF(original_image)
    elif selected_filter == 'Mean':
        filtered_image = mean_filter(original_image)
    elif selected_filter == 'Median':
        filtered_image = median_filter(original_image)
    elif selected_filter == 'Roberts':
        filtered_image = roberts(original_image)
    elif selected_filter == 'Prewitt':
        filtered_image = prewitt(original_image)
    elif selected_filter == 'Sobel':
        filtered_image = sobel(original_image)
    elif selected_filter == 'Erosion':
        filtered_image = erosion(original_image)
    elif selected_filter == 'Dilation':
        filtered_image = dilation(original_image)
    elif selected_filter == 'Opening':
        filtered_image = opening(original_image)
    elif selected_filter == 'Closing':
        filtered_image = closing(original_image)
    elif selected_filter == 'Hough Circle':
        filtered_image = hough_circle(original_image)
    elif selected_filter == 'Region Split and Merge':
        filtered_image = region_split_merge(original_image)
    elif selected_filter == 'Thresholding':
        filtered_image = threshold_segmentation(original_image)
    else:
        filtered_image = original_image

    if filtered_image is not None:
        display_image(original_image, filtered_image)

# Function to open an image file
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        global original_image
        original_image = cv2.imread(file_path)
        display_image(original_image, None)
# Function to display the image
def display_image(original, filtered):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = Image.fromarray(original)
    original.thumbnail((400, 400))
    original_photo = ImageTk.PhotoImage(original)
    label_original.config(image=original_photo)
    label_original.image = original_photo

    if filtered is not None:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        filtered = Image.fromarray(filtered)
        filtered.thumbnail((400, 400))
        filtered_photo = ImageTk.PhotoImage(filtered)
        label_filtered.config(image=filtered_photo)
        label_filtered.image = filtered_photo

# Create main window
root = tk.Tk()
root.title("Image Filter Application")
upload_button = tk.Button(root, text="Upload Image", command=open_file)
upload_button.pack()
# Create drop-down menu for filters
filters = ['LPF', 'HPF', 'Mean', 'Median', 'Roberts', 'Prewitt', 'Sobel',
           'Erosion', 'Dilation', 'Opening', 'Closing', 'Hough Circle',
           'Region Split and Merge', 'Thresholding']
filter_choice = tk.StringVar(root)
filter_choice.set(filters[0])  # Set default filter
filter_menu = tk.OptionMenu(root, filter_choice, *filters)
filter_menu.pack()

# Create button to apply filter
apply_button = tk.Button(root, text="Apply Filter", command=apply_filter)
apply_button.pack()

# Create labels to display original and filtered images
label_original = tk.Label(root)
label_original.pack(side=tk.LEFT)
label_filtered = tk.Label(root)
label_filtered.pack(side=tk.LEFT)

# Global variable to store the original image
original_image = None

root.mainloop()