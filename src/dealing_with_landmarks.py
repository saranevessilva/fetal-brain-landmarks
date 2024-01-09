# noinspection PyPep8Naming
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv
from csv import writer
import glob

from skimage import transform, filters
from scipy.ndimage import binary_dilation

from scipy import ndimage, misc
from PIL import Image
from scipy import ndimage, misc
from sklearn.cluster import KMeans
import datetime



# Define the directory where you want to search for files
search_directory = '/home/sn21/landmark-data/Landmarks/nnUNet_results/Dataset001_Landmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation'

# Use glob to find all files with "nii.gz" in the file name
file_paths = glob.glob(os.path.join(search_directory, '*nii.gz*'), recursive=True)

# Filter out files with "_0000" in the file name
file_paths = [path for path in file_paths if "_0000" and "3-labels" not in os.path.basename(path)]


# Iterate over the found file paths
for path in file_paths:
    print(path)
    # Load the image using sitk.ReadImage
    landmark = sitk.ReadImage(path)
    # Get the image data as a NumPy array
    landmark = sitk.GetArrayFromImage(landmark)
    modified_landmark = np.copy(landmark)

    # Replace label values 2 with 1
    landmark[np.where(landmark == 2.0)] = 3.0

    modified_landmark[np.where(modified_landmark == 2.0)] = 0.0

    # Define a structuring element (a kernel) for morphological operations
    structuring_element = np.ones((3, 3, 3))  # Adjust size if needed

    # Perform morphological closing
    closed_segmentation = ndimage.binary_closing(modified_landmark, structure=structuring_element).astype(modified_landmark.dtype)

    labelled_segmentation, num_features = ndimage.label(connected_segmentation)
    print("number of features", num_features)
    # region_sizes = np.bincount(labelled.ravel())[1:]

    mask = (landmark == 3.0)

    labelled_segmentation[mask] = 3.0

    def find_center_of_region(image_array, target_value):
        h, w, d = np.where(image_array == target_value)
        center_x = int(np.mean(h))
        center_y = int(np.mean(w))
        center_z = int(np.mean(d))

        return center_x, center_y, center_z

    cm_eye_1 = find_center_of_region(labelled_segmentation, 1.0)
    cm_eye_2 = find_center_of_region(labelled_segmentation, 2.0)
    cm_cereb = find_center_of_region(labelled_segmentation, 3.0)

    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Define the file name with the formatted date and time
    text_file = f"/home/sn21/landmark-data/Landmarks/output_{date_time_string}.txt"

    # Create and write to the text file
    with open(text_file, "w") as file:
        file.write("This is a text file created on " + date_time_string)
        file.write("\n" + str('CoM: '))
        file.write("\n" + "cm_eye_1" + str(cm_eye_1))
        file.write("\n" + "cm_eye_2" + str(cm_eye_2))
        file.write("\n" + "cm_cereb" + str(cm_cereb))

    print(f"Text file '{text_file}' has been created.")

    # Convert the modified NumPy array back to a SimpleITK image
    modified_image = sitk.GetImageFromArray(labelled_segmentation)

    # Save the modified image with the same file path
    output_image_path = path.replace(".nii.gz", "_3-labels.nii.gz")

    sitk.WriteImage(modified_image, output_image_path)
