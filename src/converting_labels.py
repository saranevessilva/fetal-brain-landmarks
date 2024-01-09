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
from scipy import ndimage, misc
from PIL import Image

# Define the directory where you want to search for files
search_directory = '/home/sn21/landmark-data/Landmarks/nnUNet_raw/Dataset001_Landmarks/labelsTr'

# Use glob to find all files in subdirectories with "001.nii" in the file path
file_paths = glob.glob(os.path.join(search_directory, '*nii.gz*'), recursive=True)

# Iterate over the found file paths
for path in file_paths:
    print(path)
    # Load the image using sitk.ReadImage
    im = sitk.ReadImage(path)
    # Get the image data as a NumPy array
    im = sitk.GetArrayFromImage(im)

    modified_image_array = np.copy(im)

    # Replace label values 2 with 1
    modified_image_array[np.where(im == 2)] = 1

    # Replace label values 3 with 2
    modified_image_array[np.where(im == 3)] = 2

    # Convert the modified NumPy array back to a SimpleITK image
    modified_image = sitk.GetImageFromArray(modified_image_array)

    # Save the modified image with the same file path
    output_image_path = path
    sitk.WriteImage(modified_image, output_image_path)

