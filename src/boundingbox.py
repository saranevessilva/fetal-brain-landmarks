import numpy as np
import nibabel as nib
import matplotlib

# matplotlib.use('TkAgg')  # Use this backend or try 'Qt5Agg'
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import SimpleITK as sitk
import os
import pandas as pd
import csv
from csv import writer
import time
from datetime import datetime


def calculate_expanded_bounding_box(segmentation, expansion_factor):
    # Find indices of nonzero elements in the segmentation
    nonzero_indices = np.transpose(np.nonzero(segmentation))

    # Calculate the original bounding box corners
    min_coords = np.min(nonzero_indices, axis=0)
    max_coords = np.max(nonzero_indices, axis=0)

    # Calculate the dimensions of the original bounding box
    dimensions = max_coords - min_coords

    # Calculate the expansion amount for each dimension
    expansion_amount = (dimensions * expansion_factor).astype(int)

    # Calculate the expanded bounding box corners while ensuring it stays within the image boundaries
    expanded_min_coords = np.maximum(min_coords - expansion_amount, [0, 0, 0])
    expanded_max_coords = np.minimum(max_coords + expansion_amount, segmentation.shape)

    return expanded_min_coords, expanded_max_coords


def apply_bounding_box(segmentation, image):
    segmentation_volume = segmentation

    # expansion_factor = 0.25
    expansion_factor = 0.35
    # expansion_factor = 0.5  # keep it like this for the fetal scan!
    expanded_min_coords, expanded_max_coords = calculate_expanded_bounding_box(segmentation_volume, expansion_factor)

    # Calculate the side length for a square bounding box
    side_length = max(expanded_max_coords[0] - expanded_min_coords[0], expanded_max_coords[1] - expanded_min_coords[1],
                      expanded_max_coords[2] - expanded_min_coords[2])
    print("Length", side_length)

    # Calculate the center of the brain mask
    center = ndimage.center_of_mass(segmentation_volume)
    center = tuple(round(coord, 2) for coord in center)
    print("Centre", center)

    cropped_image = segmentation_volume[
                    int(center[0] - side_length // 2):int(center[0] + side_length // 2),
                    int(center[1] - side_length // 2):int(center[1] + side_length // 2),
                    int(center[2] - side_length // 2):int(center[2] + side_length // 2)
                    ]

    date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "/" + timestamp
            + "-gadgetron-fetal-brain-localisation-mask_34x34x34.nii.gz")
    nib.save(nib.Nifti1Image(cropped_image, np.eye(4)), path)
    print("Cropped image", cropped_image.shape)

    original_lower_left_corner = (
        int(center[0] - side_length // 2),
        int(center[1] - side_length // 2),
        int(center[2] - side_length // 2)
    )

    print("Original Lower Left Corner:", original_lower_left_corner)

    path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "/" + timestamp
            + "-gadgetron-fetal-brain-localisation-mask_cropped.nii.gz")
    nib.save(nib.Nifti1Image(cropped_image, np.eye(4)), path)

    image_volume = image

    # cropped = image_volume * cropped_image
    cropped = image_volume[
              int(center[0] - side_length // 2):int(center[0] + side_length // 2),
              int(center[1] - side_length // 2):int(center[1] + side_length // 2),
              int(center[2] - side_length // 2):int(center[2] + side_length // 2)
              ]

    # # Define the desired intensity range
    # new_min = 0
    # new_max = 4095
    #
    # # Calculate the current min and max intensities in the image
    # current_min = np.min(cropped)
    # current_max = np.max(cropped)
    #
    # # Perform the linear scaling
    # cropped = (cropped - current_min) * ((new_max - new_min) / (current_max - current_min)) + new_min

    # center_cropped = ndimage.center_of_mass(cropped)

    # Create a mask to specify the region where you want to place the smaller array
    mask = np.zeros(image_volume.shape)
    mask[
        original_lower_left_corner[0]: original_lower_left_corner[0] + cropped.shape[0],
        original_lower_left_corner[1]: original_lower_left_corner[1] + cropped.shape[1],
        original_lower_left_corner[2]: original_lower_left_corner[2] + cropped.shape[2]
    ] = cropped

    # # Assign the values from the smaller array to the corresponding location in the larger array
    # image_volume[mask] = cropped

    path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "/"
            + timestamp + "-gadgetron-fetal-brain-localisation-img_new.nii.gz")
    img_new = nib.Nifti1Image(mask, np.eye(4))
    nib.save(nib.Nifti1Image(mask, np.eye(4)), path)

    expansion_factor = 128 / cropped.shape[2]
    print("expansion factor:", expansion_factor)

    # Calculate the new dimensions after expansion
    new_shape = np.array(cropped.shape) * expansion_factor
    new_shape = new_shape.astype(int)  # Convert to integer dimensions

    # Use NumPy zoom function to expand the array
    expanded = np.zeros(new_shape)
    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[1]):
            for k in range(cropped.shape[2]):
                expanded[
                    int(i * expansion_factor):int((i + 1) * expansion_factor),
                    int(j * expansion_factor):int((j + 1) * expansion_factor),
                    int(k * expansion_factor):int((k + 1) * expansion_factor)
                ] = cropped[i, j, k]

    # print("Expanded cropped image", expanded.shape)

    # Define the desired intensity range
    new_min = 0
    new_max = 1578

    # Calculate the current min and max intensities in the image
    current_min = np.min(expanded)
    current_max = np.max(expanded)

    # Perform the linear scaling
    expanded = (expanded - current_min) * ((new_max - new_min) / (current_max - current_min)) + new_min

    path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "/"
            + timestamp + "-gadgetron-fetal-brain-localisation-img_cropped.nii.gz")
    img_expanded = nib.Nifti1Image(expanded, np.eye(4))
    nib.save(nib.Nifti1Image(expanded, np.eye(4)), path)

    offset = original_lower_left_corner
    print("OFFSET", offset)

    return expanded, expansion_factor, center, offset, side_length, mask, segmentation_volume, cropped_image
