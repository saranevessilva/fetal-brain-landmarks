import os
import nibabel as nib
import numpy as np


# Define a function to count unique labels in a segmentation image
def count_labels(segmentation_image):
    labels = np.unique(segmentation_image)
    return len(labels)


# Define a directory where your NIfTI files are stored
data_dir = '/home/sn21/autoflow_landmarks/0.55T/'

# List all the NIfTI files in the directory
nifti_files = [f for f in os.listdir(data_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

# Loop through the NIfTI files and count labels in each
for file_name in nifti_files:
    file_path = os.path.join(data_dir, file_name)
    img = nib.load(file_path)
    data = img.get_fdata()

    # Count the number of labels in the segmentation image
    num_labels = count_labels(data)

    print(f"File: {file_name}, Number of labels: {num_labels}")

# Define a directory where your NIfTI files are stored
data_dir = '/home/sn21/autoflow_landmarks/1.5T/'

# List all the NIfTI files in the directory
nifti_files = [f for f in os.listdir(data_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

# Loop through the NIfTI files and count labels in each
for file_name in nifti_files:
    file_path = os.path.join(data_dir, file_name)
    img = nib.load(file_path)
    data = img.get_fdata()

    # Count the number of labels in the segmentation image
    num_labels = count_labels(data)

    print(f"File: {file_name}, Number of labels: {num_labels}")
