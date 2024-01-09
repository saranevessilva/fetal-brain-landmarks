#!/usr/bin/python


###########################################################################


import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

import skimage
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.measure import label, regionprops

import sys
import os

import torch
import monai
from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, UNet, AttentionUnet

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


###########################################################################

# test nii image to create input test image matrix

global_img = nib.load('/home/sn21/data/t2-stacks/2023-09-28/all.nii.gz')

input_matrix_image_data = global_img.get_fdata()


###########################################################################

# note: modify paths to model

cl_num_densenet = 2
cl_num_unet = 2

model_weights_path_densenet="/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/checkpoints/best_metric_model_densenet.pth"
model_weights_path_unet="/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/checkpoints/best_metric_model_unet.pth"


###########################################################################

# conver to tensor & apply transforms:
# - correct for 4.5 mm slice thickness
# - pad to square
# - resize to 128x128x128
# - scale to [0; 1]

input_image = torch.tensor(input_matrix_image_data).unsqueeze(0)

zoomer = monai.transforms.Zoom(zoom=(1,1,3), keep_size=False)
zoomed_image = zoomer(input_image)

required_spatial_size = max(zoomed_image.shape)
padder = monai.transforms.SpatialPad(spatial_size=required_spatial_size, method="symmetric")
padded_image = padder(zoomed_image)

spatial_size = [128, 128, 128]
resizer = monai.transforms.Resize(spatial_size=(128,128,128))
resampled_image = resizer(padded_image)

scaler = monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
final_image = scaler(resampled_image)


###########################################################################

# define and load UNet model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmentation_model = UNet(spatial_dims=3,
    in_channels=1,
    out_channels=cl_num_unet+1,
    channels=(32, 64, 128, 256, 512),
    strides=(2,2,2,2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=1,
    act='PRELU',
    norm='INSTANCE',
    dropout=0.5
)

with torch.no_grad():
  segmentation_model.load_state_dict(torch.load(model_weights_path_unet), strict=False)
  segmentation_model.to(device)


###########################################################################


# run segmentation (multiple times and average to reduce variability) and argmax the labels

segmentation_inputs = final_image.unsqueeze(0).to(device)

with torch.no_grad():

    # segmentation_output = sliding_window_inference(segmentation_inputs, (128, 128, 128), 4, segmentation_model, overlap=0.8)
    segmentation_output1 = segmentation_model(segmentation_inputs)
    segmentation_output2 = segmentation_model(segmentation_inputs)
    segmentation_output3 = segmentation_model(segmentation_inputs)
    segmentation_output4 = segmentation_model(segmentation_inputs)
    segmentation_output = ( segmentation_output1 + segmentation_output2 + segmentation_output3 + segmentation_output4 ) / 4


label_output = torch.argmax(segmentation_output, dim=1).detach().cpu()[0, :, :, :]
label_matrix = label_output.cpu().numpy()


###########################################################################


# extract brain label and dilate it

# extract brain label
label_2_mask = label_matrix == 2
label_brain = np.where(label_2_mask, label_matrix, 0)

# largest connected component

labeled_components, num_components = skimage.measure.label(label_brain, connectivity=2, return_num=True)
component_sizes = [np.sum(labeled_components == label) for label in np.unique(labeled_components) if label != 0]
largest_component_label = np.argmax(component_sizes) + 1
largest_component_mask = (labeled_components == largest_component_label)

# print(num_components)

label_brain = largest_component_mask
label_brain = label_brain > 0


test_zero_brain = np.all(label_brain == 0)

if not test_zero_brain:

    # dilate
    diamond = ndimage.generate_binary_structure(rank=3, connectivity=1)
    dilated_label_brain = ndimage.binary_dilation(label_brain, diamond, iterations=5)




    ###########################################################################


    # transform dilated label to the original padded image & crop padded image

    padded_image_matrix = padded_image.cpu().numpy()[0, :, :, :]

    # print(padded_image_matrix.shape)
    # print(dilated_label_brain.shape)

    scale_factors = [
        (padded_image_matrix.shape[0] / dilated_label_brain.shape[0]),
        (padded_image_matrix.shape[1] / dilated_label_brain.shape[1]),
        (padded_image_matrix.shape[2] / dilated_label_brain.shape[2])
    ]

    dilated_label_brain = ndimage.zoom(dilated_label_brain, zoom=scale_factors, order=0)

    # crop padded image
    nonzero_indices = np.argwhere(dilated_label_brain == 1)
    min_indices = np.min(nonzero_indices, axis=0)
    max_indices = np.max(nonzero_indices, axis=0)

    cropped_image_matrix = padded_image_matrix[min_indices[0]:max_indices[0] + 1, min_indices[1]:max_indices[1] + 1, min_indices[2]:max_indices[2] + 1]



    ###########################################################################



    # pad, resample and scale cropped image

    input_cropped_image = torch.tensor(cropped_image_matrix).unsqueeze(0)

    required_spatial_size = max(input_cropped_image.shape)
    padder = monai.transforms.SpatialPad(spatial_size=required_spatial_size, method="symmetric")
    padded_cropped_image = padder(input_cropped_image)

    resizer = monai.transforms.Resize(spatial_size=(96, 96, 96))
    resampled_cropped_image = resizer(padded_cropped_image)

    scaler = monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    final_cropped_image = scaler(resampled_cropped_image)



    ###########################################################################



    # define, load and run classifier model

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classification_model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=cl_num_densenet)

    with torch.no_grad():
      classification_model.load_state_dict(torch.load(model_weights_path_densenet), strict=False)
      classification_model.to(device)


    ###########################################################################




    # run model and print the class

    classifier_intput_image = final_cropped_image.unsqueeze(0).to(device)

    with torch.no_grad():
      classifier_output_tensor1 = classification_model(classifier_intput_image)
      classifier_output_tensor2 = classification_model(classifier_intput_image)
      classifier_output_tensor = classifier_output_tensor1 + classifier_output_tensor2 / 2


    predicted_probabilities = torch.softmax(classifier_output_tensor, dim=1)
    class_out = torch.argmax(predicted_probabilities, dim=1)
    predicted_class = class_out.item()

    print(" - predicted class : ", predicted_class)


else :

#    zero brain mask condition

    print(" - predicted class : ", 0)

###########################################################################


