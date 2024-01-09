import numpy as np
import gadgetron
import ismrmrd
import logging
import time
import io
import os
from datetime import datetime
import subprocess

from ismrmrd.meta import Meta
import itertools
import ctypes
import numpy as np
import copy
import glob
import warnings
from scipy import ndimage, misc
from skimage import measure
from scipy.spatial.distance import euclidean

warnings.simplefilter('default')

from ismrmrd.acquisition import Acquisition
from ismrmrd.flags import FlagsMixin
from ismrmrd.equality import EqualityMixin
from ismrmrd.constants import *

import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import sys

import nibabel as nib
import SimpleITK as sitk

import src.utils as utils
from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md
from src.boundingbox import calculate_expanded_bounding_box, apply_bounding_box
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


def get_first_index_of_non_empty_header(header):
    # if the data is under-sampled, the corresponding acquisition Header will be filled with 0
    # in order to catch valuable information, we need to catch a non-empty header
    # using the following lines

    print(np.shape(header))
    dims = np.shape(header)
    for ii in range(0, dims[0]):
        # print(header[ii].scan_counter)
        if header[ii].scan_counter > 0:
            break
    print(ii)
    return ii


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def send_reconstructed_images_wcm(connection, data_array, rotx, roty, rotz, cmx, cmy, cmz, acq_header):
    # this function sends the reconstructed images with centre-of-mass stored in the image header
    # the function creates a new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisitionHeader
    # fill additional fields
    # and send the reconstructed image and ImageHeader to the next gadget

    # get header info
    hdr = connection.header
    enc = hdr.encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    ncoils = 1

    dims = data_array.shape

    # print(acq_header)

    # base_header = acq_header
    ndims_image = (dims[0], dims[1], dims[2])

    base_header = ismrmrd.ImageHeader()
    base_header.version = 1
    # ndims_image = (dims[0], dims[1], dims[2], dims[3])
    base_header.measurement_uid = acq_header.measurement_uid
    base_header.position = acq_header.position
    base_header.read_dir = acq_header.read_dir
    base_header.phase_dir = acq_header.phase_dir
    base_header.slice_dir = acq_header.slice_dir
    base_header.patient_table_position = acq_header.patient_table_position
    base_header.acquisition_time_stamp = acq_header.acquisition_time_stamp
    base_header.physiology_time_stamp = acq_header.physiology_time_stamp
    base_header.user_float[0] = rotx
    base_header.user_float[1] = roty
    base_header.user_float[2] = rotz
    base_header.user_float[3] = cmx
    base_header.user_float[4] = cmy
    base_header.user_float[5] = cmz

    # base_header.user_float = (rotx, roty, rotz, cmx, cmy, cmz)

    print("cmx ", base_header.user_float[3], "cmy ", base_header.user_float[4], "cmz ", base_header.user_float[5])
    # print("------ BASE HEADER ------", base_header)

    ninstances = nslices * nreps
    # r = np.zeros((dims[0], dims[1], dims[2], dims[3]))
    r = data_array
    # print(data_array.shape)
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    image_array = ismrmrd.Image.from_array_wcm(rotx, roty, rotz, cmx, cmy, cmz, r, headers=acq_header)

    # image_array = ismrmrd.ImageHeader.from_acquisition(acq_header)
    print("..................................................................................")
    logging.info("Last slice of the repetition reconstructed - sending to the scanner...")
    connection.send(image_array)
    # print(base_header)
    logging.info("Sent!")
    print("..................................................................................")


def find_center_of_region(image_array, target_value):
    h, w, d = np.where(image_array == target_value)
    center_x = int(np.mean(h))
    center_y = int(np.mean(w))
    center_z = int(np.mean(d))

    return center_x, center_y, center_z


def send_reconstructed_images(connection, data_array, acq_header):
    # the function creates a new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisitionHeader
    # fill additional fields
    # and send the reconstructed image and ImageHeader to the next gadget

    # get header info
    hdr = connection.header
    enc = hdr.encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    ncoils = 1

    dims = data_array.shape

    base_header = ismrmrd.ImageHeader()
    base_header.version = acq_header.version
    ndims_image = (dims[0], dims[1], dims[2], dims[3])
    base_header.channels = ncoils  # The coils have already been combined
    base_header.matrix_size = (data_array.shape[0], data_array.shape[1], data_array.shape[2])
    base_header.position = acq_header.position
    base_header.read_dir = acq_header.read_dir
    base_header.phase_dir = acq_header.phase_dir
    base_header.slice_dir = acq_header.slice_dir
    base_header.patient_table_position = acq_header.patient_table_position
    base_header.acquisition_time_stamp = acq_header.acquisition_time_stamp
    base_header.image_index = 0
    base_header.image_series_index = 0
    base_header.data_type = ismrmrd.DATATYPE_CXFLOAT
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    base_header.repetition = acq_header.repetition

    ninstances = nslices * nreps
    r = np.zeros((dims[0], dims[1], ninstances))
    r = data_array
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    image_array = ismrmrd.image.Image.from_array(r, headers=base_header)

    print("..................................................................................")
    logging.info("Sending reconstructed slice to the scanner...")
    connection.send(image_array)
    logging.info("Sent!")
    print("..................................................................................")


def IsmrmrdToNiftiGadget(connection):
    date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    logging.info("Initializing data processing in Python...")
    # start = time.time()

    # get header info
    hdr = connection.header
    enc = hdr.encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    if enc.encodingLimits.contrast is not None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1
    print("Number of echoes =", ncontrasts)

    ncoils = 1

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    # print("eNx", eNx, "eNy", eNy, "eNz", eNz)
    pixdim_x = enc.encodedSpace.fieldOfView_mm.x / enc.encodedSpace.matrixSize.x
    pixdim_y = enc.encodedSpace.fieldOfView_mm.y / enc.encodedSpace.matrixSize.y
    pixdim_z = enc.encodedSpace.fieldOfView_mm.z

    # Initialise a storage array
    eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1

    ninstances = nslices * nreps
    # print("Number of instances ", ninstances)

    im = np.zeros((eNx, eNy, nslices), dtype=np.complex64)
    print("Image Shape ", im.shape)

    # file = "/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "-" + timestamp
    # + "-centreofmass.txt"
    # file = "/home/sns/fetal-brain-track/files/" + date_path + "-" + timestamp + "-centreofmass.txt"

    # with open(file, 'w') as f:
    #     f.write('Centre-of-Mass Coordinates')

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # #  SETTING UP LOCALISER JUST ONCE # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #

    slice_pos = 0
    min_slice_pos = 0
    first_slice = 1

    for acquisition in connection:
        # print(acquisition)
        imag = np.abs(acquisition.data.transpose(3, 2, 1, 0))
        print("Slice Dimensions ", imag.shape)

        ndim = imag.shape
        # print("ndim ", ndim)

        position = acquisition.position
        position = position[0], position[1], position[2]
        slice_dir = acquisition.slice_dir
        slice_dir = slice_dir[0], slice_dir[1], slice_dir[2]
        phase_dir = acquisition.phase_dir
        phase_dir = phase_dir[0], phase_dir[1], phase_dir[2]
        read_dir = acquisition.read_dir
        read_dir = read_dir[0], read_dir[1], read_dir[2]
        patient_table_position = (acquisition.patient_table_position[0], acquisition.patient_table_position[1],
                                  acquisition.patient_table_position[2])
        print("position ", position, "read_dir", read_dir, "phase_dir ", phase_dir, "slice_dir ", slice_dir)
        print("patient table position", patient_table_position)

        if first_slice == 1:
            min_slice_pos = position[1]
            first_slice = 0

        else:
            if position[1] < min_slice_pos:
                min_slice_pos = position[1]

        slice_pos += position[1]
        # pos_z = patient_table_position[2] + position[2]
        pos_z = position[2]
        print("accumulated slice pos", slice_pos)
        print("accumulated position", position[1])
        print("pos_z", pos_z)

        # Get crop image, flip and rotate to match with true Nifti image
        img = imag[:, :, :, 0]

        # Stuff into the buffer
        slice = acquisition.slice
        repetition = acquisition.repetition
        contrast = acquisition.contrast
        print("Repetition ", repetition, "Slice ", slice, "Contrast ", contrast)

        sform_x = acquisition.read_dir
        sform_y = acquisition.phase_dir
        sform_z = acquisition.slice_dir
        position = acquisition.position

        srow_x = (sform_x[0], sform_x[1], sform_x[2])
        srow_y = (sform_y[0], sform_y[1], sform_y[2])
        srow_z = (sform_z[0], sform_z[1], sform_z[2])

        # srow_x = np.round(srow_x).astype(int)
        # srow_y = np.round(srow_y).astype(int)
        # srow_z = np.round(srow_z).astype(int)

        srow_x = (np.round(srow_x, 3))
        srow_y = (np.round(srow_y, 3))
        srow_z = (np.round(srow_z, 3))

        srow_x = (srow_x[0], srow_x[1], srow_x[2])
        srow_y = (srow_y[0], srow_y[1], srow_y[2])
        srow_z = (srow_z[0], srow_z[1], srow_z[2])

        logging.info("Storing each slice into the 3D data buffer...")
        if contrast == 1:
            im[:, :, slice] = np.squeeze(img[:, :, 0])

        # rotx = acquisition.user_float[0]
        # roty = acquisition.user_float[1]
        # rotz = acquisition.user_float[2]
        #
        # cmx = acquisition.user_float[3]
        # cmy = acquisition.user_float[4]
        # cmz = acquisition.user_float[5]

        # for multi-echo acquisitions
        if contrast == 1:  # SNS changing this for testing!
            # if the whole stack of slices has been acquired >> apply network to the entire 3D volume
            if slice == nslices - 1:  # if last slice & second echo-time
                logging.info("All slices stored into the data buffer!")
                print("ECHO", contrast)
                # if nslices % 2 != 0:
                #     mid = int(nslices / 2) + 1
                # else:
                #     mid = int(nslices / 2)
                # print("This is the mid slice: ", mid)
                # im_corr2a = im[:, :, 0:mid]
                # im_corr2b = im[:, :, mid:]
                #
                # im_corr2ab = np.zeros(np.shape(im), dtype='complex_')
                #
                # im_corr2ab[:, :, ::2] = im_corr2a
                # im_corr2ab[:, :, 1::2] = im_corr2b

                # logging.info(f"Python reconstruction done. Duration: {(time.time() - start):.2f} s")

                # for s in range(nslices):
                #     plt.imshow(np.squeeze(np.abs(im[:, :, s, 0])), cmap="gray")
                #     plt.show()

                # ==================================================================================================== #
                #
                #  TRAIN Localisation Network with 3D images
                #
                # ==================================================================================================== #

                print("..................................................................................")
                print("This is the echo-time we're looking at: ", contrast)

                logging.info("Initializing localization network...")

                N_epochs = 100
                I_size = 128
                N_classes = 2

                # # # Prepare arguments

                args = ArgumentsTrainTestLocalisation(epochs=N_epochs,
                                                      batch_size=2,
                                                      lr=0.002,
                                                      crop_height=I_size,
                                                      crop_width=I_size,
                                                      crop_depth=I_size,
                                                      validation_steps=8,
                                                      lamda=10,
                                                      training=False,
                                                      testing=False,
                                                      running=True,
                                                      root_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron'
                                                               '/python',
                                                      csv_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron'
                                                              '/python/files/',
                                                      checkpoint_dir='/home/sn21/miniconda3/envs/gadgetron/share'
                                                                     '/gadgetron/python/checkpoints/2022-12-16-newest/',
                                                      # change to -breech or -young if needed!
                                                      train_csv=
                                                      'data_localisation_1-label-brain_uterus_train-2022-11-23.csv',
                                                      valid_csv=
                                                      'data_localisation_1-label-brain_uterus_valid-2022-11-23.csv',
                                                      test_csv=
                                                      'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                                      run_csv=
                                                      'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                                      # run_input=im_corr2ab,
                                                      run_input=im,
                                                      results_dir='/home/sn21/miniconda3/envs/gadgetron/share'
                                                                  '/gadgetron/python/results/',
                                                      exp_name='Loc_3D',
                                                      task_net='unet_3D',
                                                      n_classes=N_classes)

                # path = "/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/2023-09-25/im.nii.gz"
                # image = sitk.GetImageFromArray(im)
                # sitk.WriteImage(image, path)

                # for acquisition in connection:

                args.gpu_ids = [0]

                # RUN with empty masks - to generate new ones (practical application)

                if args.running:
                    print("Running")
                    # print("im shape ", im_corr2ab.shape)
                    logging.info("Starting localization...")
                    model = md.LocalisationNetwork3DMultipleLabels(args)
                    # Run inference
                    ####################
                    model.run(args, 1)  # Changing this to 0 avoids the plotting
                    logging.info("Localization completed!")

                    rotx = 0.0
                    roty = 0.0
                    rotz = 0.0

                    logging.info("Storing motion parameters into variables...")
                    xcm = model.x_cm
                    ycm = model.y_cm
                    zcm = model.z_cm
                    logging.info("Motion parameters stored!")

                    print("centre-of-mass coordinates: ", xcm, ycm, zcm)
                    print("Localisation completed.")

                    if repetition == 0:
                        logging.info("Calculating translational motion in repetition 0...")
                        xfrep = xcm
                        yfrep = ycm
                        zfrep = zcm

                        tx_prev2 = 0.0
                        ty_prev2 = 0.0
                        tz_prev2 = 0.0

                        tx_prev = 0.0
                        ty_prev = 0.0
                        tz_prev = 0.0

                        x = 0.0
                        y = 0.0
                        z = 0.0

                        # xcm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = xcm_prev

                        xcm_prev = xcm
                        ycm_prev = ycm
                        zcm_prev = zcm

                        # IF THERE IS A DELAY OF TWO REPETITIONS
                        # cm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = xcm_prev

                        tx = 0.0
                        ty = 0.0
                        tz = 0.0

                        tx_prev = 0.0
                        ty_prev = 0.0
                        tz_prev = 0.0

                        # tx2 = 0.0
                        # ty2 = 0.0
                        # tz2 = 0.0

                        logging.info("Motion calculated!")
                        print("These are CoM coordinates for first repetition: ", xfrep, yfrep, zfrep)

                        print("..................................................................................")
                        print("Starting landmark detection...")

                        segmentation_volume = model.seg_pr
                        image_volume = model.img_gt

                        box, expansion_factor, center, offset, side_length, mask, vol, crop = (apply_bounding_box
                                                                                               (segmentation_volume,
                                                                                                image_volume))

                        box_path = args.results_dir + date_path
                        os.mkdir(box_path + "/" + timestamp + "-nnUNet_seg/")
                        os.mkdir(box_path + "/" + timestamp + "-nnUNet_pred/")

                        box_im = nib.Nifti1Image(box, np.eye(4))
                        nib.save(box_im, box_path + "/" + timestamp + "-nnUNet_seg/FreemaxLandmark_001_0000.nii.gz")
                        im_ = nib.Nifti1Image(im, np.eye(4))
                        path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "/"
                                + timestamp + "-gadgetron-fetal-brain-localisation-img_initial.nii.gz")
                        nib.save(im_, path)
                        # sitk.WriteImage(im, path)

                        # Run Prediction with nnUNet
                        # Set the DISPLAY and XAUTHORITY environment variables
                        os.environ['DISPLAY'] = ':1'  # Replace with your X11 display, e.g., ':1.0'
                        os.environ["XAUTHORITY"] = '/home/sn21/.Xauthority'

                        start_time = time.time()
                        terminal_command = (("export nnUNet_raw='/home/sn21/landmark-data/Landmarks/nnUNet_raw'; export"
                                             "nnUNet_preprocessed='/home/sn21/landmark-data/Landmarks"
                                             "/nnUNet_preprocessed' ; export "
                                             "nnUNet_results='/home/sn21/landmark-data/Landmarks/nnUNet_results' ; "
                                             "conda activate gadgetron ; nnUNetv2_predict -i ") + box_path + "/" +
                                            timestamp + "-nnUNet_seg/ -o " + box_path + "/" + timestamp +
                                            "-nnUNet_pred/ -d 080 -c 3d_fullres -f 1")

                        subprocess.run(terminal_command, shell=True)
                        # Record the end time
                        end_time = time.time()

                        # Calculate the elapsed time
                        elapsed_time = end_time - start_time
                        print(f"Elapsed Time for Landmark Detection: {elapsed_time} seconds")

                        # Define the path where NIfTI images are located
                        l_path = os.path.join(box_path, timestamp + "-nnUNet_pred")

                        # Use glob to find NIfTI files in the directory
                        landmarks_paths = glob.glob(os.path.join(l_path, "*.nii.gz"))
                        print(landmarks_paths)

                        for landmarks_path in landmarks_paths:
                            # Load the image using sitk.ReadImage
                            landmark = nib.load(landmarks_path)
                            # Get the image data as a NumPy array
                            landmark = landmark.get_fdata()
                            modified_landmark = np.copy(landmark)

                            # Replace label values 2 with 1
                            landmark[np.where(landmark == 2.0)] = 3.0
                            modified_landmark[np.where(modified_landmark == 2.0)] = 0.0

                            # Define a structuring element (a kernel) for morphological operations
                            structuring_element = np.ones((3, 3, 3))  # Adjust size if needed

                            # Perform morphological closing
                            closed_segmentation = ndimage.binary_closing(modified_landmark,
                                                                         structure=structuring_element).astype(
                                modified_landmark.dtype)

                            labelled_segmentation, num_features = ndimage.label(closed_segmentation)
                            print("number of features", num_features)
                            # region_sizes = np.bincount(labelled.ravel())[1:]

                            mask = (landmark == 3.0)
                            labelled_segmentation[mask] = 3.0

                            eye_1 = (labelled_segmentation == 1.0).astype(int)
                            eye_2 = (labelled_segmentation == 2.0).astype(int)
                            cereb = (labelled_segmentation == 3.0).astype(int)

                            # print("EYE 1", eye_1)

                            cm_eye_1 = np.array(ndimage.center_of_mass(eye_1))
                            cm_eye_2 = np.array(ndimage.center_of_mass(eye_2))
                            cm_cereb = np.array(ndimage.center_of_mass(cereb))

                            # cm_eye_1 = cm_eye_1[2], cm_eye_1[1], cm_eye_1[0]
                            # cm_eye_2 = cm_eye_2[2], cm_eye_2[1], cm_eye_2[0]
                            # cm_cereb = cm_cereb[2], cm_cereb[1], cm_cereb[0]
                            print("LANDMARKS", cm_eye_1, cm_eye_2, cm_cereb)

                            # Extract coordinates of non-zero points in the mask
                            points = np.argwhere(segmentation_volume)

                            # Calculate the centroid (you can use a different reference point if needed)
                            centroid = xcm, ycm, zcm

                            # Calculate Euclidean distances from each point to the centroid
                            distances = [euclidean(point, centroid) for point in points]

                            # Find the index of the point with the maximum distance
                            furthest_point_index = np.argmax(distances)

                            # Retrieve the coordinates of the furthest point
                            furthest_point = points[furthest_point_index]
                            print("Furthest Point Coordinates:", furthest_point)

                            # Get the current date and time
                            current_datetime = datetime.now()

                            # Format the date and time as a string
                            date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

                            # Define the file name with the formatted date and time
                            text_file_1 = args.results_dir + date_path + "/" + timestamp + "-nnUNet_pred/" + "com.txt"
                            text_file = "/home/sn21/freemax-transfer/Sara/landmarks-interface-autoplan/sara.dvs"

                            cm_brain = model.x_cm, model.y_cm, model.z_cm

                            # Calculate the scaling factor
                            scaling_factor = expansion_factor
                            original_dimensions = vol.shape
                            scaled_dimensions = crop.shape
                            print("original_dimensions", original_dimensions, "scaled_dimensions", scaled_dimensions)

                            # Calculate the corresponding coordinates in the scaled (34x34x34) image
                            cropped_eye_1_x = cm_eye_1[0] / scaling_factor
                            cropped_eye_1_y = cm_eye_1[1] / scaling_factor
                            cropped_eye_1_z = cm_eye_1[2] / scaling_factor

                            cropped_eye_2_x = cm_eye_2[0] / scaling_factor
                            cropped_eye_2_y = cm_eye_2[1] / scaling_factor
                            cropped_eye_2_z = cm_eye_2[2] / scaling_factor

                            cropped_cereb_x = cm_cereb[0] / scaling_factor
                            cropped_cereb_y = cm_cereb[1] / scaling_factor
                            cropped_cereb_z = cm_cereb[2] / scaling_factor

                            # Print the corresponding coordinates in the scaled image
                            print("Coordinates in the scaled image: ({:.2f}, {:.2f}, {:.2f})".format(cropped_eye_1_x,
                                                                                                     cropped_eye_1_y,
                                                                                                     cropped_eye_1_z))

                            # Calculate the center of mass in the original 128x128x128 matrix
                            cm_eye_1 = (
                                int(center[0] - side_length // 2) + cropped_eye_1_x,
                                int(center[1] - side_length // 2) + cropped_eye_1_y,
                                int(center[2] - side_length // 2) + cropped_eye_1_z
                            )
                            cm_eye_2 = (
                                int(center[0] - side_length // 2) + cropped_eye_2_x,
                                int(center[1] - side_length // 2) + cropped_eye_2_y,
                                int(center[2] - side_length // 2) + cropped_eye_2_z
                            )
                            cm_cereb = (
                                int(center[0] - side_length // 2) + cropped_cereb_x,
                                int(center[1] - side_length // 2) + cropped_cereb_y,
                                int(center[2] - side_length // 2) + cropped_cereb_z
                            )

                            # Dimensions of the padded 128x128x128 image
                            padded_dimensions = vol.shape

                            # Original dimensions of the image (128x128x60)
                            original_dimensions = im.shape

                            # Calculate the padding in each dimension
                            padding = ((padded_dimensions[0] - original_dimensions[0]) // 2,
                                       (padded_dimensions[1] - original_dimensions[1]) // 2,
                                       (padded_dimensions[2] - original_dimensions[2]) // 2)

                            # Calculate equivalent coordinates in the original 128x128x60 image
                            cm_eye_1 = (
                                cm_eye_1[0] - padding[0],
                                cm_eye_1[1] - padding[1],
                                cm_eye_1[2] - padding[2]
                            )

                            cm_eye_2 = (
                                cm_eye_2[0] - padding[0],
                                cm_eye_2[1] - padding[1],
                                cm_eye_2[2] - padding[2]
                            )

                            cm_cereb = (
                                cm_cereb[0] - padding[0],
                                cm_cereb[1] - padding[1],
                                cm_cereb[2] - padding[2]
                            )

                            cm_brain = (
                                cm_brain[0] - padding[0],
                                cm_brain[1] - padding[1],
                                cm_brain[2] - padding[2]
                            )

                            furthest_point = (
                                furthest_point[0] - padding[0],
                                furthest_point[1] - padding[1],
                                furthest_point[2] - padding[2]
                            )

                            print("EYE 1", cm_eye_1)
                            print("EYE 2", cm_eye_2)
                            print("CEREB", cm_cereb)
                            print("BRAIN", cm_brain)
                            print("FURTHEST BRAIN VOXEL", furthest_point)

                            cm_eye_1 = pixdim_x * cm_eye_1[0], pixdim_y * cm_eye_1[1], pixdim_z * cm_eye_1[2]
                            cm_eye_2 = pixdim_x * cm_eye_2[0], pixdim_y * cm_eye_2[1], pixdim_z * cm_eye_2[2]
                            cm_cereb = pixdim_x * cm_cereb[0], pixdim_y * cm_cereb[1], pixdim_z * cm_cereb[2]
                            cm_brain = pixdim_x * cm_brain[0], pixdim_y * cm_brain[1], pixdim_z * cm_brain[2]
                            furthest_point = (pixdim_x * furthest_point[0], pixdim_y * furthest_point[1],
                                              pixdim_z * furthest_point[2])

                            # # # # # # # # # # # # # # # # # # # # # # # # # #
                            # position = np.array(position)
                            # position = [position[0], position[1], position[2]]
                            pos = slice_pos / (nslices * ncontrasts)  # slice position mid volume
                            print("POS", pos)
                            print("slice_pos", slice_pos)
                            print("nslices", nslices)
                            position = (position[0], pos, position[2])

                            # lowerleftcorner = ((np.int(enc.encodedSpace.fieldOfView_mm.x/2),
                            #                     np.int(enc.encodedSpace.fieldOfView_mm.y/2), np.int(min_slice_pos)))
                            centreofimageposition = ((np.float64(enc.encodedSpace.fieldOfView_mm.x) / 2,
                                                      np.float64(enc.encodedSpace.fieldOfView_mm.y) / 2,
                                                      np.float64(nslices * pixdim_z) / 2))
                            print("centreofimageposition", centreofimageposition)
                            
                            # position = np.round(position).astype(int)
                            position = (position[0], position[1], position[2])
                            # position = - position[1], position[2], position[0]
                            cm_eye_1 = np.round(cm_eye_1, 3)
                            cm_eye_2 = np.round(cm_eye_2, 3)
                            cm_cereb = np.round(cm_cereb, 3)
                            cm_brain = np.round(cm_brain, 3)
                            furthest_point = np.round(furthest_point, 3)

                            print("POSITION MM", position)
                            print("EYE 1 MM", cm_eye_1)
                            print("EYE 2 MM", cm_eye_2)
                            print("CEREB MM", cm_cereb)
                            print("BRAIN MM", cm_brain)
                            print("FURTHEST BRAIN VOXEL MM", furthest_point)

                            cm_brain = (cm_brain[0] - centreofimageposition[0],
                                        cm_brain[1] - centreofimageposition[1],
                                        cm_brain[2] - centreofimageposition[2])

                            furthest_point = (furthest_point[0] - centreofimageposition[0],
                                              furthest_point[1] - centreofimageposition[1],
                                              furthest_point[2] - centreofimageposition[2])

                            cm_eye_1 = ((cm_eye_1[0]) - centreofimageposition[0],
                                        (cm_eye_1[1]) - centreofimageposition[1],
                                        cm_eye_1[2] - centreofimageposition[2])

                            cm_eye_2 = ((cm_eye_2[0]) - centreofimageposition[0],
                                        (cm_eye_2[1]) - centreofimageposition[1],
                                        cm_eye_2[2] - centreofimageposition[2])

                            cm_cereb = ((cm_cereb[0]) - centreofimageposition[0],
                                        cm_cereb[1] - centreofimageposition[1],
                                        cm_cereb[2] - centreofimageposition[2])

                            print("centreofimageposition", centreofimageposition)
                            print("EYE 1 OFFSET", cm_eye_1)
                            print("EYE 2 OFFSET", cm_eye_2)
                            print("CEREB OFFSET", cm_cereb)
                            print("BRAIN OFFSET", cm_brain)
                            print("FURTHEST BRAIN VOXEL OFFSET", furthest_point)

                            # x = -ty  # -ty # seems to work
                            # y = tz  # tz
                            # z = tx  # tx  # seems to work

                            cm_eye_1 = (-cm_eye_1[1], cm_eye_1[2], cm_eye_1[0])
                            cm_eye_2 = (-cm_eye_2[1], cm_eye_2[2], cm_eye_2[0])
                            cm_cereb = (-cm_cereb[1], cm_cereb[2], cm_cereb[0])
                            cm_brain = (-cm_brain[1], cm_brain[2], cm_brain[0])
                            furthest_point = (-furthest_point[1], furthest_point[2], furthest_point[0])

                            cm_brain = (cm_brain[0] + position[0],
                                        cm_brain[1] + position[1],
                                        cm_brain[2] + position[2])

                            furthest_point = (furthest_point[0] + position[0],
                                              furthest_point[1] + position[1],
                                              furthest_point[2] + position[2])

                            cm_eye_1 = ((cm_eye_1[0]) + position[0],
                                        (cm_eye_1[1]) + position[1],
                                        cm_eye_1[2] + position[2])

                            cm_eye_2 = ((cm_eye_2[0]) + position[0],
                                        (cm_eye_2[1]) + position[1],
                                        cm_eye_2[2] + position[2])

                            cm_cereb = ((cm_cereb[0]) + position[0],
                                        cm_cereb[1] + position[1],
                                        cm_cereb[2] + position[2])

                            # print("EYE 1 ROT", cm_eye_1)
                            # print("EYE 2 ROT", cm_eye_2)
                            # print("CEREB ROT", cm_cereb)
                            # print("BRAIN ROT", cm_brain)
                            # print("FURTHEST BRAIN VOXEL ROT", furthest_point)

                            # Find the indices where cm_cereb is NaN
                            idx_eye_1 = np.isnan(cm_eye_1)
                            # Use numpy.where to replace NaN values with corresponding values from cm_brain
                            cm_eye_1 = np.where(idx_eye_1, (cm_brain[0], cm_brain[1], cm_brain[2]), cm_eye_1)

                            # Find the indices where cm_cereb is NaN
                            idx_eye_2 = np.isnan(cm_eye_2)
                            # Use numpy.where to replace NaN values with corresponding values from cm_brain
                            cm_eye_2 = np.where(idx_eye_2, (cm_brain[0], cm_brain[1], cm_brain[2]), cm_eye_2)

                            # Find the indices where cm_cereb is NaN
                            idx_cereb = np.isnan(cm_cereb)
                            # Use numpy.where to replace NaN values with corresponding values from cm_brain
                            cm_cereb = np.where(idx_cereb, (cm_brain[0], cm_brain[1], cm_brain[2]), cm_cereb)

                            cm_eye_1 = (cm_eye_1[0], cm_eye_1[1], cm_eye_1[2])
                            cm_eye_2 = (cm_eye_2[0], cm_eye_2[1], cm_eye_2[2])
                            cm_cereb = (cm_cereb[0], cm_cereb[1], cm_cereb[2])

                            print("EYE 1 ROT", cm_eye_1)
                            print("EYE 2 ROT", cm_eye_2)
                            print("CEREB ROT", cm_cereb)
                            print("BRAIN ROT", cm_brain)
                            print("FURTHEST BRAIN VOXEL ROT", furthest_point)

                            # transformation = [(srow_x[0], srow_x[1], srow_x[2], srow_x[3]),
                            #                   (srow_y[0], srow_y[1], srow_y[2], srow_y[3]),
                            #                   (srow_z[0], srow_z[1], srow_z[2], srow_z[3]),
                            #                   (0, 0, 0, 1)]

                            # Create and write to the text file
                            with open(text_file, "w") as file:
                                # file.write("This is a text file created on " + date_time_string)
                                # file.write("\n" + str('CoM: '))
                                file.write("eye1 = " + str(cm_eye_1))
                                file.write("\n" + "eye2 = " + str(cm_eye_2))
                                file.write("\n" + "cere = " + str(cm_cereb))
                                file.write("\n" + "brain = " + str(cm_brain))
                                file.write("\n" + "furthest = " + str(furthest_point))
                                file.write("\n" + "position = " + str(position))
                                # file.write("\n" + "centreofimageposition = " + str(centreofimageposition))
                                file.write("\n" + "srow_x = " + str(srow_x))
                                file.write("\n" + "srow_y = " + str(srow_y))
                                file.write("\n" + "srow_z = " + str(srow_z))

                            with open(text_file_1, "w") as file:
                                # file.write("This is a text file created on " + date_time_string)
                                # file.write("\n" + str('CoM: '))
                                file.write("eye1 = " + str(cm_eye_1))
                                file.write("\n" + "eye2 = " + str(cm_eye_2))
                                file.write("\n" + "cere = " + str(cm_cereb))
                                file.write("\n" + "brain = " + str(cm_brain))
                                file.write("\n" + "furthest = " + str(furthest_point))
                                file.write("\n" + "position = " + str(position))
                                file.write("\n" + "srow_x = " + str(srow_x))
                                file.write("\n" + "srow_y = " + str(srow_y))
                                file.write("\n" + "srow_z = " + str(srow_z))

                            print(f"Text file '{text_file}' has been created.")

                            # Convert the modified NumPy array back to a SimpleITK image
                            # modified_image = sitk.GetImageFromArray(labelled_segmentation)
                            # modified_image = labelled_segmentation

                            # Save the modified image with the same file path
                            output_image_path = landmarks_path.replace(".nii.gz", "_3-labels.nii.gz")

                            # sitk.WriteImage(modified_image, output_image_path)
                            modified_image = nib.Nifti1Image(labelled_segmentation, np.eye(4))
                            nib.save(modified_image, output_image_path)

                    else:  # if it's not the first repetition
                        logging.info("Calculating motion parameters in current repetition...")
                        print("xfrep yfrep zfrep", xfrep, yfrep, zfrep)
                        # tx = (xcm - xcm_prev) * 3.0 + tx_prev
                        # ty = (ycm - ycm_prev) * 3.0 + ty_prev
                        # tz = (zcm - zcm_prev) * 3.0 + tz_prev

                        # xcm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = xcm_prev

                        # xcm_prev = xcm
                        # ycm_prev = ycm
                        # zcm_prev = zcm

                        tx = (xcm - xcm_prev) * 3.0 + tx_prev2
                        ty = (ycm - ycm_prev) * 3.0 + ty_prev2
                        tz = (zcm - zcm_prev) * 3.0 + tz_prev2

                        print("xcm xcm_prev tx_prev tx_prev2 tx: ", xcm, xcm_prev, tx_prev, tx_prev2, tx)
                        print("ycm ycm_prev ty_prev ty_prev2 ty: ", ycm, ycm_prev, ty_prev, ty_prev2, ty)
                        print("zcm zcm_prev tz_prev tz_prev2 tz: ", zcm, zcm_prev, tz_prev, tz_prev2, tz)

                        xcm_prev2 = xcm_prev
                        ycm_prev2 = ycm_prev
                        zcm_prev2 = xcm_prev

                        xcm_prev = xcm
                        ycm_prev = ycm
                        zcm_prev = zcm

                        # IF THERE IS A DELAY OF TWO REPETITIONS
                        # tx2 = (xcm - xcm_prev2) * 3.0
                        # ty2 = (ycm - ycm_prev2) * 3.0
                        # tz2 = (zcm - zcm_prev2) * 3.0

                        # print("xcm xcm_prev tx_prev tx_prev2 tx: ", xcm, xcm_prev, tx_prev, tx_prev2, tx)
                        # print("ycm ycm_prev ty_prev ty_prev2 ty: ", ycm, ycm_prev, ty_prev, ty_prev2, ty)
                        # print("zcm zcm_prev tz_prev tz_prev2 tz: ", zcm, zcm_prev, tz_prev, tz_prev2, tz)

                        # tx = 0.0
                        # ty = 0.0
                        # tz = 0.0

                        print("I am slice ", slice, " and I am applying shift ", tx, ty, tz)
                        print("CoM coordinates for the repetition: ", xcm, ycm, zcm)
                        print("Translational motion for the repetition: ", tx, ty, tz)

                        tx_prev2 = tx_prev
                        ty_prev2 = ty_prev
                        tz_prev2 = tz_prev

                        tx_prev = tx
                        ty_prev = ty
                        tz_prev = tz

                        # tx_prev2 = tx_prev
                        # ty_prev2 = ty_prev
                        # tz_prev2 = tz_prev

                        xcm_prev = xcm
                        ycm_prev = ycm
                        zcm_prev = zcm

                        # xcm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = zcm_prev

                        x = -ty  # -ty
                        y = tz  # tz
                        z = tx  # tx

                        # IF THERE IS A DELAY OF TWO REPETITIONS
                        # x = -ty2
                        # y = tx2
                        # z = tz2

                        logging.info("Motion calculated!")
                        print("here's x y and z", x, y, z)

                    send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, x, y, z, acquisition)
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)

                # x = xcm
                # y = ycm
                # z = zcm

                # calculated motion parameters to be passed onto all slices but last of the next repetition
                x = -ty  # -ty # seems to work
                y = tz  # tz
                z = tx  # tx  # seems to work

                # if repetition == 0:
                #     xfrep = xcm
                #     yfrep = ycm
                #     zfrep = zcm
                #
                #     print("These are CoM coordinates for first repetition: ", xfrep, yfrep, zfrep)

            else:  # if it's not the last slice of the repetition (still in TE2!)
                # send_reconstructed_images(connection, im_corr2ab, acquisition)
                if repetition == 0:
                    rotx = 0.0
                    roty = 0.0
                    rotz = 0.0

                    xcm = 0.0
                    ycm = 0.0
                    zcm = 0.0

                    send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)
                    # del rotx, roty, rotz, xcm, ycm, zcm

                else:
                    rotx = 0.0
                    roty = 0.0
                    rotz = 0.0

                    tx_ = x
                    ty_ = y
                    tz_ = z

                    # print("I am slice ", slice, " and I am applying shift ", tx_, ty_, tz_)

                    send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, tx_, ty_, tz_, acquisition)
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)
                    # del rotx, roty, rotz, xcm, ycm, zcm

                continue

            # calculated motion parameters to be passed onto all slices but last of the next repetition
            x = -ty  # -ty # seems to work
            y = tz  # tz
            z = tx  # tx  # seems to work

        else:  # for all other echo-times but the second echo-time!
            if repetition == 0:  # for the first repetition of all TEs but TE2
                rotx = 0.0
                roty = 0.0
                rotz = 0.0

                xcm = 0.0
                ycm = 0.0
                zcm = 0.0

                send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)
                # motion(rotx, roty, rotz, xcm, ycm, zcm)
                # send_reconstructed_images(connection, imag, acquisition)
                # del rotx, roty, rotz, xcm, ycm, zcm

            else:  # all repetitions but first for all echo-times but TE2
                rotx = 0.0
                roty = 0.0
                rotz = 0.0

                tx_ = x
                ty_ = y
                tz_ = z

                # print("I am slice ", slice, " and I am applying shift ", tx_, ty_, tz_)

                send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, tx_, ty_, tz_, acquisition)
                # motion(rotx, roty, rotz, xcm, ycm, zcm)
                # send_reconstructed_images(connection, imag, acquisition)
                # del rotx, roty, rotz, xcm, ycm, zcm

            continue

# # # # # # # # # # # # # # # # # # # # # # # # #
