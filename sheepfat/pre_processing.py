#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: J-P Praetorius
@email: jan-philipp.praetorius@leibniz-hki.de or p.e.mueller07@gmail.com

Copyright by Jan-Philipp Praetorius

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology -
Hans Knöll Insitute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany
"""


from glob import glob
import os
import numpy as np
from scipy import ndimage
import math
import matplotlib.pyplot as plt
import tifffile as tif
from skimage import io, morphology, img_as_ubyte, img_as_float32
import cv2


def plot(img, img_binary):
    """
    Plot the original image with drawn contours based on the binary annotations.

    Args:
        img ([type]): original image
        img_binary ([type]): binary image
    """
    
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 15))
    
    ax1.imshow(img)
    ax1.set_title("Original image with drawn contour of annotations", fontsize=20)
    ax1.grid(b=True, which='major', color='red', linewidth=1)
    plt.minorticks_on()
    ax1.grid(b=True, which='minor', color='green', linewidth=1)
    ax1.contour(img_binary, colors='green', linewidths=0.3)

    plt.tight_layout()    
    plt.show()


def plot_window(img, img_binary, title=None):
    """
    Plot the cut-out original image with drawn contours based on the binary annotations.

    Args:
        img ([type]): original image
        img_binary ([type]): binary image
        title: general title for the figure
    """
    
    fig, ax_arr = plt.subplots(1, 2, figsize=(18, 8))    
    ax1, ax2 = ax_arr.ravel()

    if len(img.shape) == 3:
        ax1.imshow(img)
    else:
        ax1.imshow(img, cmap='gray')
    ax1.set_title("Original image with drawn contour of annotation", fontsize=18)
    ax1.contour(img_binary, colors='green', linewidths=2)

    ax2.imshow(img_binary, cmap='gray')
    ax2.set_title("Annotations image", fontsize=20)

    if title is not None:
        fig.suptitle('{}'.format(title), fontsize=30)

    plt.tight_layout()    
    plt.show()



def sliding_window(img, step_size=(256, 256), window_size=(256, 256)):
    """
    Slide over the specified input image
    Args:
        img: the input image
        step_size: the step size (x1, x0)
        window_size: the window size (width, height)

    Returns: Generator of (x0, x1, window)
    """

    for x0 in range(0, img.shape[0], step_size[0]):
        for x1 in range(0, img.shape[1]+1, step_size[1]):
            yield x0, x1, img[x0:x0 + window_size[1], x1:x1 + window_size[0]]


def get_contours(img, linewidth):
    """
    Draw the outline contour of in the underlying binary image

    Args:
        img ([type]): [description]
        linewidth ([type]): [description]

    Returns:
        [type]: [description]
    """

    tmp = morphology.dilation(img, selem=morphology.disk(2))

    _, contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros_like(img)

    cv2.drawContours(mask, contours, -1, 255, thickness=linewidth)

    return mask
    


def create_subwindows(path_original, path_binary, output_path, window_size, img_original=None, img_binary=None, verbose=0, var=100):
    """[summary]

    Args:
        path_original ([type]): path for input image
        path_binary ([type]): path for binary image
        output_path ([type]): save directory path
        window_size ([type]): windows size for each subwindow
        img_original ([type], optional): if already available - input image. Defaults to None.
        img_binary ([type], optional): if already available - binary image. Defaults to None.
        verbose (int, optional): verbose level. Defaults to 0.
    """

    # read original and binary image if necessary
    if img_original is None:
        try:
            img_original = tif.imread(path_original)
        except:
            img_original = io.imread(path_original)
        print(f'read image with shape {img_original.shape} from {path_original}')

    if img_binary is None:
        try:
            img_binary = cv2.imread(path_binary, 0)
        except:
            img_binary = io.imread(path_binary)
        print(f'read image with shape {img_binary.shape} from {path_binary}')

    assert img_original.shape[:2] == img_binary.shape[:2], "Caution both images do not have the same image size" 
 
    # create a padding at the right and bottom for original and label image depending on the window size (if necessary)
    img_x0 = img_original.shape[0]
    img_x1 = img_original.shape[1]
  
    if img_x0 % window_size != 0 or img_x1 % window_size != 0:
        
        pad_width_2c = [(0, int(math.ceil(1.0 * img_x0 / window_size) * window_size) - img_x0),
                        (0, int(math.ceil(1.0 * img_x1 / window_size) * window_size) - img_x1)]
        pad_width_3c = [(0, int(math.ceil(1.0 * img_x0 / window_size) * window_size) - img_x0),
                    (0, int(math.ceil(1.0 * img_x1 / window_size) * window_size) - img_x1),
                    (0,0)]

        print('\nPadding is required with values:', pad_width_3c)
    
        if len(img_original.shape) == 2:
            img_original = np.pad(img_original, pad_width=pad_width_2c, mode='constant', constant_values=0)
        else:
            img_original = np.pad(img_original, pad_width=pad_width_3c, mode='constant', constant_values=0)
        
        if len(img_binary.shape) == 2:
            img_binary = np.pad(img_binary, pad_width=pad_width_2c, mode='constant', constant_values=0)
        else:
            img_binary = np.pad(img_binary, pad_width=pad_width_3c, mode='constant', constant_values=0)
        
        print('new image size for original image after performing padding:\t', img_original.shape)
        print('new image size for label image after performing padding:\t', img_binary.shape)
    
    # create output folders for the subwindows
    if not os.path.exists( os.path.join(output_path, 'original') ):
        os.makedirs(os.path.join(output_path, 'original'))
        print('\ncreate output_path:', os.path.join(output_path, 'original'))
    if not os.path.exists( os.path.join(output_path, 'labels') ):
        os.makedirs(os.path.join(output_path, 'labels'))
        print('create output_path:', os.path.join(output_path, 'labels'))

    first_shape = None

    # get the starting index for the stored filenames by get the last saved index in output directory + 1
    if len(glob(os.path.join(output_path, 'original', '*'))) == 0:
        start_idx = 0
    else:
        all_filenames = glob(os.path.join(output_path, 'original', '*'))
        all_filenames = [os.path.basename(tmp).split('_')[0] for tmp in all_filenames]
        start_idx = np.sort([int(tmp) for tmp in all_filenames])[-1] + 1
    print('starting index:', start_idx)

    counter_label, counter_total = 0, 0

    # start sliding-window-operations in subwindow-size (e.g. 256²)
    for idx, (x, y, window) in enumerate(sliding_window(img_original,
                                                        step_size=(window_size, window_size),
                                                        window_size=(window_size, window_size))):
        
        # get the first shape to compare this with all the others
        if counter_total == 0:
            first_shape = window.shape
            print('The first-shape:', first_shape)       
        
        counter_total += 1
        if verbose > 0:
            print(f'[num_label / num_total] -> [{counter_label} / {counter_total}] - (x,y) -> ({x}/{y})')

        # get label image by extracting binary image at the same location
        y1, y2 = y, y + window.shape[0]
        x1, x2 = x, x + window.shape[1]
        img_window_binary = img_as_ubyte(img_binary[ y1:y2 , x1:x2 ])

        # remove small occuring hole-artefacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        img_window_binary = morphology.binary_closing(img_window_binary, selem=kernel)
        img_window_binary = morphology.remove_small_holes(img_window_binary, area_threshold=270)
        
        # remove small occuring object-artefacts which are note desired
        img_window_binary = morphology.remove_small_holes(img_window_binary, area_threshold=50)
       
        img_window_binary = img_as_ubyte(img_window_binary)

        # get contoures instead of entire ROIs
        #img_window_binary = get_contours(img_window_binary, linewidth=3)
        
        ################################################################################################
        # CRITERION(s) to skip/select cut-out image (e.g. check if there is a manual annotation image) #
        ################################################################################################

        ### skip invalid regions at the border
        if window.shape[:2] != img_window_binary.shape:
            print('invalid shape:', window.shape[:2], img_window_binary)
            continue

        ### just take cut-out images which consists of more than 20% white surface in the original image
        # unq, counts = np.unique(window, return_counts=True)
        # n_values = np.sum(counts)
        # idx_val = list(unq).index(np.max(window))
        # n_maxVal = counts[idx_val]
        # maxVal_fraction = n_maxVal/n_values

        #if len(np.unique(window)) == 1 or maxVal_fraction >= 0.2:
            #print('skip (x,y):', x, y, ' --- id:', idx, '--- more than 20% white region (most likely background), fraction of white pixels:', maxVal_fraction)
            #continue

        ### only take cut-out images -WITH- annotations
        if np.count_nonzero(img_window_binary) == 0:
            # print('skip (x,y):', x, y, ' --- no fungi: -id:', idx)
            continue
            
        ### only take cut-out images -WITHOUT- annotations
        # if np.count_nonzero(img_window_binary) != 0:
        #     # var_img = ndimage.variance(img_window_binary)
        #     continue
        
        # var_img = ndimage.variance(window)
        # print('variance:', var_img)

        # if var_img < var or math.isnan(var_img):
        #     continue

        ### discard cut-out images -WITHOUT- annotations with a likelihood of 90%
        #if np.count_nonzero(annot_window) == 0 and np.random.random() < 0.9:
        #    continue        

        # extract the original cut-out
        img_window_original = img_original[ y1:y2 , x1:x2 ]

        # save original image
        filename_original = '{0}_{1}'.format(str(start_idx+counter_label).zfill(4), os.path.basename(path_original))
        save_path_original = os.path.join(output_path, 'original', filename_original)
        
        # save label image
        filename_label = '{0}_{1}'.format(str(start_idx+counter_label).zfill(4), os.path.basename(path_binary))
        save_path_label = os.path.join(output_path, 'labels', filename_label)
                
        io.imsave(save_path_original, img_window_original)
        io.imsave(save_path_label, img_window_binary)
        print(f'save original-image to:\t {filename_original} with size: {img_window_original.shape} and type: {img_window_original.dtype}')
        print(f'save labeled-image to:\t {filename_label} with size: {img_window_binary.shape} and type: {img_window_binary.dtype} and unique values: {np.unique(img_window_binary)}')

        counter_label += 1

        if verbose > 1:
            plot_window(img_window_original, img_window_binary)
        
    print('\n---> number of cut-out-images with labels: {0} / {1}'.format(counter_label, counter_total))


def create_subwindows_from_union_samples(path_original, path_binary1, path_binary2, output_path, window_size, verbose=0):
    """
    extract labels from 2 samples

    Args:
        path_original ([type]): path for input image
        path_binary1 ([type]): path for first binary image
        path_binary2 ([type]): path for second binary image
        output_path ([type]): save directory path
        window_size ([type]): windows size for each subwindow
        img_original ([type], optional): if already available - input image. Defaults to None.
        img_binary ([type], optional): if already available - binary image. Defaults to None.
        verbose (int, optional): verbose level. Defaults to 0.
    """

    # read original and the 2 binary images if necessary
    try:
        img_original = tif.imread(path_original)
    except:
        img_original = io.imread(path_original)
    print(f'read image with shape {img_original.shape} from {path_original}')

    try:
        img_binary1 = cv2.imread(path_binary1, 0)
    except:
        img_binary1 = io.imread(path_binary1)
    print(f'read image with shape {img_binary1.shape} from {path_binary1}')
    assert img_original.shape[:2] == img_binary1.shape[:2], "Caution original and binary image 1 do not have the same image size" 
 
    try:
        img_binary2 = cv2.imread(path_binary2, 0)
    except:
        img_binary2 = io.imread(path_binary2)
    print(f'read image with shape {img_binary2.shape} from {path_binary2}')
    assert img_original.shape[:2] == img_binary2.shape[:2], "Caution original and binary image 2 do not have the same image size" 
     
    # create a padding at the right and bottom for original and label image depending on the window size (if necessary)
    img_x0 = img_original.shape[0]
    img_x1 = img_original.shape[1]
  
    if img_x0 % window_size != 0 or img_x1 % window_size != 0:
        
        pad_width_2c = [(0, int(math.ceil(1.0 * img_x0 / window_size) * window_size) - img_x0),
                        (0, int(math.ceil(1.0 * img_x1 / window_size) * window_size) - img_x1)]
        pad_width_3c = [(0, int(math.ceil(1.0 * img_x0 / window_size) * window_size) - img_x0),
                    (0, int(math.ceil(1.0 * img_x1 / window_size) * window_size) - img_x1),
                    (0,0)]

        print('\nPadding is required with values:', pad_width_3c)
    
        if len(img_original.shape) == 2:
            img_original = np.pad(img_original, pad_width=pad_width_2c, mode='constant', constant_values=0)
        else:
            img_original = np.pad(img_original, pad_width=pad_width_3c, mode='constant', constant_values=0)
        
        if len(img_binary1.shape) == 2:
            img_binary1 = np.pad(img_binary1, pad_width=pad_width_2c, mode='constant', constant_values=0)
        else:
            img_binary1 = np.pad(img_binary1, pad_width=pad_width_3c, mode='constant', constant_values=0)
        
        if len(img_binary2.shape) == 2:
            img_binary2 = np.pad(img_binary2, pad_width=pad_width_2c, mode='constant', constant_values=0)
        else:
            img_binary2 = np.pad(img_binary2, pad_width=pad_width_3c, mode='constant', constant_values=0)
        
        print('new image size for original image after performing padding:\t', img_original.shape)
        print('new image size for label image <1> after performing padding:\t', img_binary1.shape)
        print('new image size for label image <2> after performing padding:\t', img_binary2.shape)
    
    # create output folders for the subwindows
    if not os.path.exists( os.path.join(output_path, 'original') ):
        os.makedirs(os.path.join(output_path, 'original'))
        print('\ncreate output_path:', os.path.join(output_path, 'original'))
    if not os.path.exists( os.path.join(output_path, 'labels') ):
        os.makedirs(os.path.join(output_path, 'labels'))
        print('create output_path:', os.path.join(output_path, 'labels'))

    first_shape = None

    # get the starting index for the stored filenames by get the last saved index in output directory + 1
    if len(glob(os.path.join(output_path, 'original', '*'))) == 0:
        start_idx = 0
    else:
        all_filenames = glob(os.path.join(output_path, 'original', '*'))
        all_filenames = [os.path.basename(tmp).split('_')[0] for tmp in all_filenames]
        start_idx = np.sort([int(tmp) for tmp in all_filenames])[-1] + 1
    print('starting index:', start_idx)

    counter_label, counter_total = 0, 0

    # start sliding-window-operations in subwindow-size (e.g. 256²)
    for idx, (x, y, window) in enumerate(sliding_window(img_original,
                                                        step_size=(window_size, window_size),
                                                        window_size=(window_size, window_size))):
        
        # get the first shape to compare this with all the others
        if counter_total == 0:
            first_shape = window.shape
            print('The first-shape:', first_shape)       
        
        counter_total += 1
        if verbose > 0:
            print(f'[num_label / num_total] -> [{counter_label} / {counter_total}] - (x,y) -> ({x}/{y})')

        # get label image 1 and 2 by extracting binary image at the same location
        y1, y2 = y, y + window.shape[0]
        x1, x2 = x, x + window.shape[1]
        img_window_binary1 = img_as_ubyte(img_binary1[ y1:y2 , x1:x2 ])
        img_window_binary2 = img_as_ubyte(img_binary2[ y1:y2 , x1:x2 ])

        # remove small occuring hole-artefacts from both binary label images
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        img_window_binary1 = morphology.binary_closing(img_window_binary1, selem=kernel)
        img_window_binary1 = morphology.remove_small_holes(img_window_binary1, area_threshold=270)
        
        img_window_binary2 = morphology.binary_closing(img_window_binary2, selem=kernel)
        img_window_binary2 = morphology.remove_small_holes(img_window_binary2, area_threshold=270)

        # remove small occuring object-artefacts which are note desired from both binary images
        img_window_binary1 = morphology.remove_small_holes(img_window_binary1, area_threshold=50)
        img_window_binary2 = morphology.remove_small_holes(img_window_binary2, area_threshold=50)

        img_window_binary1 = img_as_ubyte(img_window_binary1)
        img_window_binary2 = img_as_ubyte(img_window_binary2)

        #############################################################################################
        # CRITERION(s) to skip/select cut-out image (e.g. check if there is a manual annotation image) #
        #############################################################################################

        ### just take cut-out images which consists of more than 20% white surface in the original image
       
        ### only take cut-out images -WITH- annotations in one OR the other label image
        if (np.count_nonzero(img_window_binary1) == 0) and (np.count_nonzero(img_window_binary2) == 0):
            # print('skip (x,y):', x, y, ' --- no fungi: -id:', idx)
            continue

        # extract the original cut-out
        img_window_original = img_original[ y1:y2 , x1:x2 ]
                              
        # save the SAME original image 2 times and increment after the first 
        filename_original1 = '{0}_{1}'.format(str(start_idx+counter_label).zfill(4), os.path.basename(path_original))
        save_path_original1 = os.path.join(output_path, 'original', filename_original1)
        
        filename_original2 = '{0}_{1}'.format(str(start_idx+counter_label+1).zfill(4), os.path.basename(path_original))
        save_path_original2 = os.path.join(output_path, 'original', filename_original2)

        # save label image 1 and 2 
        filename_label1 = '{0}_{1}'.format(str(start_idx+counter_label).zfill(4), os.path.basename(path_binary1))
        save_path_label1 = os.path.join(output_path, 'labels', filename_label1)

        filename_label2 = '{0}_{1}'.format(str(start_idx+counter_label+1).zfill(4), os.path.basename(path_binary1))
        save_path_label2 = os.path.join(output_path, 'labels', filename_label2)
        
        io.imsave(save_path_original1, img_window_original)
        io.imsave(save_path_original2, img_window_original)
        
        io.imsave(save_path_label1, img_window_binary1)
        io.imsave(save_path_label2, img_window_binary2)
        
        print(f'save <1> original-image to:\t {filename_original1} with size: {img_window_original.shape} and type: {img_window_original.dtype}')
        print(f'save <2> original-image to:\t {filename_original2} with size: {img_window_original.shape} and type: {img_window_original.dtype}')
        
        print(f'save <1> labeled-image to:\t {filename_label1} with size: {img_window_binary1.shape} and type: {img_window_binary1.dtype} and unique values: {np.unique(img_window_binary1)}')
        print(f'save <2> labeled-image to:\t {filename_label2} with size: {img_window_binary2.shape} and type: {img_window_binary2.dtype} and unique values: {np.unique(img_window_binary2)}')

        counter_label += 2

        if verbose > 1:
            plot_window(img_window_original, img_window_binary1)
            plot_window(img_window_original, img_window_binary2)

    print('\n---> number of cut-out-images with labels: {0} / {1}'.format(counter_label, counter_total))


def plot_subwindows(path_original, path_binary, idx_start):
    """
    Plot all subwidnows with their binary annotations 

    Args:
        path_original ([type]): path for input image
        path_binary ([type]): path for binary image
        idx_start ([type]): start index 
    """

    all_files_original = np.sort(glob(os.path.join(path_original, '*')))
    all_files_binary = np.sort(glob(os.path.join(path_binary, '*')))
    
    assert len(all_files_original) == len(all_files_binary), "not the same number of images" 
    
    print(len(all_files_original), len(all_files_binary))
    print(all_files_original[:5])
    print(all_files_binary[:5])
    
    all_files_original = np.sort([ x for x in all_files_original if int(x.split('/')[-1].split('_')[0]) >= idx_start])
    all_files_binary = np.sort([ x for x in all_files_binary if int(x.split('/')[-1].split('_')[0]) >= idx_start])
        
    print('\nafter filtering:')
    print(len(all_files_original), len(all_files_binary))
    print(all_files_original[:5])
    print(all_files_binary[:5])    
    
    all_files_original = list(all_files_original)
    all_files_binary = list(all_files_binary)
        
    imgs_original = io.ImageCollection(all_files_original)
    imgs_binary = io.ImageCollection(all_files_binary)

    print(len(imgs_original), len(imgs_binary), '\n')
    
    for idx, image_original in enumerate(imgs_original):    
        
        image_binary = imgs_binary[idx]
        
        img_index = all_files_original[idx].split('/')[-1].split('_')[0]

        print(idx, img_index, image_original.shape, image_binary.shape, '\tfilename:', all_files_original[idx])
    
        plot_window(image_original, image_binary, title=img_index)


def preprocessing(img, mode):
    """
    Normalize the specified input image
    Args:
        img: the input image
        mode: the normalization mode

    Returns:
    """

    print(f"[preprocessing] preprocess image with mode: <{mode}>")

    if str(mode) == 'none':
        return img_as_float32(img)

    def apply_per_image(img):

        if str(mode) == 'zero_one':
            return img_as_float32(img / 255.)
        elif str(mode) == 'minus_one_to_one':
            return img.astype(np.float32) / 127.5 - 1.
        else:
            raise AttributeError("Could not find valid normalization mode - {zero_one, minus_one_to_one, none}")

    # multiple images with different shapes
    if len(img.shape) == 1 and len(img) > 1:

        for idx, x in enumerate(img):
            img[idx] = apply_per_image(x)

    # all images have the same shape
    else:
        img = apply_per_image(img)

    return img