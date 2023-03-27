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


import os
import numpy as np
import tensorflow as tf
import time
import cv2
from skimage import morphology, segmentation, measure, io, filters, color, img_as_float32, util
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import tifffile as tif
from sklearn import metrics
from scipy.spatial.distance import directed_hausdorff
from IPython.display import clear_output


def plot_img(img, save_path=None):
    plt.figure(figsize=(16,10))
    plt.imshow(img, cmap='gray')

    if save_path is not None:
        io.imsave(save_path+'.png', img)
    plt.show()

    print('img-shape:', img.shape, img.dtype)


def analyse_fat_fraction(files_orig, files_pred, files_annot, path_save_directory, hyper_parameter):
    
    print('\nhyper-parameter for sample:\n', hyper_parameter, '\n')       
        
    tmp_data = []
    df_all_together = pd.DataFrame(columns=['sample', 'area_slice', 'fat_prediction', 'fat_annotation', 'difference'])

    # create save directories
    path_save_directory_tissueSegmentation = os.path.join(path_save_directory, 'tissueSegmentation')
    if not os.path.exists(path_save_directory_tissueSegmentation):
        os.makedirs(path_save_directory_tissueSegmentation)
        print('create folder:', path_save_directory_tissueSegmentation)

    path_save_directory_fatFraction = os.path.join(path_save_directory, 'fatFraction')
    if not os.path.exists(path_save_directory_fatFraction):
        os.makedirs(path_save_directory_fatFraction)
        print('create folder:', path_save_directory_fatFraction)


    # iterate over all predictions
    for i, tmp_pred_path in enumerate(files_pred):

        # choose sample: remove the ending '_prediction_threshold.png' (last 25 chars) and the first 6 chars
        abbrevation = tmp_pred_path.split('/')[-1].split('_threshold.tif')[0]
                
        print('\n[{0} / {1}] - abbrevation: {2}'.format(i, len(files_pred)-1, abbrevation))

        #if abbrevation in ['UTS03', 'S51.1', 'S35.4', 'S41.1_C', 'S47.', 'S37.3', 'S20.1_D', 'S46.', 'GTS01.4', 'S29.B4_', 'S03.B2_', 'GTS01.1']:
        #    print('skip', abbrevation)
        #    continue
            
        tissue_path_orig = [x for x in files_orig if abbrevation in x][0]
        print('original tissue:', tissue_path_orig.split('/')[-2:])

        tissue_path_pred = tmp_pred_path
        print('prediction:\t', tissue_path_pred.split('/')[-2:])

        # if (not) perform_inference:
        if files_annot is None or len(files_annot) == 0:  
            tissue_path_annot = None          
        else:
            tissue_path_annot = [x for x in files_annot if abbrevation in x][0]        
        print('annotation:\t', tissue_path_annot)
                
        try:

            # determine total area-size of the tissue slice and the erosion mask
            area_whole_tissue, img_mask_erosion = segment_slice(path=tissue_path_orig,
                                                        sample_hyper_parameter=hyper_parameter,
                                                        path_save_directory=path_save_directory_tissueSegmentation,
                                                        verbose=True)
                    
            # calculate the fat-fraction of the prediction-map (and if available for the annotation-image)
            fat_pred, fat_annot = determine_fat_fraction(path_orig=tissue_path_orig,
                                                        path_pred=tissue_path_pred,
                                                        area_total=area_whole_tissue,
                                                        img_mask_erosion=img_mask_erosion,
                                                        path_annot=tissue_path_annot,
                                                        path_save_directory=path_save_directory_fatFraction,
                                                        verbose=True)

            difference = np.abs(fat_pred - fat_annot)
            
            print('\n\nfat-fraction of prediction-map:', fat_pred / area_whole_tissue)
            if tissue_path_annot is not None:
                print('fat-fraction of annotation-map:', fat_annot / area_whole_tissue)
                print('difference of fat-fractions:', difference)
            
            tmp_data.append([tissue_path_orig.split('/')[-1][:-4], area_whole_tissue, fat_pred, fat_annot, difference])
            
            df2 = {
                'sample': tissue_path_orig.split('/')[-1][:-4], 
                'area_slice': area_whole_tissue, 
                'fat_prediction': fat_pred, 
                'fat_annotation': fat_annot, 
                'difference': difference 
            } 
            df_all_together = df_all_together.append(df2, ignore_index = True) 
            df_all_together.to_csv( os.path.join(path_save_directory, 'FatFraction_total.csv'), sep=',')

        # if analysis failed for sample store -1 values for this sample
        except:
            
            tmp_data.append([tissue_path_orig.split('/')[-1][:-4], -1, -1, -1, -1])
            
            df2 = {
                'sample': tissue_path_orig.split('/')[-1][:-4], 
                'area_slice': -1, 
                'fat_prediction': -1, 
                'fat_annotation': -1, 
                'difference': -1 
            } 
            df_all_together = df_all_together.append(df2, ignore_index = True) 
            df_all_together.to_csv( os.path.join(path_save_directory, 'FatFraction_total.csv'), sep=',')
            print('[FAIL] for sample:', tissue_path_orig.split('/')[-1], 'store -1 values')
        
    
    df = pd.DataFrame(tmp_data, columns=['sample', 'area_slice', 'fat_prediction', 'fat_annotation', 'difference'])
    df.to_csv( os.path.join(path_save_directory, 'FatFraction_total_backup.csv'), sep=',')
    
    print('Finish analysis')
    
    return df


def segment_slice(path, sample_hyper_parameter, path_save_directory=None, verbose=False):

    ##### extract hyper-parameter
    min_area_size = sample_hyper_parameter['min_area']
    smallest_filled_hole = sample_hyper_parameter['smallest_hole']
    closing_kernelSize = sample_hyper_parameter['closing_kernelSize']
    opening_kernelSize = sample_hyper_parameter['opening_kernelSize']
        
    # edit and prepare save path
    tmp = path.split('/')[-1].split('.')[:-1]
    
    print('read image from path:', path)
    try:
        img = io.imread(path)
    except:
        try:
            img = tif.imread(path)
        except:
            img = cv2.imread(path)
    print('\nimage-size:', img.shape)    
    
    if sample_hyper_parameter['use_HSV']:

        lowerRange = sample_hyper_parameter['hsv_lower_bound']
        upperRange = sample_hyper_parameter['hsv_upper_bound']

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        print('[1/10] converted to hsv-color-space:', hsv.shape)
        
        mask = cv2.inRange(hsv, lowerRange, upperRange )
        print('[2/19] performed HSV-color tresholding', mask.min(), mask.max(), mask.dtype)

    else:

        gray = color.rgb2gray(img)
        mymedian = np.median(gray)
        # set all black pixels to white one
        gray[gray == 0] = 1
        # gray[gray == 1] = mymedian 
        #print('[1/10] converted to gray-scale-color space and set all background pixels to the median:', mymedian)
        print('[1/10] converted to gray-scale-color space and set all black pixels to white one')

        # treshold via otsu
        thresh = filters.threshold_otsu(gray)
        mask = gray < thresh
        print('[2/10] performed otsu-thresolding with threshold:', thresh)    
        
    kernelClosing = morphology.disk(closing_kernelSize)
    closing = morphology.binary_closing(mask, kernelClosing)
    print('[3/10] performed morphology-closing ... with disk-shape:', kernelClosing.shape)
    
    kernelOpening = morphology.disk(opening_kernelSize)
    opening = morphology.binary_opening(closing, kernelOpening)
    print('[4/10] performed morphology-opening ... with disk-shape:', kernelOpening.shape)
    
    rm_small_objects = morphology.remove_small_objects(opening.astype(np.bool), min_size=min_area_size)
    print('[5/10] removed objects smaller than the minimal size of: ', min_area_size)
    rm_small_holes = morphology.remove_small_holes(rm_small_objects, area_threshold=smallest_filled_hole)
    print('[6/10] filled holes which are larger than: ', smallest_filled_hole)
    
    # (optional) get only the N largest regions    
    # idx_max_areas = np.argmax([cv2.contourArea(cnt) for cnt in contours]) # deprecated
    # idx_max_areas = np.array(contours).argsort()[N:][::-1]
    
    erode = morphology.binary_erosion(rm_small_holes, kernelOpening)
    print('[7/10] performed morphology-erosion ... with disk-shape:', kernelOpening.shape)
    
    label_image = ndimage.label(rm_small_holes, structure=np.ones((3, 3)))[0]
    print('[8/10] created a label image of the final binary segmentation')
        
    region_props = measure.regionprops(label_image)
    
    areas_sizes = [ obj.area for obj in region_props ]
    total_area = np.sum(areas_sizes)
    print('[INFO] total-area in px:', total_area, '\t\t- number of objects:', len(areas_sizes))
    if verbose:
        print('[INFO] object-sizes:', areas_sizes)

    print('[9/10] Finish post-processing -  start plotting')

    ### Display result ###
    if verbose:

        fig, ax_arr = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(35, 40))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8  = ax_arr.ravel()

        ax1.imshow(img)
        ax1.set_title("Origin image", fontsize=20)

        if sample_hyper_parameter['use_HSV']:
            ax2.imshow(hsv, cmap="hsv")
            ax2.set_title("HSV color-map", fontsize=20)
        else:
            ax2.imshow(gray, cmap="gray")
            ax2.set_title("Gray-scaled color-map with median: {0}".format(mymedian), fontsize=20)

        ax3.imshow(mask, cmap="gray")
        ax3.set_title("HSV based thresholding", fontsize=20)

        ax4.imshow(opening, cmap="gray")
        ax4.set_title("Morphology - closing + opening", fontsize=20)
        
        ax5.imshow(rm_small_objects, cmap="gray")
        ax5.set_title("Remove small objects", fontsize=20)

        ax6.imshow(rm_small_holes, cmap="gray")
        ax6.set_title("Fill small holes", fontsize=20)

        ax7.imshow(erode, cmap="gray")
        ax7.set_title("Erosion", fontsize=20)
        
        ax8.imshow(segmentation.mark_boundaries(img, label_image))
        ax8.contour(rm_small_holes, colors='red', linewidths=2)
        ax8.set_title("Resulting segmentation of whole slice without erosion", fontsize=20)

        plt.tight_layout()
        if path_save_directory is not None:
            plt.savefig( os.path.join(path_save_directory, '.'.join(tmp)+'_TissueSegmentation.png') )
        plt.show()

    print('[10/10] Finish plotting')
    
    return total_area, erode.astype(np.uint8)


def determine_fat_fraction(path_orig, path_pred, area_total, img_mask_erosion, path_annot=None, path_save_directory=None, verbose=False):
    
    # edit and prepare save path
    tmp = path_orig.split('/')[-1].split('.')[:-1] # TODO: ACHTUNG, hierdurch können tmp überschreiben werden (Bsp: GTS01.1 => GTS01 ; GTS01.4 => GTS01 )
    print('\nprepare path:', tmp)

    # read original image
    try:
        img_orig = io.imread(path_orig)
    except:
        try:
            img_orig = tif.imread(path_orig)
        except:
            img_orig = cv2.imread(path_orig)
    print('[1/5] read img-original:', img_orig.shape, path_orig)
    
    # read prediction image
    try:
        img_pred = io.imread(path_pred)
    except:
        try:
            img_pred = tif.imread(path_pred)
        except:
            img_pred = cv2.imread(path_pred)
    try:
        print('[2/5] read img-prediction:', img_pred.shape, path_pred)
    except:
        print('[ERROR] fail with sample:', path_orig)
        return -1, -1

    # read annotation image if available
    if path_annot is not None:
        try:
            img_annot = io.imread(path_annot)
        except:
            try:
                img_annot = tif.imread(path_annot)
            except:
                img_annot = cv2.imread(path_annot, 0)
        print('[2.5/5] read img-annotation', path_annot)

    print(img_mask_erosion.dtype, img_pred.dtype, img_mask_erosion.shape, img_pred.shape)
    
    # remove prediction at the image-boundary with the erosion mask from the original tissue
    try:
        img_pred_withoutBoundary = cv2.bitwise_and(img_mask_erosion, img_pred)
    except:
        print('[ERROR] fail with sample:', path_orig)
        return -1, -1

    num_nonZero_pred = np.count_nonzero(img_pred)
    num_nonZero_pred_withoutBoundary = np.count_nonzero(img_pred_withoutBoundary)
    frac_diff = (1 - (num_nonZero_pred_withoutBoundary / num_nonZero_pred)) * 100
    print('[3/5] binary overlay with erosion mask to exclude prediction at the image-boundary, difference in:', frac_diff, '%')
     
    label_image = ndimage.label(img_pred_withoutBoundary, structure=np.ones((3, 3)))[0]
    region_props = measure.regionprops(label_image)    
    area_fat_pred = np.sum([ obj.area for obj in region_props ])

    fat_fraction_pred = area_fat_pred / area_total
    print('[4/5] fat-fraction of the prediction-image:\t', fat_fraction_pred)

    # calculate the fraction of the manual annotation of the whole tissue slice (if available)
    if path_annot is not None:
        
        label_image = ndimage.label(img_annot, structure=np.ones((3, 3)))[0]
        region_props = measure.regionprops(label_image)
        area_fat_annot = np.sum([ obj.area for obj in region_props ])
        
        fat_fraction_annot = area_fat_annot / area_total
        print('[4.5/5] fat-fraction of the annotation-image:\t', fat_fraction_annot)
    else:
        # if there is no manual annotated image is available, return a default -1
        area_fat_annot = -1

    ### Display result ###
    if verbose:

        fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(35, 25))
        ax1, ax2, ax3, ax4 = ax_arr.ravel()
        title_fontsize = 24

        ax1.imshow(img_pred, cmap="gray")
        ax1.set_title("Prediction-map - fat-fraction: {0} %".format(fat_fraction_pred*100), fontsize=title_fontsize)

        if path_annot is not None:
            ax2.imshow(img_annot, cmap="gray")
            ax2.set_title("Annotation-map - fat-fraction: {0} %".format(fat_fraction_annot*100), fontsize=title_fontsize)
        else:
            ax2.imshow(img_pred_withoutBoundary, cmap="gray")
            ax2.set_title("Erosion prediction-map, difference in %: {0}".format(frac_diff), fontsize=title_fontsize)

        ax3.imshow(img_orig)
        ax3.contour(img_pred, colors='green', linewidths=1.2)
        if path_annot is not None:
            ax3.contour(img_annot, colors='red', linewidths=0.5)
        ax3.set_title("Resulting origin image (annotation)/prediction-map", fontsize=title_fontsize)
        
        ax4.imshow(img_orig)
        ax4.contour(img_pred_withoutBoundary, colors='green', linewidths=1.5)
        if path_annot is not None:
            ax4.contour(img_annot, colors='red', linewidths=0.5)
        ax4.set_title("Resulting image of erosion-mask (annotation)/prediction-map", fontsize=title_fontsize)
        
        plt.tight_layout()
        if path_save_directory is not None:
            plt.savefig( os.path.join(path_save_directory, '.'.join(tmp)+'_FatFraction.png') )
        plt.show()

    print('[5/5] Finish plotting')

    return area_fat_pred, area_fat_annot


def evaluate_oilred(tissue_path, fat_path):

    print(f'[0] # of tissue-path: {len(tissue_path)} - # of fat-path: {len(fat_path)}')

    results = []

    for i, t_path in enumerate(tissue_path):

        f_path = fat_path[i]
        
        print(f"[1.0] df-tissue-path: {t_path.split('#Dataset=')[-1][:15]} - df-fat-path: {f_path.split('#Dataset=')[-1][:15]}")

        df_tissue = pd.read_csv(t_path)
        df_fat = pd.read_csv(f_path)

        print(f"[1.1] df-tissue-shape: {df_tissue.shape} - df-fat-shape: {df_fat.shape}")

        tissue_area_sum = df_tissue['Area'].sum()
        fat_area_sum = df_fat['Area'].sum()

        fat_fraction = fat_area_sum / tissue_area_sum

        print(f"[1.2] tissue-area-sum: {tissue_area_sum} - fat-area-sum: {fat_area_sum} - fat-fraction: {fat_fraction}")

        file_abbreviation = f_path.split('#Dataset=')[-1][:15]
        
        results.append([file_abbreviation, tissue_area_sum, fat_area_sum, fat_fraction])
        
    column_names = ['file', 'total-tissue-area', 'total-fat-area', 'fat-fraction']
    df_results = pd.DataFrame(results, columns=column_names)

    return df_results

################################################################
# determine an evaluation for given samples (with binary data) #
################################################################

def perform_single_evaluation(img_true_path, img_pred_path, sample, verbose, img_pred=None):

    # load images, either load the prediciton image OR use directly the image for the evaluation
    img_true = cv2.imread(img_true_path, 0) 
    img_pred = cv2.imread(img_pred_path, 0)
    if img_pred is None: 
        img_pred = io.imread(img_pred_path, as_gray=True)
    
    assert img_true.shape == img_pred.shape, "True and Prediction images do not have the same shape"

    print(f'img_true-shape: {img_true.shape} - img_pred-shape: {img_pred.shape}')            

    # normalize to [0,1]
    img_true = (img_true / np.max(img_true)).astype(np.uint8)
    img_pred = (img_pred / np.max(img_pred)).astype(np.uint8)
    
    # prepare arrays by flatten these
    y_pred = img_pred.flatten()
    y_true = img_true.flatten()
    
    if verbose > 1:
        print(f'unique-values of img_true: {np.unique(img_true)} and img_pred: {np.unique(img_pred)}')
        print(f'img_true-shape: {img_true.shape} ; img_pred-shape: {img_pred.shape}\t\timg_true-type {img_true.dtype} ; img_pred-type: {img_pred.dtype}')
        print(f'number of values for y_true = {y_true.shape} and {y_pred.shape}\t\t #-y_true == #-y_pred : {y_pred.shape == y_true.shape}')
    
    # calculate the scores precision, recall, F-1 and separated accuracy
    acc_score = metrics.accuracy_score(y_true, y_pred)
    prec_score, rec_score, F1_score, support_score = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    # calculate the True-Positive, False-Positive, True-Negative and False-Negative 
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    
    if verbose > 0:
        print(f'\nAccurary:\t{acc_score}\nPrecision:\t{prec_score}\nRecall:\t\t{rec_score}\nF-1 Measure:\t{F1_score}\nTrue-Positive:\t{tp}\nFalse-Positive:\t{fp}\nFalse-Negative:\t{fn}\nTrue-Negative:\t{tn}')

    # create dataframe with all performance metrics    
    df_scores = pd.DataFrame( {
        'sample': sample,
        'Accuracy': acc_score, 
        'Precision': prec_score, 
        'Recall': rec_score, 
        'F-1': F1_score,
        'TruePositive': tp,
        'FalsePositive': fp,
        'FalseNegative': fn,
        'TrueNegative': tn,
    }, index=[ img_true_path.split('/')[-1].split('.png')[0] ] )
        
    return df_scores

################################################################
# determine an evaluation for given samples (with binary data) #
################################################################

def perform_multiple_evaluation(all_files_true, all_files_prediction, path_save_directory):

    df_scores_all = pd.DataFrame(columns=['sample', 'Accuracy', 'Precision', 'Recall', 'F-1', 'TruePositive', 'FalsePositive', 'FalseNegative', 'TrueNegative'])

    # iterate over all predictions
    for i, tmp_pred_path in enumerate(all_files_prediction):

        # choose sample: remove the file extension and just get the file name
        abbrevation = os.path.splitext(os.path.basename(tmp_pred_path))[0]

        # TODO: hier noch das sample hinzufügen
        #abbrevation = abbrevation.split('_')[0]

        print('\n[{0} / {1}] - abbrevation: {2}'.format(i, len(all_files_prediction)-1, abbrevation))

        true_path = [x for x in all_files_true if abbrevation in x][0]
        pred_path = [x for x in all_files_prediction if abbrevation in x][0]
        
        print('\ttrue_path:\t', true_path.split('/')[-2:])
        print('\tpred_path:\t', pred_path.split('/')[-2:])
        
        df_score_single = perform_single_evaluation(true_path, pred_path, sample=abbrevation, verbose=1)
        
        df_score_single.to_csv( os.path.join(path_save_directory, 'Metric_evaluation_{0}.csv'.format(i) ), sep=',')
        
        df_scores_all = df_scores_all.append(df_score_single, ignore_index=True) 
        
        #break

    df_scores_all.to_csv( os.path.join(path_save_directory, 'Metric_evaluation.csv' ), sep=',')

    return df_scores_all

################################################################
#    determine the hausdorff distance over all test samples    #
################################################################

def plot_image_label_prediction(img_input, img_label, img_prediction_binary, img_prediction_prob, point_label, point_prediction):

    (x_true, y_true) = point_label
    (x_pred, y_pred) = point_prediction
    
    fig, ax_arr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(25, 10))
    ax1, ax2, ax3, ax4 = ax_arr.ravel()
    title_fontsize = 15

    ax1.imshow(img_input, cmap="gray")
    ax1.contour(img_label, colors='green', linewidths=1.5)
    ax1.contour(img_prediction_binary, colors='red', linewidths=1.5)
    ax1.set_title("Input image", fontsize=title_fontsize)

    ax2.imshow(img_label, cmap="gray")
    ax2.set_title("Label image", fontsize=title_fontsize)
    
    ax3.imshow(img_prediction_binary, cmap="gray")
    ax3.set_title("Predicted binary image", fontsize=title_fontsize)

    ax4.imshow(img_prediction_prob, cmap="gray")
    ax4.set_title("Predicted probability image", fontsize=title_fontsize)

    # draw the indices of the points that generate the Hausdorff distance (the Hausdorff pair):
    ax1.add_artist(plt.Circle((y_true, x_true), 10, color='g'))
    ax2.add_artist(plt.Circle((y_true, x_true), 10, color='g'))
    ax3.add_artist(plt.Circle((y_true, x_true), 10, color='g'))
    ax4.add_artist(plt.Circle((y_true, x_true), 10, color='g'))
    
    ax1.add_artist(plt.Circle((y_pred, x_pred), 8, color='r'))
    ax2.add_artist(plt.Circle((y_pred, x_pred), 8, color='r'))
    ax3.add_artist(plt.Circle((y_pred, x_pred), 8, color='r'))
    ax4.add_artist(plt.Circle((y_pred, x_pred), 8, color='r'))
    
    plt.tight_layout()
    plt.show()


def determine_cross_validation_hausdorff_distance(path_test_data, path_models, threshold_dict, verbose):

    arr_collected = []

    # iterate over each k-fold
    for path_single_test_data in path_test_data:
        
        model_id = os.path.basename(path_single_test_data).split('_test_data.csv')[0][-1]
        
        df_single_test = pd.read_csv(path_single_test_data, index_col=0)
            
        model_path = [x for x in path_models if int(model_id) == int(x.split('/')[-1][-1])][0]
        model_path = os.path.join(model_path, "untrained-model.hdf5")
        
        print(f"[ {model_id} / {len(path_test_data)-1} ] - test data: {path_single_test_data} with shape: {df_single_test.shape} - model path: {model_path}")
        
        # load and compile the model
        model = tf.keras.models.load_model(model_path, compile=False)
        adam = tf.keras.optimizers.Adam()
        model.compile(optimizer=adam, loss='binary_crossentropy')
            
        # iterate over each subwindow / patch
        for index, row in df_single_test.iterrows():
            
            path_input_image = row['input']
            path_label_image = row['label']
            
            tmp_sample = os.path.splitext(os.path.basename(path_input_image))[0][5:].split('_walluks')[0]    
            
            if tmp_sample in path_input_image:
                #print(tmp_sample, threshold_dict[tmp_sample])
                pass
            else:
                print('ERROR:', tmp_sample)        
            
            img_input = io.imread(path_input_image)
            img_label = io.imread(path_label_image)
            
            # preprocess input image
            img_input = img_as_float32(img_input / 255.)
            img_input = np.expand_dims(img_input, axis=0)
            
            # predict and postprocess prediction
            img_prediction_prob = model.predict(img_input)
            img_prediction_prob = np.squeeze(img_prediction_prob)
            
            # threshold via isodata method
            thresh = threshold_dict[tmp_sample]
            img_prediction_binary = img_prediction_prob >= thresh
            
            # determine the contours of all segmented/labeled ROIs
            contours_label_tmp = measure.find_contours(img_label, 0.8)
            contours_prediction_tmp = measure.find_contours(img_prediction_binary, 0.8)
            
            # check if there are no labeled / segmented ROIs: if yes -> skip image
            if len(contours_label_tmp) == 0:
                print(f"\t[WARNING] no contours / ROIs in labeled image: {len(contours_label_tmp)}")
                continue
            if len(contours_prediction_tmp) == 0:
                print(f"\t[WARNING] no contours / ROIs in predicted image: {len(contours_prediction_tmp)}")
                continue

            if len(contours_prediction_tmp) == 1 or len(contours_label_tmp) == 1:
                continue
            

            # create one list with all concatenated points for true/pred contours
            contours_label = []
            for cnt_label in contours_label_tmp:
                contours_label.extend(list(cnt_label))
                        
            contours_prediction = []
            for cnt_prediction in contours_prediction_tmp:
                contours_prediction.extend(list(cnt_prediction))    

            # Find the general (symmetric) Hausdorff distance between two 2-D arrays of coordinates:
            max_hausdorff_dist = max(directed_hausdorff(contours_label, contours_prediction)[0], directed_hausdorff(contours_prediction, contours_label)[0])
          
            # save results
            arr_tmp = [tmp_sample, path_input_image, thresh, max_hausdorff_dist, model_id]
            arr_collected.append(arr_tmp)
                    
            print(f"\timage-idx: {index} with shape: {img_input.shape} and min/max value: {img_input.min()} / {img_input.max()} - prediction min/max probabilty: {np.around(img_prediction_prob.min(), decimals=4)} / {np.around(img_prediction_prob.max(), decimals=4)} - threshold: {thresh} - max. hausdorff distance: {max_hausdorff_dist}")
                    
            if verbose:
                
                d_h_true_pred = directed_hausdorff(contours_label, contours_prediction)
                d_h_pred_true = directed_hausdorff(contours_prediction, contours_label)

                # get the maximum directed distance

                if d_h_true_pred[0] > d_h_pred_true[0]:
                    index_true = d_h_true_pred[1]
                    index_pred = d_h_true_pred[2]
                else:
                    index_pred = d_h_true_pred[1]
                    index_true = d_h_true_pred[2]

                try:
                    (x_true, y_true) = contours_label[index_true]
                    (x_pred, y_pred) = contours_prediction[index_pred] 

                    plot_image_label_prediction(np.squeeze(img_input), img_label, img_prediction_binary, img_prediction_prob, 
                                                point_label=(x_true, y_true), point_prediction=(x_pred, y_pred))
                    
                    clear_output(wait=True)
                    time.sleep(1.5)
                
                except:
                    print('index issue')

    df_all = pd.DataFrame(arr_collected, columns =['sample', 'input_path', 'threshold', 'hausdorff_distance', 'model_id'])

    return df_all


def determine_hausdorff_distance(files_labels, files_prediction, verbose):

    arr_collected = []

    # iterate over each label image
    for idx, path_label in enumerate(files_labels):

        sample = os.path.basename(path_label).split('_')[0]
        sample_code = os.path.basename(path_label).split('_')[1]
                        
        # get the file name for the binary predicted patch
        # path_prediction = [x for x in path_prediction_patches if sample_code in x][0] 
        path_prediction = files_prediction[idx] 
                
        sample_prediction = os.path.basename(path_prediction).split('_')[0]
                              
        # read images
        img_true = io.imread(path_label)
        img_pred = io.imread(path_prediction)
        
        # for ilastik
        # img_pred = img_pred.astype(np.uint8)
        # img_pred[img_pred == 1] = 255
        # img_pred[img_pred != 255] = 0
                        
        # determine the contours of all segmented/labeled ROIs
        contours_label_tmp = measure.find_contours(img_true, 0.8)
        contours_prediction_tmp = measure.find_contours(img_pred, 0.8)
        
        # check if there are no labeled / segmented ROIs: if yes -> skip image
        if len(contours_label_tmp) == 0:
            print(f"\t[WARNING] no contours / ROIs in labeled image: {len(contours_label_tmp)}")
            continue
        if len(contours_prediction_tmp) == 0:
            print(f"\t[WARNING] no contours / ROIs in predicted image: {len(contours_prediction_tmp)}")
            continue

        if len(contours_prediction_tmp) == 1 or len(contours_label_tmp) == 1:
            continue
        
        # create one list with all concatenated points for true/pred contours
        contours_label = []
        for cnt_label in contours_label_tmp:
            contours_label.extend(list(cnt_label))
                    
        contours_prediction = []
        for cnt_prediction in contours_prediction_tmp:
            contours_prediction.extend(list(cnt_prediction))    

        # Find the general (symmetric) Hausdorff distance between two 2-D arrays of coordinates:
        max_hausdorff_dist = max(directed_hausdorff(contours_label, contours_prediction)[0], directed_hausdorff(contours_prediction, contours_label)[0])
        
        # save results
        arr_tmp = [sample_code, sample, path_label, max_hausdorff_dist]
        arr_collected.append(arr_tmp)

        if verbose:
            print(f"[ {idx} / {len(files_labels)-1} ] - sample-nr from label: {sample} - sample-nr from predictions: {sample_prediction} - sample-code: {sample_code} - max-distance: {max_hausdorff_dist}")
        elif idx % 50 == 0:
            print(f"[ {idx} / {len(files_labels)-1} ] - sample-nr from label: {sample} - sample-nr from predictions: {sample_prediction} - sample-code: {sample_code} - max-distance: {max_hausdorff_dist}")
        
    df_all = pd.DataFrame(arr_collected, columns =['sample_code', 'sample', 'path_label', 'hausdorff_distance'])

    return df_all