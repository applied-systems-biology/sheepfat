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

Script to create a SegNet model
"""


import os
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io, color
from glob import glob
from sklearn.cluster import KMeans


def build_and_fit_model(config):

    # collect parameter
    number_clusters = config["n_classes"]
    max_iterations = config['max_epochs']
    input_dir = config['input_dir']
    output_model_path = config['output_model_path']

    # create the model directory
    save_directory = str(Path(output_model_path).parent)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print('[build_and_fit_model] create directory folder for model:', save_directory)

    # build model
    kmeans = KMeans(n_clusters=number_clusters, max_iter=max_iterations, verbose=1)

    print(f'[build_and_fit_model] Read images from path: {input_dir}')

    # load data
    path_images = glob(input_dir)

    # read images
    imgs_original = [io.imread(path) for path in path_images]
    print(f'[build_and_fit_model] Read images: {len(imgs_original)}')

    # convert images to lab color space
    imgs_lab = [color.rgb2lab(img) for img in imgs_original]
    print(f'[build_and_fit_model] Convert images to LAB color space: {len(imgs_lab)}')

    img_shape = imgs_lab[0].shape

    # reshape image to vector
    imgs_flatten = [img.reshape(-1, img_shape[-1]) for img in imgs_lab]
    print(f'[build_and_fit_model] Flatten color images to shape: {img_shape}')

    vector_flatten = []
    for img in imgs_flatten:
        vector_flatten.extend(img.astype(np.float32))

    vector_flatten = np.array(vector_flatten)

    print(f'[build_and_fit_model] Start clustering with vector: {vector_flatten.shape}')

    # fit model
    kmeans.fit(vector_flatten)

    print(f'[build_and_fit_model] k-means labels: {kmeans.labels_} and \ncluster-centers: \n{kmeans.cluster_centers_}')

    # save cluster center
    df_cluster_center = pd.DataFrame(kmeans.cluster_centers_, columns=["L*", "a*", "b*"])
    df_cluster_center.index.name = 'Cluster'

    df_cluster_center.to_csv(output_model_path)
    print(f'[build_and_fit_model] stored k-means center to: {output_model_path}')

    return kmeans

def predict_samples(config, model):

    input_dir = config['input_dir']
    output_dir = config['output_dir']

    # load data
    path_images = glob(input_dir)

    print(f'[predict_samples] Found paths to images: {len(path_images)}')

    for idx, path in enumerate(path_images):

        # read images
        img_original = io.imread(path)

        # convert images to lab color space
        img_lab = color.rgb2lab(img_original)

        img_shape = img_lab.shape

        # reshape image to vector
        img_flatten = img_lab.reshape(-1, img_shape[-1]).astype(np.float32)

        # perform prediction model
        arr_y = model.predict(img_flatten)

        # reshape the array back to its image format
        img_pred = arr_y.reshape(img_shape[0], img_shape[1])

        # save image
        filepath = os.path.splitext(os.path.basename(path))[0] + ".png"
        filepath = os.path.join(output_dir, filepath)

        io.imsave(filepath, img_pred, check_contrast=False)

        if idx % 10 == 0:
            print(f'Accomplished sample: [{idx} / {len(path_images) - 1}] image with shape: {img_pred.shape} to filename: {filepath}')




