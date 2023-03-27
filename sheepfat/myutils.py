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

Script for utils functions
"""

import os
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from skimage import io, img_as_float32
from pathlib import Path
from glob import glob

from sheepfat import SegNet


def imread_collection(pattern, load_func=io.imread, verbose=False, **kwargs):
    """
    Custom version of skimage imread_collection that correctly handles TIFF files
    Args:
        verbose: verbose during the image file iteration
        load_func: function used for loading an image. the first argument will be the image path (default: skimage.io.imread)
        pattern: glob pattern

    Returns:
        imgs: images array
        filepath: absolute path to corresponding images

    """

    files = glob(pattern)

    # sort the file paths, by finding a numerical value in the basename
    try:
        files = sorted(files, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)])
        print(f'[imread_collection] INFO: Successfully applied natural string sorting')
    except:
        files.sort()
        print(f'[imread_collection] CAUTION: applied standard sort() method')

    imgs = []
    filepath = []

    for file in files:
        if verbose:
            print("[I/O] Reading", file, "...")

        img = load_func(file, **kwargs)

        # check if the image was loaded properly, if not skip this image
        if img is None:
            continue

        imgs.append(img)
        filepath.append(file)

    return imgs, filepath


def read_images(path_dir, model_input_shape, read_input):
    """
    Read images from directory if string-pattern is specified OR from csv. table
    Args:
        path_dir: directory of the images to be read
        model_input_shape: model input shape in model-config
        read_input: boolean whether the input are the inputs (X: True) or labels (Y: False)

    Returns:
        XY: input OR label images
        filepath: absolute path to corresponding images

    """

    # check whether the input data are specified within a table OR as images
    read_as_images = not str(path_dir).endswith('csv')

    # read input images in case they are provided in a table, check for <input> column in table
    if not read_as_images:

        df = pd.read_csv(path_dir)
        print(f'[Read images] Columns names in table: {df.columns}')

        if read_input:
            assert 'input' in list(df.columns), "Input format not valid: provide a table with column <input>"

            print(f'[Read images] Input is represented as images: {read_as_images} - '
                  f'with shape: {df.shape} and column names: {list(df.columns)}')

            filepath = df['input'].tolist()

        else:
            assert 'label' in list(df.columns), "Label format not valid: provide a table with column <label>"

            print(f'[Read images] Label is represented as images: {read_as_images} - '
                  f'with shape: {df.shape} and column names: {list(df.columns)}')

            filepath = df['label'].tolist()

        # differ between gray scale image with 1 pseudo channel and RGB image
        if model_input_shape[-1] == 1:
            XY = [io.imread(path, as_gray=True) for path in filepath]
        else:
            XY = [io.imread(path, as_gray=False) for path in filepath]

    elif model_input_shape[-1] > 1:
        XY, filepath = imread_collection(path_dir, verbose=False, as_gray=False)
        print(f'[Read images] Model shape: {model_input_shape} with number of channels: {model_input_shape[-1]}'
              f' - read images as RGB images')
    elif model_input_shape[-1] == 1:
        XY, filepath = imread_collection(path_dir, verbose=False, as_gray=True)
        print(f'[Read images] Model shape: {model_input_shape} with number of channels: {model_input_shape[-1]}'
              f' - read images as gray-scaled images')
    else:
        XY, filepath = imread_collection(path_dir, verbose=False, as_gray=True)
        print(f'[Read images] Model shape: {model_input_shape} with number of channels: {model_input_shape[-1]}'
              f' - read images as gray-scaled images')

    return XY, filepath


def validate_image_shape(model_shape, images):
    """
    Validate between model and input shape. Expand dimensions in loaded image if necessary

    Args:
        model_shape: should shape of the model (input or output)
        images: images load from directory

    Returns: image array in the required shape. Take care of situation if only 1 images is loaded.

    """

    images_unique_shapes = [img.shape for img in images]
    images_unique_shapes = list(set(images_unique_shapes))

    # exclude the batch size if length of dimensions (batch size, x, y, c)
    if len(model_shape) == 4:
        model_shape = model_shape[1:]

    images_arr = np.array(images)

    # if images have different shapes
    if len(images_unique_shapes) != 1:
        print('[validate_image_shape] CAUTION: shape of images do not have the same shape:', images_unique_shapes)

        for idx, img in enumerate(images_arr):

            image_shape = img.shape

            print(f'[validate_image_shape] INFO: shape of individual image : {image_shape} and model: {model_shape}')

            # check if length of shapes (and channel dimension of image and model match)
            if len(image_shape) == 2:
                print(
                    f'[validate_image_shape] CAUTION: image shape <{image_shape}> != from model shape <{model_shape}>')
                img = np.expand_dims(img, axis=-1)
                images_arr[idx] = img
                print(f'[validate_image_shape] Expand artificial channel dimension to: <{images_arr.shape}>')

    # if all images have the same shape
    else:

        image_shape = images_unique_shapes[0]

        print(f'[validate_image_shape] INFO: shape of images: {image_shape} and model: {model_shape}')

        # check if length of shapes (and channel dimension of image and model match)
        if len(image_shape) == 2:
            print(f'[validate_image_shape] CAUTION: image shape <{image_shape}> != from model shape <{model_shape}>')
            images_arr = np.expand_dims(images_arr, axis=-1)
            print(f'[validate_image_shape] Expand artificial channel dimension to: <{images_arr.shape}>')

    # Image and model shape match (including batch-dimension)
    if images_arr.shape[1:] == model_shape:
        return images_arr
    else:
        print(f"[validate_image_shape] CAUTION: image shape <{images_arr.shape[1:]}> != model shape <{model_shape[1:]}>")
        return images_arr


def setup_devices(config=None):
    """
    Sets up GPU processing according to the current config
    Args:
        config: the config

    Returns: None

    """
    if config is None:
        config = {}
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    import tensorflow as tf

    cpu_config = config.get("cpus", "all")
    gpu_config = config.get("gpus", "all")

    # Configure CPUs
    if cpu_config == "all":
        print("Using all available CPUs")
    else:
        cpus = tf.config.list_physical_devices('CPU')
        visible_cpus = []
        for id in cpu_config:
            visible_cpus.append(cpus[id])

        tf.config.set_visible_devices(visible_cpus, device_type="CPU")

    # Configure GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpu_config == "all":
        tf.config.set_visible_devices(gpus, device_type="GPU")
        print("Using all available GPUs")
    else:
        visible_gpus = []

        try:
            for id in gpu_config:
                visible_gpus.append(gpus[id])
        except:
            print('[setup_devices] please specify <all> or set the desired gpus in a list (e.g. [0,1])')

        tf.config.set_visible_devices(visible_gpus, device_type="GPU")

    # Enable/Disable device placement logging
    tf.debugging.set_log_device_placement(config.get("log-device-placement", False))


def save_model_with_json(model, model_path, model_json_path, model_config, operation_config):
    """
    Save the model, its architecture as a JSON and the config file.
    Args:
        model: the corresponding model
        model_path: the model path
        model_json_path: the path of the model architecture within a JSON
        model_config: the config file to create the model
        operation_config: the config file used for the underlying operation

    Returns:
    """

    # create the model directory
    save_directory = str(Path(model_path).parent)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print('[Save model] create directory folder for model:', save_directory)

    # save model depending on tensorflow / sklearn
    if isinstance(model, tf.keras.models.Model):
        model.save(model_path)

        # save model architecture as JSON
        model_json = model.to_json()
        with open(model_json_path, "w") as f:
            f.write(model_json)
        print('[Save model] Saved model JSON to:', model_json_path)

    elif isinstance(model, KMeans):
        with open(model_path, 'wb') as file:
            pickle.dump(model, file=file)
    print('[Save model] Saved model to:', model_path)

    # save model config
    model_config_save_path = Path(model_json_path).parent / 'model-config.json'
    with open(model_config_save_path, "w+") as f:
        json.dump(model_config, f)
    print('[Save model] Save model config file JSON to:', model_config_save_path)

    # save operation config
    if operation_config is not None:
        operation_config_save_path = Path(model_json_path).parent / 'operation-config.json'
        with open(operation_config_save_path, "w+") as f:
            json.dump(operation_config, f)
        print('[Save model] Save model config file JSON to:', operation_config_save_path)


def get_filesnames(path_dir):
    """
    Read image filenames from directory if string-pattern is specified OR from csv. table
    Args:
        path_dir: directory of the images to be read

    Returns:
        filepath: absolute path to corresponding images

    """

    # check whether the input data are specified within a table OR as images
    read_as_images = not str(path_dir).endswith('csv')

    # read input images filenames in case they are provided in a table, check for <input> column in table
    if not read_as_images:

        df = pd.read_csv(path_dir)
        print(f'[get_filesnames] Columns names in table: {df.columns}')

        assert 'input' in list(df.columns), "Input format not valid: provide a table with column <input>"

        files = df['input'].tolist()

    else:

        files = glob(path_dir)

        # sort the file paths, by finding a numerical value in the basename
        try:
            files = sorted(files, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)])
            print(f'[get_filesnames] INFO: Successfully applied natural string sorting')
        except:
            files.sort()
            print(f'[get_filesnames] CAUTION: applied standard sort() method')

    print(f'[Get filenames] Number of detected filenames: {len(files)}')

    return files


def load_and_compile_model(model_config, model_path) -> tf.keras.Model:
    """
    Loads an compiles a model
    Args:
        model_config: Model configuration
        model_path: Path to the model file

    Returns: The model
    """

    learning_rate = model_config['learning_rate']

    # load the model
    model = tf.keras.models.load_model(model_path, compile=False)

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=SegNet.bce_dice_loss)

    return model

    
