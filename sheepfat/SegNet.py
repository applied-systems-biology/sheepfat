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
import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from skimage import io, img_as_float32
from pathlib import Path
import tifffile
import time

from sheepfat import pre_processing
from sheepfat import myutils


def dice_coeff(y_true, y_pred):
    """
    Sørensen-Dice coefficient
    Args:
        y_true: true labels
        y_pred: model prediction

    Returns: Score

    """
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    """
    Sørensen-Dice loss: 1 - Sørensen-Dice coefficient
    Args:
        y_true: true labels
        y_pred: model prediction

    Returns: Score

    """

    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    """
    Custom defined binary-cross-entropy with dice loss & IoU as regularization term
    Args:
        y_true: true labels
        y_pred: model prediction

    Returns: Score

    """

    # add a small epsilon to the predictions
    EPSILON = 1e-05
    y_pred = y_pred + EPSILON

    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss


def build_model(config):
    """
    Builds a SegNet model
    Args:
        config: The model parameters

    Returns: The model

    """

    img_shape = tuple(config["image_shape"])
    reg_method = config['regularization_method']
    reg_method_rate = config['regularization_lambda']
    num_classes = config['n_classes']
    model_path = config["input_model_path"] if "input_model_path" in config else config['output_model_path']
    model_json_path = config["input_model_json_path"] if "input_model_json_path" in config else config['output_model_json_path']
    kernel_size = config["kernel_size"] if "kernel_size" in config else 3
    learning_rate = config["learning_rate"]

    def conv_block(input_tensor, num_filters):
        """

        Args:
            input_tensor:
            num_filters:

        Returns:

        """

        encoder = tf.keras.layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(input_tensor)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('relu')(encoder)
        encoder = tf.keras.layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)

        if reg_method == 'Dropout':
            encoder = tf.keras.layers.Dropout(reg_method_rate)(encoder)
        elif reg_method == 'GaussianDropout':
            encoder = tf.keras.layers.GaussianDropout(reg_method_rate)(encoder)
        elif reg_method == 'GaussianNoise':
            encoder = tf.keras.layers.GaussianNoise(reg_method_rate)(encoder)

        encoder = tf.keras.layers.Activation('relu')(encoder)

        return encoder

    def encoder_block(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

        return encoder_pool, encoder

    def decoder_block(input_tensor, concat_tensor, num_filters):
        """

        Args:
            input_tensor:
            concat_tensor:
            num_filters:

        Returns:

        """

        decoder = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)
        decoder = tf.keras.layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)
        decoder = tf.keras.layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)

        decoder = tf.keras.layers.Activation('relu')(decoder)

        return decoder

    inputs = tf.keras.layers.Input(shape=img_shape)

    encoder0_pool, encoder0 = encoder_block(inputs, 16)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 256)
    center = conv_block(encoder4_pool, 512)
    decoder4 = decoder_block(center, encoder4, 256)
    decoder3 = decoder_block(decoder4, encoder3, 128)
    decoder2 = decoder_block(decoder3, encoder2, 64)
    decoder1 = decoder_block(decoder2, encoder1, 32)
    decoder0 = decoder_block(decoder1, encoder0, 16)
   
    if num_classes == 2:
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(decoder0)

    # create the model
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=bce_dice_loss)

    model.summary()

    # save the model, model-architecture and model-config
    myutils.save_model_with_json(model=model,
                               model_path=model_path,
                               model_json_path=model_json_path,
                               model_config=config,
                               operation_config=None)

    return model


def train_model(model_config, config, model):
    """
    Trains an existing model. The existing model path is either extracted from a parameter input_model_path of the config,
    or if it does not exist, from the output_model_path of the model config
    Args:
        model: The model. If None, it is loaded from the model config or config
        model_config: Parameters of the model
        config: Training parameters

    Returns: The trained model
    """

    # assign hyper-parameter for training procedure
    input_dir = config['input_dir']
    label_dir = config['label_dir']
    normalization_mode = config['normalization']
    n_epochs = config['max_epochs']
    batch_size = config['batch_size']
    validation_split = config['validation_split']
    output_model_path = config['output_model_path']
    output_model_json_path = config["output_model_json_path"]

    augment_factor = config['augmentation_factor']

    # load the model
    assert isinstance(model, tf.keras.models.Model)
    print(f'[Train model] Use model with input shape: {model.input_shape} and output shape: {model.output_shape}')

    # read the input and label images in dependence of their specified format: directory or .csv-table
    X, X_filepath = myutils.read_images(input_dir, model_input_shape=model.input_shape, read_input=True)
    Y, Y_filepath = myutils.read_images(label_dir, model_input_shape=model.output_shape, read_input=False)

    print('[Train model] Input-images:', len(X), ', Label-images:', len(Y))

    assert len(X) == len(Y) > 0, "Unequal number of input - label images/values"

    # validate input data
    x = myutils.validate_image_shape(model.input_shape, images=X)

    # validate label data
    y = myutils.validate_image_shape(model.output_shape, images=Y)

    print('[Train model] Input data:', x.shape)
    print('[Train model] Label data:', y.shape)

    # Preprocessing of the input data (normalization)
    print('[Train model] Input image intensity min-max-range before preprocessing:', x.min(), x.max())
    if x.max() > 1:
        x = pre_processing.preprocessing(x, mode=normalization_mode)
        print('[Train model] Input image intensity min-max-range after preprocessing:', x.min(), x.max())

    # Preprocessing of the label data (normalization)
    print('[Train model] Label image intensity min-max-range before preprocessing:', y.min(), y.max())
    if y.max() > 1:
        y = pre_processing.preprocessing(y, mode=normalization_mode)
        print('[Train model] Label image intensity min-max-range after preprocessing:', y.min(), y.max())

    # Split into train - test data
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=validation_split, shuffle=True)

    print('[Train model] train data:\t', x_train.shape, y_train.shape, np.unique(y_train))
    print('[Train model] validation data:\t', x_valid.shape, y_valid.shape, np.unique(y_valid))

    data_gen_args = dict(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    # combine generators into one which yields image and masks for input and label images for training purpose
    train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    train_label_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    # provide the same seed and keyword arguments to the fit and flow methods
    seed = 42
    train_image_datagen.fit(x_train, augment=True, seed=seed)
    train_label_datagen.fit(y_train, augment=True, seed=seed)

    train_image_generator = train_image_datagen.flow(x_train, seed=seed)
    train_label_generator = train_label_datagen.flow(y_train, seed=seed)
    train_generator = zip(train_image_generator, train_label_generator)

    # calculate the augmented number of steps per epoch for the training and validation
    steps_epoch = x_train.shape[0] // batch_size
    print(f'[Train model] Number of steps per epoch-original: {steps_epoch}')
    steps_epoch = np.max([int(steps_epoch * augment_factor), 1])
    print(f'[Train model] Number of steps per epoch-augmented: {steps_epoch}')

    # fits the model on batches with real-time data augmentation:
    print('[Train model] Start training ...')

    model.fit(train_generator,
              steps_per_epoch=steps_epoch,
              epochs=n_epochs,
              verbose=1,
              validation_data=(x_valid, y_valid))

    # save the model, model-architecture and model-config
    myutils.save_model_with_json(model=model,
                               model_path=output_model_path,
                               model_json_path=output_model_json_path,
                               model_config=model_config,
                               operation_config=config)

    return model


def predict_samples(model_config, config, model=None):
    """
    Predicts a model with some input data in a dilated way
    Args:
        config: the config for this prediction
        model_config: the model config
        model: An existing model (optional)

    Returns: Prediction results (list of predictions, list of input files)

    """

    input_dir = config['input_dir']
    output_dir = config['output_dir']
    model_path = config["input_model_path"] if "input_model_path" in config else model_config['output_model_path']
    normalization_mode = config['normalization']
    model_img_shape = tuple(model_config["image_shape"])

    # load the model
    if model is not None:
        assert isinstance(model, tf.keras.models.Model)
        print(f'[Predict] Use model with input shape: {model.input_shape} and output shape: {model.output_shape}')
    else:
        model = myutils.load_and_compile_model(model_config, model_path)
        print(f'[Predict] Model successfully loaded from path: {model_path}') # and input shape: {model.input_shape}')

    # create save directory if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('[Predict] create directory folder for predicted images:', output_dir)

    # get all filenames
    filepaths = myutils.get_filesnames(path_dir=input_dir)

    assert len(filepaths) > 0, "No images found"

    # loop over individual samples
    for idx, file_name in enumerate(filepaths):

        # read the input image in dependence of their specified format: directory or .csv-table
        XY, _ = myutils.read_images(path_dir=file_name, model_input_shape=model_img_shape, read_input=True)
        image = XY[0]

        print(f"\n[Predict] [{idx + 1} / {len(filepaths)}] read image with shape: {image.shape} from path: {file_name}")

        # Preprocessing of the input data (normalization)
        x_min, x_max = image.min(), image.max()
        print('[Predict] Input image intensity min-max-range before preprocessing:', x_min, x_max)

        if x_max > 1:

            image = pre_processing.preprocessing(image, mode=normalization_mode)
            print('[Predict] Input image intensity min-max-range after preprocessing:', x_min, x_max)

        img_height = image.shape[0]
        img_width = image.shape[1]

        try:

            whole_image = image

            if len(whole_image.shape) == 3:
                whole_image = np.expand_dims(whole_image, axis=0)

            prediction = model.predict(whole_image, batch_size=1)

            print("[Predict] Attempting to predict whole image accomplished on image size:", prediction.shape)
            print("[Predict] Hint: do not use the extended mode")

        except:
            print(sys.exc_info()[1])
            print("[Predict] Predicting the whole image failed. Retrying with tiling.")

            model_height = model_img_shape[0]
            model_width = model_img_shape[1]

            # calculate the size of the prediction of interest (POI) as center windows of the actual window
            poi_height = int(model_height / 2)
            poi_width = int(model_width / 2)

            print(f"[Predict] Model size is ({model_height}, {model_width}) - POI size ({poi_height}, {poi_width})")

            # Perform padding the image to an additional border (windows_w+h / 2) to focus on center window-predictions
            pad_left, pad_top = int(model_height / 4), int(model_width / 4)

            # first calculate the minimum padding
            pad_right = pad_left
            tmp_width = pad_left + img_width + pad_right

            pad_bottom = pad_top
            tmp_height = pad_top + img_height + pad_bottom

            # second calculate the rest of the padding that it fits with the original model size
            if tmp_width % poi_width != 0:
                pad_right += poi_width - (tmp_width % poi_width)
            if tmp_height % poi_height != 0:
                pad_bottom += poi_height - (tmp_height % poi_height)

            pad_width = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
            print("[Predict] Dilated border padding with " + str(pad_width))

            img_padded = np.pad(image, pad_width=pad_width)
            print("[Predict] Dilated padded image has shape " + str(img_padded.shape))

            # perform the prediction via tiling
            prediction = None
            for (window_x0, window_x1, img_window) in pre_processing.sliding_window(img_padded,
                                                                           step_size=(poi_height, poi_width),
                                                                           window_size=(model_height, model_width)):
                # print("[Predict] [0/1] Predicting window at " + str((window_x0, window_x1, model_height, model_width))
                #       + " in image " + str(img_padded.shape))

                x_write_start = window_x1 + pad_left
                x_write_end = x_write_start + poi_width

                y_write_start = window_x0 + pad_top
                y_write_end = y_write_start + poi_height

                # boundary condition: skip the last unnecessary iterations (at the right/bottom edge)
                if x_write_start >= (pad_left + img_width):
                    print(
                        f"[Predict] Reach right-most image-border at: {pad_left} + {img_width} = {pad_left + img_width}"
                        f" and writing to [{x_write_start}-{x_write_end}]")
                    continue
                if y_write_start >= (pad_top + img_height):
                    print(
                        f"[Predict] Reach bottom-most image border at: {pad_top} + {img_height} = {pad_top + img_height}"
                        f" and writing to [{y_write_start} - {y_write_end}]")
                    continue

                img_window_expanded = img_window

                while len(img_window_expanded.shape) < 4:
                    img_window_expanded = np.expand_dims(img_window_expanded, axis=0)

                window_prediction = img_as_float32(model.predict_on_batch([img_window_expanded]))
                window_prediction = np.squeeze(window_prediction)

                # read out the center prediction window (Prediction of interest)
                x1 = int((model_width - poi_width) / 2)
                x2 = model_width - x1
                y1 = int((model_height - poi_height) / 2)
                y2 = model_height - y1

                poi_window_prediction = window_prediction[y1:y2, x1:x2]

                if prediction is None:
                    # Generate output image
                    prediction_shape = list(window_prediction.shape)
                    prediction_shape[0] = img_padded.shape[0]
                    prediction_shape[1] = img_padded.shape[1]
                    print("[Predict] Initializing output image with shape " + str(prediction_shape))
                    prediction = np.zeros(prediction_shape, dtype=np.float32)

                # print(f"[Predict] [1/1] Place centered prediction window at prediction[{y_write_start}:{y_write_end} , "
                #       + f"{x_write_start}:{x_write_end}] with window image {window_prediction.shape}")

                # Add centered window into output
                prediction[y_write_start:y_write_end, x_write_start:x_write_end] = poi_window_prediction

            # crop the original image size
            y1_crop = pad_top
            y2_crop = y1_crop + image.shape[0]
            x1_crop = pad_left
            x2_crop = x1_crop + image.shape[1]

            prediction = prediction[y1_crop:y2_crop, x1_crop:x2_crop]
            print("[Predict] Final cropped image has shape " + str(prediction.shape))

        # Postprocessing
        prediction = np.squeeze(prediction)
        prediction = img_as_float32(prediction)

        if output_dir:
            predicted_file_name = Path(output_dir) / os.path.basename(file_name)
            print("[Predict] Saving prediction result to " + str(predicted_file_name))
            if str(file_name).endswith(".tif") or str(file_name).endswith(".tiff"):
                tifffile.imsave(predicted_file_name, prediction)
            else:
                io.imsave(predicted_file_name, prediction)

    return filepaths
