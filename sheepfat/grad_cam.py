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
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from skimage import io, color, util
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib import ticker

from sheepfat import myutils
from sheepfat import pre_processing


class GradCAM:

    def __init__(self, model, classIdx, layerName=None):
        """
        Store the model, the class index used to measure the class activation map, and the layer to be used
        when visualizing the class activation map
        Args:
            model: the model object
            classIdx: the class - index
            layerName: the optional layer name for the layer to be analysed
        """

        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        """
        Attempt to find final convolutional layer in the network by looping over layers of the network in reverse order
        Returns: the layer name

        """

        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        """
        Construct our gradient model by supplying (1) the inputs to our pre-trained model,
        (2) the output of the (presumably) final 4D layer in the network, and
        (3) the output of the softmax activations from the model

        Args:
            image: input image
            eps: regularization epsilon

        Returns: return a 2-tuple of the color mapped heatmap and the output, overlaid image

        """

        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # print(gradModel.input_shape, gradModel.output_shape)

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the image through the gradient model, and
            # grab the loss associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, int(self.classIdx)]

            # print(predictions.shape, self.classIdx)

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them as weights,
        # compute the ponderation of the filters with respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range [0, 1], scale the resulting values
        # to the range [0, 255], and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_HOT):
        """
        Apply the supplied color map to the heatmap and then overlay the heatmap on the input image.
        Args:
            heatmap: heatmap image of gradients
            image: input image
            alpha: value for overlay of heatmap and input image
            colormap: color range (original default colormap=COLORMAP_HOT)

        Returns:

        """

        image = np.squeeze(image)
        heatmap = cv2.applyColorMap(heatmap, colormap).astype(np.float32)

        # convert image to RGB image if necessary
        if len(image.shape) == 2:
            image = color.gray2rgb(image)

        # parse heatmap to image type if necessary
        if heatmap.dtype != image.dtype:
            heatmap = heatmap.astype(image.dtype)

        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output, overlaid image
        return (heatmap, output)


def visualize_grad_cam(model_config, config, model=None):
    """
    Visualize the gradients of a input image with the specified model on its predicted.
    Args:
        model_config: model configuration
        config: evaluation configuration
        model: optional model object

    Returns:

    """

    # assign hyper-parameter for training procedure
    model_path = config["model_path"]
    input_dir = config['input_dir']
    output_figure_path = config['output_figure_path']
    model_img_shape = tuple(model_config["image_shape"])
    normalization_mode = config['normalization']
    label_dict = config["label_dict"]
    layer_name = config["layer_name"]

    if model is not None:
        assert isinstance(model, tf.keras.models.Model)
        print(
            f'[Visualize Grad-CAM] Use model with input shape: {model.input_shape} and output shape: {model.output_shape}')
    else:
        model = myutils.load_and_compile_model(model_config, model_path)
        print(f'[Visualize Grad-CAM] Model was successfully loaded from path: {model_path}')

    # read the input and label images in dependence of their specified format: directory or .csv-table
    X, filepath = myutils.read_images(path_dir=input_dir, model_input_shape=model_img_shape, read_input=True)

    print('[Visualize Grad-CAM] Input-images:', len(X))

    assert len(X) > 0, "No images found"

    # validate input data
    x = myutils.validate_image_shape(model.input_shape, images=X)

    print('[Visualize Grad-CAM] Input data:', x.shape)

    # Preprocessing of the input data (normalization)
    if len(x.shape) == 1 and len(x) > 1:
        # multiple images with different shapes
        x_min, x_max = x[0].min(), x[0].max()
    else:
        # all images have the same shape
        x_min, x_max = x.min(), x.max()

    print('[Visualize Grad-CAM] Input image intensity min-max-range before preprocessing:', x_min, x_max)

    if x_max > 1:
        x = pre_processing.preprocessing(x, mode=normalization_mode)

        if len(x.shape) == 1 and len(x) > 1:
            # multiple images with different shapes
            x_min, x_max = x[0].min(), x[0].max()
        else:
            # all images have the same shape
            x_min, x_max = x.min(), x.max()

        print('[Visualize Grad-CAM] Input image intensity min-max-range after preprocessing:', x_min, x_max)

    if not os.path.exists(output_figure_path):
        os.makedirs(output_figure_path)
        print('[Visualize Grad-CAM] Create directory folder for figures:', output_figure_path)

    # visualize each single input image
    for idx, image in enumerate(x):
        image = np.expand_dims(image, axis=0)

        print(f'[Visualize Grad-CAM] Image number [ {idx + 1} / {x.shape[0]} ] with shape: {image.shape}')

        # use network to make predictions on input and find class label index with largest corresponding probability
        preds = model.predict(image)

        # foreground = IMF
        i = "1"
        label = list(label_dict.keys())[list(label_dict.values()).index(i)]
        print('[Visualize Grad-CAM] Label: <{}> with min-max-probability: {:.2f}%-{:.2f}%'.format(label,
                                                                                                  preds[0].min()*100,
                                                                                                  preds[0].max()*100))
        image_label = "{}: {:.2f}%".format(label, preds[0].max()*100)
        label_probability = preds[0].max() * 100

        cam = GradCAM(model=model, classIdx=i, layerName=layer_name)
        heatmap = cam.compute_heatmap(image)

        # convert to 8-bit image
        image = (image * 255).astype("uint8")

        # invert gradient map for colormap JET (it is inverted in the plot)
        heatmap = util.invert(heatmap)

        # overlay heatmap on top of the image
        (heatmap, output) = cam.overlay_heatmap(heatmap, image, alpha=0.7, colormap=cv2.COLORMAP_JET)

        # draw the predicted label on the output image
        cv2.rectangle(output, (0, 0), (190, 40), (0, 0, 0), -1)
        cv2.putText(output, image_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        image = np.squeeze(image)

        # display the original image and resulting heatmap and output image to our screen
        if len(image.shape) != len(heatmap.shape):
            image = color.gray2rgb(image)

        # concatenate all images together
        output = np.hstack([image, heatmap, output])
        save_path = os.path.join(output_figure_path, os.path.splitext(os.path.basename(filepath[idx]))[0]+'.png')

        # create the final figure
        plt.figure(figsize=(21, 7))
        ax = plt.subplot()
        # title = "(max-) probability {:.2f}%".format(label_probability)
        # plt.title(title, fontsize=20)
        # set all axis ticks off
        plt.axis('off')
        im = ax.imshow(output)

        # create Axes on the right side. Width of cax is 2% of ax and padding between cax and ax is fixed at 0.05 inch
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cb = plt.colorbar(mpl.cm.ScalarMappable(cmap="jet"), cax=cax, format=lambda x, _: f"{x:.0%}")

        # cb.ax.set_yticklabels(["{:.0%}".format(i) for i in cb.get_ticks()])  # set ticks in format
        cb.ax.tick_params(labelsize=25)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        try:
            plt.show()
        except:
            print(f'[Visualize Grad-CAM] Skip plotting of figure')

        print(f'[Visualize Grad-CAM] Save figure to: {save_path}')

        # return