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


# own written scripts
import sys
sys.path.append('./../../sheepfat/')
print(sys.path[-1])

from sheepfat import SegNet
from sheepfat import myutils

device_config = {
  "cpus": "all",
  "gpus": [6],
  "log-device-placement": False
}

# Configure the devices
myutils.setup_devices(device_config)  # = Empty config = All CPUs, All GPUs

model_id = "6"

model_config = {
  "architecture": "SegNet",
  "learning_rate": 0.001,
  "output_model_path": f"./model/NN/KW/SegNet_cross-validation/model_{model_id}/trained-model.hdf5",
  "output_model_json_path": f"./model/NN/KW/SegNet_cross-validation/model_{model_id}/trained-model.json",
  "regularization_method": "none",
  "regularization_lambda": 0.0,
  "weight_regularization": "l2",
  "weight_regularization_lambda": 0.001,
  "image_shape": [512, 512, 3],
  "n_classes": 2
}

print(model_config)

# Create the model
model = SegNet.build_model(model_config)

predict_config = {
    "input_dir": f"./train_test_data/original/modelID_{model_id}_test_data.csv",
    "output_dir": f"./data/prediction/H&E/egNet_cross-validation/images_KW/probabilty_map_POI/",
    "normalization": "zero_one",
}

print(predict_config)

# Predict the data
filepaths = SegNet.predict_samples(model_config=model_config, config=predict_config)