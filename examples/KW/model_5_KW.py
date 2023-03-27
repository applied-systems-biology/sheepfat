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
  "gpus": [5],
  "log-device-placement": False
}

# Configure the devices
myutils.setup_devices(device_config)  # = Empty config = All CPUs, All GPUs

model_id = "5"

model_config = {
  "architecture": "SegNet",
  "learning_rate": 0.001,
  "output_model_path": f"./model/NN/KW/SegNet_cross-validation/model_{model_id}/untrained-model.hdf5",
  "output_model_json_path": f"./model/NN/KW/SegNet_cross-validation/model_{model_id}/untrained-model.json",
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

training_config = {
  "output_model_path": f"./model/NN/KW/SegNet_cross-validation/model_{model_id}/trained-model.hdf5",
  "output_model_json_path": f"./model/NN/KW/SegNet_cross-validation/model_{model_id}/trained-model.json",

  "input_dir": f"./train_test_data/original/modelID_{model_id}_training_data.csv",
  "label_dir": f"./train_test_data/original/modelID_{model_id}_training_data.csv",

  "max_epochs": 1000,
  "batch_size": 32,
  "validation_split": 0.8,
  "augmentation_factor": 3,
  "normalization": "zero_one",
}

print(training_config)

# Train the model
# Here we have to set the model parameter as we disabled saving of the untrained model
trained_model = SegNet.train_model(model_config=model_config, config=training_config, model=model)