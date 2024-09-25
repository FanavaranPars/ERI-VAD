import os
import json
import argparse
import timeit
from functools import partial
from glob import glob
import numpy as np
import pandas as pd
import torch
import pickle
from tools.train_utils import run

from models.pyannote.models import PyanNet
from models.pyannote.utils import pyannote_target_fn, cal_frame_sample_pyannote
from recipes.utils import load_model_config
from dataio.utils import load_data_pathes
from dataio.dataset import (audio_data_loader,
                            evaluation_data_loader)



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model_cfg", "--model_config_path",
                default= "./recipes/models/pyannote.json",
                required=False, # True,
                type=str,
                help="config of model")

ap.add_argument("-train_cfg", "--train_config_path",
                default= "./recipes/training/pyannote.json",
                required=False, # True,
                type=str,
                help="config of training")

ap.add_argument("-data_cfg", "--data_config_path",
                default= "./recipes/dataio/dataio.json",
                required=False, # True,
                type=str,
                help="config of dataio")


args = vars(ap.parse_args())


data_cfg_pth = args["data_config_path"]
train_cfg_pth = args["train_config_path"]
model_cfg_pth = args["model_config_path"]

data_configs = load_model_config(data_cfg_pth)

# DATA PATHS
base_data_pth = data_configs["dataset"]["base_data_pth"]

clean_train_paths, noise_train_paths, rirs_train_paths, eval_filenames, dict_eval,\
          base_clean_pth,base_noise_pth,base_rever_pth, \
              base_valid_pth =load_data_pathes(data_configs["dataset"],base_data_pth)


train_loader = audio_data_loader(base_clean_pth,
                                base_noise_pth,
                                base_rever_pth,
                                clean_train_paths,
                                noise_train_paths,
                                rirs_train_paths,
                                data_configs["dataset"]["base_lbl_pth"],
                                data_configs["train"]["SAMPLE_RATE"],
                                data_configs["train"]["MAX_LENGTH"],
                                data_configs["train"]["MAX_NOISE_N"], #max = 2
                                data_configs["train"]["T_REVERB"], # no reverb = -1
                                data_configs["train"]["MIN_SNR"],
                                data_configs["train"]["POSTPROCESSING"],
                                data_configs["train"]["SENS_MS"],
                                data_configs["train"]["BATCH_SIZE"], 
                                data_configs["train"]["NUM_WORKER"], 
                                data_configs["train"]["PIN_MEMORY"],
                                data_configs["train"]["TRAINING"]
                                )

validation_loader = evaluation_data_loader(data_configs["dataset"]["base_lbl_pth"],
                                       base_valid_pth,
                                       eval_filenames,
                                       dict_eval,
                                       data_configs["train"]["POSTPROCESSING"],
                                       data_configs["train"]["SENS_MS"],
                                       data_configs["evaluation"]["SAMPLE_RATE"],
                                       data_configs["evaluation"]["BATCH_SIZE"], 
                                       data_configs["evaluation"]["NUM_WORKER"], 
                                       data_configs["evaluation"]["PIN_MEMORY"])

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# use cuda if cuda available 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# DEVICE = "cpu"

model_configs = load_model_config(model_cfg_pth)

if model_configs["name"] == "Pyannote":
    model = PyanNet(model_configs)
    target_fn = partial(pyannote_target_fn, model_configs=model_configs)
    frame_pyannote_fn = partial(cal_frame_sample_pyannote,
                                 sinc_step= model_configs["sincnet_stride"],
                                 n_conv = len(model_configs["sincnet_filters"]) - 1
                                     )
else:
    raise ValueError("Your model is not supported!!")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of model's parameters : {total_params}")

model = model.to(DEVICE)


training_configs = load_model_config(train_cfg_pth)
optimizer = torch.optim.Adadelta(model.parameters(),
                                  lr= training_configs["LEARNING_RATE"],
                                    rho= training_configs["rho"],
                                      eps= training_configs["eps"])

loss_fn = torch.nn.BCELoss()

if training_configs["IS_FINETUNE"]:
    model.load_state_dict(torch.load(model_configs["param_save_path"]))

start = timeit.default_timer()
train_losses, val_results, result_dict = run(model,
                                            train_loader,
                                            validation_loader,
                                            optimizer,
                                            loss_fn,
                                            target_fn,
                                            frame_pyannote_fn,
                                            model_configs["param_save_path"],
                                            model_configs["loss_save_path"],
                                            training_configs["STEP_SHOW"],
                                            training_configs["NUM_EPOCH"],
                                            training_configs["GRAD_STEP"],
                                            training_configs["IS_FINETUNE"],
                                            DEVICE
                                            )

print('\nTOTAL TRAINING TIME (Hour) : ', round((timeit.default_timer() - start) / 3600, 3))

best_val_result = result_dict["best_val_result"]
print(f"Best loss: {best_val_result[0]}, best F1_Score: {best_val_result[1]}, best MCC: {best_val_result[2]}")
