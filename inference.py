import os
import glob
import argparse
import numpy as np
import librosa
import torch
import torchaudio
import timeit
import soundfile
from functools import partial

from dataio.utils import reduce_VAD_sensitive, post_processing_VAD
from recipes.utils import load_model_config
from models.pyannote.models import PyanNet
from models.pyannote.utils import pyannote_target_fn, cal_frame_sample_pyannote

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0),torch.cuda.get_device_name(1))


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model_cfg", "--model_config_path",
                default= "./recipes/models/pyannote.json",
                required=False, # True,
                type=str,
                help="config of model")

ap.add_argument("-base_output", "--base_output_path",
                default= "./outputs",
                required=False, # True,
                type=str,
                help="path of saving the output of vad")

ap.add_argument("-wav_fls_pth", "--waves_file_paths",
                required= True,
                type=str,
                help="path of the file yhat is includes audio files")

args = vars(ap.parse_args())


wav_fls_pth = args["waves_file_paths"]
base_saved_path = args["base_output_path"]
model_config_path = args["model_config_path"]


model_configs = load_model_config(model_config_path)

if model_configs["name"] == "Pyannote":
    model = PyanNet(model_configs)
    target_fn = partial(pyannote_target_fn, model_configs=model_configs)
    frame_pyannote_fn = partial(cal_frame_sample_pyannote,
                                 sinc_step= model_configs["sincnet_stride"],
                                 n_conv = len(model_configs["sincnet_filters"]) - 1)

else :
    raise ValueError("the name of the VAD model is not supported!!")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of model's parameters : {total_params}")

model = model.to(DEVICE)
model.load_state_dict(torch.load(model_configs["param_save_path"]))
model.eval()

main_pathes = ["*", "*/*", "*/*/*" ] 
path_list = []
for main_path in main_pathes :
    for path in glob.glob(os.path.join(wav_fls_pth,main_path)):
        if path[-3:] == "mp3" or path[-3:] == "wav":
            path_list.append(path)
            print(path)

sr=16000
batch_size = 128

torch.cuda.empty_cache()

for path_audio in path_list:

    start_total = timeit.default_timer()

    print(f"****File name: {path_audio}")
    
    metadata =torchaudio.info(path_audio)
    main_sr = metadata.sample_rate
    sig, sr = librosa.load(path_audio, sr=sr, dtype='float32')
    print(f"Readed the signal: {sig.shape}, main_sr: {sr} \n ")
    
    chunk_size = sr * 10
    sig_l = len(sig)
    chunk_n = sig_l // chunk_size

    padded_sig = np.zeros(((chunk_n+1)* chunk_size,),dtype='float32')
    padded_sig[:sig_l] = sig

    padded_sig_l = len(padded_sig)

    chunked_sig = padded_sig.reshape(-1,chunk_size)
    voiced_chunked_sig  = np.zeros_like(chunked_sig)

    model.eval()

    _ , len_frame = cal_frame_sample_pyannote(chunk_size,
                                                         sinc_step= model_configs["sincnet_stride"])
    
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in range(0,chunked_sig.shape[0],batch_size):
            vad_predict = model(torch.from_numpy(chunked_sig[batch:batch + batch_size]).to(DEVICE)).cpu()
            vad_predict = (vad_predict > 0.5).int()
            vad_predict = vad_predict[...,0]
            vad_predict = reduce_VAD_sensitive(vad_predict, len_frame_ms = len_frame/16 , sensitivity_ms = 60)
            vad_predict = post_processing_VAD(vad_predict, goal = 1, len_frame_ms = len_frame/16 , sensitivity_ms = 100)
            vad_predict = torch.repeat_interleave(vad_predict,len_frame, dim=-1)
            voiced_chunked_sig[batch:batch + batch_size,:vad_predict.shape[-1]] = vad_predict.numpy()
            voiced_chunked_sig[batch:batch + batch_size,vad_predict.shape[-1]:] = vad_predict[:,-1][...,None]


    print(f"Total Time (min): {(timeit.default_timer() - start_total)/60}")
    
    voiced_chunked_sig = voiced_chunked_sig.reshape(-1,)[:sig_l]
    # removing non speech
    voiced_sig = sig[voiced_chunked_sig!=0]
    path = path_audio.split("/")[-1][:-4]
    soundfile.write(os.path.join(base_saved_path,path+"_VAD"+".mp3") , voiced_sig, sr)

