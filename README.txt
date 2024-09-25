README of VAD project.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
For inference use bellow code in your terminal:

## Checkpoints
Download the checkpoints [here](https://huggingface.co/FanavaranPars/ERI-VAD).


python inference.py --waves_file_paths "BASE_PATH_OF_audio_files"
The results will be in outputs file.


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
For training use bellow code in your terminal:

python train_pyannote.py

and before that adjust the configs in recipes file.
During training the best model is saved in checkpoints fils and data of training is saved in train_result.




























