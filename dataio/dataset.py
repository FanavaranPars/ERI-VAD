import numpy as np
import librosa
import random
import pickle
import os
import torch

from torch.utils.data import Dataset, DataLoader
from speechbrain.processing.signal_processing import reverberate
from torch.nn.utils.rnn import pad_sequence

from dataio.utils import post_processing_VAD

class VAD_DATASET(Dataset):
    def __init__(self,
                 base_clean_path,
                 base_noise_path,
                 base_rever_path,
                 clean_paths,
                 noise_paths,
                 reverb_paths,
                 base_lbl = None,
                 sampling_rate = 16000,
                 max_length = 10 * 16000,
                 max_noise_n = 2, #max = 2
                 t_reverb = -1,
                 min_snr = -14,
                 is_post_process = False,
                 sens_ms = 100
                ):
        
        self.base_clean_path = base_clean_path
        self.base_noise_path = base_noise_path
        self.base_rever_path = base_rever_path
        self.base_lbl = base_lbl
        self.clean_paths = clean_paths
        self.noise_paths = noise_paths
        self.reverb_paths = reverb_paths
        self.is_post_process = is_post_process
        self.sens_ms = sens_ms
        self.len_clean = len(clean_paths)
        self.len_noise = len(noise_paths)
        self.len_reverb = len(reverb_paths)
        
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.max_noise_n = max_noise_n
        self.t_reverb = t_reverb
        
        
        self.SNR_amount = range(min_snr,31,2)
        
        self.len_snr = len(self.SNR_amount)
        print("Train dataset is ready.")
    
    def create_reverb(self, sig, reverb_filename): 
        reverb_ = torch.from_numpy(self.load_sample(reverb_filename))
        reverb_sig = reverberate(sig.unsqueeze(dim = 0), reverb_, rescale_amp= 'peak')

        return reverb_sig.squeeze()
    
    def load_sample(self, path):
        waveform, _ = librosa.load(path, sr=self.sampling_rate)
        return waveform
    
    def crop_noise(self, noise, len_x):
        len_n = len(noise)
        extra = len_n - len_x
        if extra > 0:
            first_ind = random.randint(0,extra - 1)
            noise = noise[first_ind:first_ind+len_x]
        
        return noise
    
    def crop_audio(self, x):
        len_x = len(x)
        extra = len_x - self.max_length
        if extra > 0:
            x = x[:self.max_length]
            len_x = self.max_length
        
        return x, len_x
    
    def creat_noisy_data(self, x_clean, noise, SNR):
        sp_ener = torch.sum(x_clean**2)
        noi_ener = torch.sum(noise**2)
        a = (sp_ener/(noi_ener + 1e-10))**0.5 * 10**(-SNR/20)
        x_noisy = x_clean + a * noise
        return x_noisy
    
    def prepare_noise(self, path, len_x):
        noise = self.load_sample(path)
        len_n = len(noise)
        if len_n < len_x:
            repeat = len_x // len_n + 1 
            noise = [noise for _ in range(repeat)]
            noise = np.concatenate(noise, axis=0)

        noise = self.crop_noise(noise, len_x)
        return noise
    
    def creat_target(self, clean_flnm, len_x):
        dataset_nm = clean_flnm.split("/")[0]
        if dataset_nm == "TinyTIMIT":
            label_flnm = clean_flnm.split(dataset_nm)[-1][1:-4].replace("/","_")+".txt"
        else: 
            label_flnm = os.path.basename(clean_flnm).split("SPLIT")[0] + ".txt"

        with open(os.path.join(self.base_lbl,dataset_nm,label_flnm), 'rb') as handle:
            framed_label = np.array(pickle.load( handle))
        
        if self.is_post_process:
            framed_label = framed_label[None,...]
            framed_label = post_processing_VAD(framed_label, 
                                        goal = 1, 
                                        len_frame_ms = 20, 
                                        sensitivity_ms = self.sens_ms).squeeze()
        label = np.repeat(framed_label, 320, axis=0)
        
        if label.shape[-1] > len_x:
            label = label[:len_x]

        return label, framed_label
    
    

    def __len__(self):
        return len(self.clean_paths)
        

    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(os.path.join(self.base_clean_path,
                                                self.clean_paths[index]))
        x_clean, len_x = self.crop_audio(x_clean)
        x_clean = x_clean * np.random.uniform(0.7,1,1)

        noise_index = random.sample(range(self.len_noise),1)[0]
        noise = self.prepare_noise(os.path.join(self.base_noise_path,
                                           self.noise_paths[noise_index]),len_x)
                
        x_clean = torch.from_numpy(x_clean).float()
        noise = torch.from_numpy(noise).float()
        
        is_reverb = torch.rand(1) < self.t_reverb
        
        if is_reverb:
            rev_index = random.sample(range(self.len_reverb),1)[0]
            x_clean = self.create_reverb(x_clean,
                                         os.path.join(self.base_rever_path,
                                                      self.reverb_paths[rev_index]))

            rev_index = random.sample(range(self.len_reverb),1)[0]
            noise = self.create_reverb(noise,
                                       os.path.join(self.base_rever_path,
                                                    self.reverb_paths[rev_index]))
            

        n_o_n = random.randint(1,self.max_noise_n)
        if n_o_n == 2:
            noise_index = random.sample(range(self.len_noise),1)[0]
            noise_2 = self.prepare_noise(os.path.join(self.base_noise_path,
                                                 self.noise_paths[noise_index]),
                                    len_x)
            
            noise_2 = torch.from_numpy(noise_2).float()
            
            if is_reverb:
                rev_index = random.sample(range(self.len_reverb),1)[0]
                noise_2 = self.create_reverb(noise,
                                             os.path.join(self.base_rever_path,
                                                          self.reverb_paths[rev_index]))
            noise = noise + noise_2
        snr = self.SNR_amount[random.sample(range(self.len_snr),1)[0]]
        
        
        x_noisy = self.creat_noisy_data(x_clean, noise, snr)

        
        
        if self.base_lbl != None:
            target, framed_target = self.creat_target(self.clean_paths[index], len_x)
            if len_x > target.shape[-1]:
                x_noisy = x_noisy[:target.shape[-1]]

            target = torch.from_numpy(target)
            framed_target = torch.from_numpy(framed_target)
        else:
            target, framed_target = None, None
        return x_noisy, target, framed_target, is_reverb, n_o_n, snr


class VAD_evaluation_DATASET(Dataset):
    def __init__(self,
                 base_lbl,
                 base_eval_pth,
                 eval_filenames,
                 information_dict,
                 sampling_rate = 16000,
                 is_post_process = False,
                 sens_ms = 100
                ):
        
        self.base_lbl = base_lbl
        self.base_eval_pth = base_eval_pth
        self.eval_filenames = eval_filenames
        self.information_dict = information_dict
        
        self.sampling_rate = sampling_rate
        self.is_post_process = is_post_process
        self.sens_ms = sens_ms
    
    def load_sample(self, path):
        waveform, _ = librosa.load(path, sr=self.sampling_rate,  dtype='float32')
        return waveform
    
    def creat_target(self, clean_flnm, len_x):
        dataset_nm = clean_flnm.split("/")[0]
        if dataset_nm == "TinyTIMIT":
            label_flnm = clean_flnm.split(dataset_nm)[-1][1:-4].replace("/","_")+".txt"
        else: 
            label_flnm = os.path.basename(clean_flnm).split("SPLIT")[0] + ".txt"
        
        with open(os.path.join(self.base_lbl,dataset_nm,label_flnm), 'rb') as handle:
            framed_label = np.array(pickle.load( handle))
        
        if self.is_post_process:
            framed_label = framed_label[None,...]
            framed_label = post_processing_VAD(framed_label, 
                                        goal = 1, 
                                        len_frame_ms = 20, 
                                        sensitivity_ms = self.sens_ms).squeeze()
        label = np.repeat(framed_label, 320, axis=0)
        
        if label.shape[-1] > len_x:
            label = label[:len_x]

        return label, framed_label
        
    def __len__(self):
        return len(self.eval_filenames)
        

    def __getitem__(self, index):
        # load to tensors and normalization
        num, data_name, snr, f_n_n,\
              s_n_n, c_r_n, f_r_n_n, s_r_n_n = self.eval_filenames[index][:-4].split("_")
        
        x_noisy = self.load_sample(os.path.join(self.base_eval_pth,
                                                data_name,
                                                self.eval_filenames[index]))
        
        inform = self.information_dict[int(num)]

        x_noisy = torch.from_numpy(x_noisy).float()

        target, framed_target = self.creat_target(inform["clean_path"],
                                                   x_noisy.shape[-1])
        target = torch.from_numpy(target).float()
        framed_target = torch.from_numpy(framed_target).float()

        return x_noisy, target, framed_target, inform, "", ""
        
        
def collate_fn(batch):
    inputs, targets, framed_targets, length_ratio = [], [], [], []
    for noisy_input, target, framed_target, _, _, _ in batch:
        inputs.append(noisy_input)
        targets.append(target)
        framed_targets.append(framed_target)
        length_ratio.append(len(noisy_input))

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
    framed_targets = pad_sequence(framed_targets, batch_first=True, padding_value=0.0)
    length_ratio = torch.tensor(length_ratio, dtype=torch.long) / inputs.shape[-1]

    return inputs, targets, framed_targets, length_ratio
    
def evaluation_data_loader(base_lbl,
                        base_eval_pth,
                        eval_filenames,
                        information_dict,
                        is_post_process ,
                        sens_ms ,
                        sampling_rate ,
                        batch_size, 
                        num_workers, 
                        pin_memory
                        ):
        
    dataset = VAD_evaluation_DATASET(base_lbl,
                                    base_eval_pth,
                                    eval_filenames,
                                    information_dict,
                                    sampling_rate ,
                                    is_post_process ,
                                    sens_ms )
    
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        shuffle = False,
                        drop_last = False,
                        collate_fn = collate_fn,
                        num_workers = num_workers,
                        pin_memory = pin_memory
                        )
    
    return loader 

# for reading and preparing dataset
def audio_data_loader(base_clean_path,
			base_noise_path,
			base_rever_path,
			clean_paths,
			noise_paths,
			reverb_paths,
			base_lbl,
			sampling_rate ,
			max_length ,
			max_noise_n, #max = 2
			t_reverb,
			min_snr,
			is_post_process,
			sens_ms,
			batch_size, 
			num_workers, 
			pin_memory,
			training
			):
    
    
    dataset = VAD_DATASET(base_clean_path,
				 base_noise_path,
				 base_rever_path,
				 clean_paths,
				 noise_paths,
				 reverb_paths,
				 base_lbl,
				 sampling_rate,
				 max_length,
				 max_noise_n, #max = 2
				 t_reverb,
				 min_snr,
				 is_post_process,
				 sens_ms)
    
    
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        shuffle = training,
                        drop_last = True,
                        collate_fn = collate_fn,
                        num_workers = num_workers,
                        pin_memory = pin_memory
                        )
    
    
    return loader
