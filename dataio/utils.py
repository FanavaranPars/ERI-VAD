
import random
import pickle
from glob import glob
import os
import torch

def load_data_pathes(dataset_config, base_data_pth):
    base_clean_pth = os.path.join(base_data_pth, dataset_config["clean_file"])
    base_noise_pth = os.path.join(base_data_pth, dataset_config["noise_file"])
    base_rever_pth = os.path.join(base_data_pth, dataset_config["reverb_file"])

    clean_groups = dataset_config["clean_groups"]
    clean_scalse = dataset_config["clean_scalse"]

    noise_groups = dataset_config["noise_groups"]
    noise_scalse =  dataset_config["noise_scalse"]

    rirs_groups = dataset_config["rirs_groups"]
    rirs_scalse = dataset_config["rirs_scalse"]

    # LOAD DATA PATHS
    clean_train_paths = creat_data_pathes(base_clean_pth,
                                        clean_groups,
                                            clean_scalse,
                                            filenames = "train_files_path.txt")

    noise_train_paths = creat_data_pathes(base_noise_pth,
                                        noise_groups,
                                            noise_scalse,
                                            filenames = "train_files_path.txt")

    rirs_train_paths = creat_data_pathes(base_rever_pth,
                                        rirs_groups,
                                            rirs_scalse,
                                            filenames = "train_files_path.txt")
    
    base_valid_pth = os.path.join(base_data_pth, dataset_config["eval_file"])

    evalution_datasets = clean_groups

    eval_filenames = []
    for data_name in evalution_datasets:
        eval_filenames += glob(os.path.join(base_valid_pth,data_name,"*.mp3"))

    for i in range(len(eval_filenames)):
        eval_filenames[i] = eval_filenames[i].split("/")[-1]


    with open(os.path.join(base_valid_pth, "information.txt"), 'rb') as handle:
                dict_eval = pickle.load(handle)

    print(f"number of clean train : {len(clean_train_paths)}\n \
        number of noise of train : {len(noise_train_paths)} \n \
        number of reverb of train :       {len(rirs_train_paths)}")

    print(f"number of valuation data : {len(eval_filenames)}")

    return clean_train_paths, noise_train_paths, rirs_train_paths,\
    eval_filenames, dict_eval, base_clean_pth,base_noise_pth, base_rever_pth, base_valid_pth 
         
def creat_data_pathes(base_path,
                            groups,
                            g_scale,
                            filenames = "train_files_path.txt"):
    total_paths = []
    i = 0
    for group_name in groups:
        with open( os.path.join(base_path,group_name,filenames ), 'rb') as fp:
            pathes = pickle.load(fp)
        pathes = pathes * g_scale[i]
        for path in pathes:
            total_paths.append(os.path.join(group_name,path))
        
        i+=1
    
    random.seed(12)
    random.shuffle(total_paths)
    random.shuffle(total_paths)
    return total_paths 

def changed_index(ind, step = 0):
    ind_bool = ind < ind.min() - 1
    if step == -1 :
        ind_bool[1:] = (ind+1)[:-1] == ind[1:] 
    else:
        ind_bool[:-1] = (ind-step)[1:] == ind[:-1]
    
    ind_bool = ~ind_bool
    return ind_bool

def reduce_VAD_sensitive(vad_out, len_frame_ms = 20, sensitivity_ms = 60):
    vad_out = torch.tensor(vad_out)
    Th = max(int(sensitivity_ms // len_frame_ms), 1)
    
    ind0,ind1 = torch.where(vad_out== 1)

    if len(ind0) != 0:
        
        ind1_max = vad_out.shape[-1] - 1
        for i in range(-Th , Th):
            ind_1 = torch.clip(ind1-i,0,ind1_max)
            vad_out[ind0, ind_1] = 1
    return vad_out

def post_processing_VAD(vad_out, goal = 1, len_frame_ms = 20, sensitivity_ms = 100):
    """Post-processing of VAD models to change 0 label0 with 1 labels according to a sensitivity.
    """
    vad_out = torch.tensor(vad_out)
    Th = max(int(sensitivity_ms // len_frame_ms), 1)
    ind0,ind1 = torch.where(vad_out== goal)
    
    if len(ind0) != 0:
        ind1_max = vad_out.shape[-1] - 1
        ind0_last_bool = changed_index(ind0.clone())

        ind0_last = torch.where(ind0_last_bool)[0]
        ind0_first = torch.zeros_like(ind0_last)
        ind0_first[1:] = ind0_last[:-1] + 1
        ind0_first[0] = 0

        ind1_l1_bool = changed_index(ind1.clone(), step = 1)
        ind1_l1_bool[ind0_last] = False

        ind1_f1_bool = changed_index(ind1.clone(), step = -1)
        ind1_f1_bool[ind0_first] = False


        dif_bool = ind1[ind1_f1_bool] - ind1[ind1_l1_bool] > Th + 1
        l1_bool_temp = ind1_l1_bool[ind1_l1_bool].clone()
        l1_bool_temp[dif_bool] = False
        ind1_l1_bool[ind1_l1_bool.clone()] = l1_bool_temp

        f1_bool_temp = ind1_f1_bool[ind1_f1_bool].clone()
        f1_bool_temp[dif_bool] = False
        ind1_f1_bool[ind1_f1_bool.clone()] = f1_bool_temp


        second_ind = ind1[ind1_l1_bool].clone()
        for i in range(1,Th+1):
            second_ind = torch.clip(ind1[ind1_l1_bool]+i,0,ind1_max)
            desired_out = (second_ind < ind1[ind1_f1_bool])
            temp_b = vad_out[ind0[ind1_l1_bool], second_ind].clone()
            temp_b[desired_out] = goal
            vad_out[ind0[ind1_l1_bool], second_ind] = temp_b.clone()
    vad_out = vad_out
    return vad_out
   
