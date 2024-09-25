import math
from tqdm import tqdm
import torch
from models.pyannote.utils import wav_label_to_frame_label

def precision(TP, FP):
    Precision = TP / (TP + FP + 1e-10)
    return Precision

def recall(TP, FN):
    Recall = TP / (TP + FN + 1e-10)
    return Recall

def F1_Score(Precision, Recall):
    """calcuale F1-Score criteria
    """
    output = 2 * Precision * Recall / (Precision + Recall + 1e-10)
    return output

# calcuale MCC criteria
def MCC(TP, FP, TN, FN):
    """calcuale MCC criteria
    """
    output = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)
    return output 



def evaluate_epoch(model, data_loader, loss_fn, frame_pyannote_fn, device):
    """Evaluate model with loss, F1-Score and MCC
    """
    
    model.eval()

    loss = 0
    TP = 0 # pred 1, actual 1
    FP = 0 # pred 1, actual 0
    TN = 0 # pred 0, actual 0
    FN = 0 # pred 0, actual 1
    counter = 0

    with torch.no_grad():  
        for data, _, frm_targ, lens in tqdm(data_loader):
            if frame_pyannote_fn != None:
                _, len_frame = frame_pyannote_fn(data.shape[-1])
                output = model(data.to(device)).cpu()
            else:
                len_frame = 160
                output = model(data.to(device),lens.to(device)).cpu()
            	
            del data

            output = torch.repeat_interleave(output,len_frame, dim=-2).squeeze()
            output_t, output  = wav_label_to_frame_label(output, output.shape[-1]//320, 320)
            
            frm_targ = frm_targ[:,:,None]

            if output_t.shape[1] < frm_targ.shape[1]:
                frm_targ = frm_targ[:,:output_t.shape[1],:]
            else:
                output_t = output_t[:, : frm_targ.shape[1],:]
                output = output[:, : frm_targ.shape[1],:]
                
            loss += loss_fn(output, frm_targ)

            ind_pred = output_t == 1
            ind_target =  frm_targ == 1
            
            # Calculate TP, FP, FN, TN
            TP += len(frm_targ[ind_pred * ind_target])
            FP += len(frm_targ[ind_pred * ~ind_target])
            FN += len(frm_targ[~ind_pred * ind_target])
            TN += len(frm_targ[~ind_pred * ~ind_target])

            counter += 1
    prec = precision(TP, FP)

    rec = recall(TP, FN)
    f1 = F1_Score(prec, rec)
    mcc = MCC(TP, FP, TN, FN)
    loss = loss.cpu().item() / counter

    return loss, round(f1, 3), round(mcc, 3),\
    round(prec, 3), round(rec, 3), TP, FP, TN, FN
                





