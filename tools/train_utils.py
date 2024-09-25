import math
import timeit
import pickle
import os
from tqdm import tqdm

import torch
from models.pyannote.utils import wav_label_to_frame_label

from tools.utils import(precision, recall, F1_Score, MCC)

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
                

def train_epoch(model, dataset, optimizer, loss_fn,
                 target_fn, device, step_show, grad_acc_step):
    """train each epoch
    """
    
    model.train()
    
    total_loss = 0
    counter = 0
    ex_counter = 0
    section = 1
    loss_section = 0

    start = timeit.default_timer()
    for data, target,_,lens in tqdm(dataset):
        target = target.to(device)
        if target_fn != None:
            target = target_fn(target).to(device)
            output = model(data.to(device))
        else:
            target, _   = wav_label_to_frame_label(target,target.shape[1]//160,160)
            output = model(data.to(device),lens.to(device)) 
            #print(output.shape, target.shape)
        
        if target.shape[1]>output.shape[1]:
            target = target[:, : output.shape[1],:]
        else:
            output = output[:, : target.shape[1],:]
        
        loss = loss_fn(output,target)
        
        
        loss.backward()
        
        # graph is cleared here
        if counter % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss
        counter += 1
        
        # change the lr at defined sections of each epoch
        if counter % step_show == 0:
            finish = timeit.default_timer()

            lr = optimizer.param_groups[0]['lr']
            l = (total_loss.detach().cpu().numpy() - loss_section) / (counter - ex_counter)
            print(f"Section {section}. lr: {lr:.3f}, Loss: {l:.4f}, Time (Min): {round((finish - start) / 60, 3)}")

            loss_section = total_loss.detach().cpu().numpy()
            ex_counter = counter
            
            for g in optimizer.param_groups:
                if g['lr'] > 0.01:
                    g['lr'] = round(g['lr'] - 0.005, 3)
            section += 1
            start = finish
    total_loss = total_loss.detach().cpu().numpy()
    if section == 1 :
        finish = timeit.default_timer()

        lr = optimizer.param_groups[0]['lr']
        l = total_loss / counter
        print(f"Section {section}. lr: {lr:.3f}, Loss: {l:.4f}, Time (Min): {round((finish - start) / 60, 3)}")

        loss_section = total_loss
        for g in optimizer.param_groups:
            if g['lr'] > 0.01:
                g['lr'] = round(g['lr'] - 0.005, 3)
        start = finish
    
    total_loss = total_loss / counter
    print(f" Train Loss: {total_loss:.5f}")

    return total_loss



# run the training and evaluation.
def run(model,
        train_loader,
        validation_loader,
        optimizer,
        loss_fn,
        target_fn,
        frame_pyannote_fn,
        save_model_path,
        chkp_path,
        step_show,
        n_epoch,
        grad_acc_step=1,
        is_finetune = False,
        DEVICE = 'cuda'
        ):
    
    """execuation of training, evaluating and saving best model
        
    """
    
    if is_finetune:
        best_loss, _, _, _, _,\
              _, _, _, _ = evaluate_epoch(model, validation_loader,
                                          loss_fn, frame_pyannote_fn,
                                        DEVICE)
        print(f'FINE TUNING \n VAlidation loss Starts from : {best_loss}')
    else: 
        best_loss = 1e10
        print(f'MAIN TRAINING')
    
    train_losses = []
    train_lrs = []
    val_results = []
    best_val_result = 0
    best_epoch = 0
    
    for epoch in range(n_epoch):
        start = timeit.default_timer()
        train_lrs.append(optimizer.param_groups[0]['lr'])
        print('\n',f"--- start epoch {epoch+1} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, 
                                 target_fn, DEVICE, step_show, grad_acc_step)
        train_losses.append(train_loss)
        
        # val_loss, val_fscore, val_mcc, precision, recall, TP, FP, TN, FN 
        val_result = evaluate_epoch(model, validation_loader,
                                    loss_fn, frame_pyannote_fn,
                                    DEVICE)
        val_results.append(val_result)

        finish = timeit.default_timer()
        print(f"Val_Loss: {val_result[0]:.4f}, Val_F1score: {val_result[1]:.3f}, Val_MCC: {val_result[2]:.3f}, Epoch_Time (min): {round((finish - start) / 60, 3)}")

        # save best model
        if best_loss > val_result[0]:
            best_loss = val_result[0]         
            best_fscore = val_result[1] 
            best_val_result = val_results[-1]
            best_mcc = val_result[2] 
            best_epoch = epoch + 1
            best_train_lr = train_lrs[-1]
            #torch.save(model.state_dict(), save_model_path) 
            print("Best model has been saved.")
        
        torch.save(model.state_dict(), save_model_path) 
        
        result_dict = {"train_losses": train_losses,
                        "val_results": val_results,
                        "best_val_result": best_val_result,
                        "best_valid_loss": best_loss,
                        "best_epoch" : best_epoch,
                        "n_f_epoch" : epoch+1,
                        "finetuning": is_finetune
                        }

        with open(os.path.join(chkp_path), 'wb') as handle:
            pickle.dump(result_dict, handle)

    print(f"\nTop validation accuracy. Epoch: {best_epoch}, lr: {best_train_lr}, Best_loss: {best_loss:.4f}, Best_Fscore: {best_fscore:.3f}, Best_MCC: {best_mcc:.3f}")
    
    
    
    return train_losses, val_results, result_dict
    
    


