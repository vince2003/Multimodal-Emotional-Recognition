#import argparse
#import json
import os
#import shutil
import sys
import numpy as np
import random
#import configs.config_template as CONFIG  # Allows use of autocomplete, this is overwritten by cmd line argument
import torch
import torch.nn as nn
#import pandas as pd
#from sklearn.metrics import roc_auc_score
import torchvision
from PIL import Image
from cqa_dataloader import build_dataloaders
import time
import pdb

from model import PReFIL

#============for bert======
#import os
#import torch
#import pandas as pd
from scipy import stats
#import numpy as np

#from tqdm import tqdm
#import torch.nn as nn
#import joblib
import pdb

#import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
#import sys

#import pandas as pd
#import transformers
#from sklearn import model_selection
import torch.backends.cudnn as cudnn
#=============================

#===============evoked expression==================
#from scipy import stats

GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
#os.environ['WANDB_MODE'] = 'dryrun'
cudnn.benchmark = True
#--------------------------------
from trial_mode import trial_arg

#---------Fix model-----------
def seed_torch(seed=0):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] =  ":16:8" # ":4096:8"
    torch.manual_seed(seed)#
    random.seed(seed)#
    np.random.seed(seed)#
    #====================================================================
    # os.environ['PYTHONHASHSEED'] = str(seed)#   
    # torch.cuda.manual_seed(seed)#
    # torch.cuda.manual_seed_all(seed)# # if you are using multi-GPU.
    #====================================================================
    torch.backends.cudnn.benchmark = False#
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True#
    
#seed_torch()



#------------config-------------------------
import my_utility as UTI
from config import get_config
args = get_config()

args.current_direction=os.getcwd()

name_project, trial_ck, max_epochs, data_subset_train, data_subset_val, shuffle, batch_size, resume, num_workers=trial_arg()




#---------wandb-----------------------------
import wandb
wandb.init(config=args, project=name_project, name=args.name_graph, save_code=True)
api = wandb.Api()
id_wandb=wandb.run.id
#pdb.set_trace()

#-----------Folder to save checkpoint--------
folder_ck=args.folder_ck
path_ck=folder_ck+trial_ck+id_wandb
if not os.path.exists(path_ck):
    os.makedirs(path_ck)  
#-------------------------------------------



def inline_print(text):
    """
    A simple helper to print text inline. Helpful for displaying training progress among other things.
    Args:
        text: Text to print inline
    """
    sys.stdout.write('\r' + text)
    sys.stdout.flush()

def show_dataloader(tensor,name="gt_",normalize=True):
    img_grid=torchvision.utils.make_grid(tensor,nrow=len(tensor),padding=0,normalize=normalize)
    print("min value:",torch.min(img_grid))
    print("max value:",torch.max(img_grid))
    npimg = img_grid.cpu().data.numpy()
    npimg_hwc=np.transpose(npimg, (1, 2, 0))
    image_alb_save = Image.fromarray((npimg_hwc* 255).astype(np.uint8))
    image_alb_save.save(name+"_.png")
    print("saving images ...........")
    
    
def fit(net, dataloader, criterion_con, optimizer, epoch, scheduler_linear_warmup):

    net.train()
    #correct = 0
    total = 0
    total_loss = 0
    take_sample=0
    
    results_con=[]
    #results_bnr=[]
    #id_image=[]
    label=[]   
    
    start = time.time()
    for token_audio, ans_con, img_tensor, id_cur in dataloader:
        #pdb.set_trace()
        token_audio=torch.squeeze(token_audio,1)
        #=======for tranformer=========
        token_audio = token_audio.cuda()
 
        ans_con = ans_con.cuda() 
        img_tensor = img_tensor.cuda()
        #pdb.set_trace()
        #show_dataloader(i, 'image_', normalize=True)      
        optimizer.zero_grad()
        
        #pdb.set_trace() 
        p_con2 = net(token_audio, img_tensor)    
        

        loss = criterion_con(p_con2, ans_con)  
        
        
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2)
        optimizer.step()
        
        #----cu 10 lan lay sample 1 lan de tinh metric------------------
        take_sample+=1
        if not take_sample%5:        
            #-------continuous------
            batch_pred_con = p_con2.tolist()
            results_con.append(batch_pred_con)       
            
            #-------label------
            batch_label = ans_con.tolist()
            label.append(batch_label)  
        #--------------------------------------------------------------
 
        if not args.warmup_step:
           scheduler_linear_warmup.step()       
 
        total += token_audio.shape[0]
        total_loss += loss * token_audio.shape[0]
        #pdb.set_trace()      
        
        inline_print(
            f'Running {dataloader.dataset.split}, Processed {total} of {len(dataloader) * dataloader.batch_size} '
            f'Loss: {total_loss / total}')
        
    
    res_con=np.vstack(results_con)    
    lbl=np.vstack(label)    

    pear_metrics_con=pear_metrics(lbl,res_con)     
        
    print("---train pear con: %3f"%(pear_metrics_con))    
    wandb.log({"PEAR_train_con":pear_metrics_con}, step=epoch)
    
    end = time.time()
    interval=end - start
    print(f'---time to train epoch {epoch} is {interval}')     
    wandb.log({"time to train each epoch":interval}, step=epoch)


def pear_metrics(label, result):
    #pear=[]
    label=label.reshape(-1)
    result=result.reshape(-1)    
    pearsonr=stats.pearsonr(label, result)[0]
    return pearsonr    
    
    
def predict(net, dataloaders, epoch):

    start = time.time()
    
    net.eval()
    for data in dataloaders:
        #results_con=[]
        #results_bnr=[]
        results_con2=[]
        #id_image=[]
        label=[]           
        with torch.no_grad():
            for token_audio, ans_con, img_tensor, id_cur in data:
                token_audio=torch.squeeze(token_audio,1)
                #pdb.set_trace()
                #=======for tranformer==========
                token_audio = token_audio.cuda()
 
                ans_con = ans_con.cuda() 
                img_tensor = img_tensor.cuda()
                #pdb.set_trace()
                #show_dataloader(i, 'image_', normalize=True)                
                
                p_con2 = net(token_audio, img_tensor)

                #------continous2--------
                batch_pred_con2=p_con2.tolist()
                results_con2.append(batch_pred_con2)
                
                #------label--------
                batch_label=ans_con.tolist()
                label.append(batch_label)                   

        
        #pdb.set_trace()                   
        #res_con=np.vstack(results_con)

        res_con2=np.vstack(results_con2)
        
        lbl=np.vstack(label)  
        
        MAE=np.average(np.absolute(res_con2-lbl))
        #pdb.set_trace()  
        #pear_metrics_con=pear_metrics(lbl,res_con)
        pear_metrics_con2=pear_metrics(lbl,res_con2)

        # ============ measure time ==============
        end = time.time()
        interval=end - start
        print(f'---time to predict epoch {epoch} is {interval}')          
        wandb.log({"time to predict epoch":interval}, step=epoch)   
        #==========================================
        
        #pdb.set_trace()
        print("---val pear_metrics_con2: %3f"%(pear_metrics_con2))
        print("---val MAE_val_con2: %8f"%(MAE))
        #print("---val pear_metrics_con: %3f --- epoch:%3i/%3i: "%(pear_metrics_con, epoch, max_epochs))    
        
        wandb.log({"PEAR_val_con2":pear_metrics_con2}, step=epoch)
        #wandb.log({"PEAR_val_con":pear_metrics_con}, step=epoch)
        
        wandb.log({"MAE_val_con2":MAE}, step=epoch)
        



def update_learning_rate(epoch, optimizer):
    if epoch < len(UTI.lr_warmup_steps):
        optimizer.param_groups[0]['lr'] = UTI.lr_warmup_steps[epoch]
    elif epoch in UTI.lr_decay_epochs:
        optimizer.param_groups[0]['lr'] *= args.lr_decay_rate    
    return optimizer.param_groups[0]['lr']



def training_loop(net, train_loader, val_loaders, optimizer, criterion_con, start_epoch=0, scheduler_linear_warmup=None):
    
    print("start to train.....")
    for epoch in range(start_epoch, max_epochs):
        if args.warmup_step:
            up_lr=update_learning_rate(epoch, optimizer)            
        
        fit(net, train_loader, criterion_con, optimizer, epoch, scheduler_linear_warmup)   # --- train      
        
        curr_epoch_path = os.path.join(path_ck, str(epoch) + '.pth')
        #pdb.set_trace()
        latest_path = os.path.join(folder_ck, 'latest.pth')
        data = {'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']}
        if epoch > 10:
            torch.save(data, curr_epoch_path)
        torch.save(data, latest_path)
            #print("save checkpoint at ",curr_epoch_path)

        if epoch % args.test_interval == 0:
            #pdb.set_trace()
            predict(net, val_loaders, epoch)  # --- predict
            
        wandb.log({"Learning rate":optimizer.param_groups[0]['lr']}, step=epoch)


def evaluate_saved(net, dataloader):
    weights_path = os.path.join(path_ck, 'latest.pth')
    saved = torch.load(weights_path)
    net.eval()
    net.load_state_dict(saved['model_state_dict'])
    predict(net, dataloader, saved['epoch'])


def main():

    print('------MODEL EFFICIENTET B0-----------')

    if args.evaluate or resume:
        #UTI.lut_location = os.path.join(EXPT_DIR, 'LUT.json')
        train_data, val_data= build_dataloaders(UTI)
    else:
        train_data, val_data = build_dataloaders(UTI)

    net = PReFIL(195)
    wandb.watch(net, log='all')

    net.cuda()
    start_epoch = 0
    if not args.evaluate:
        print('Training...')
        #optimizer = UTI.optimizer(net.parameters(), lr=UTI.lr)
        optimizer = AdamW(net.parameters(), lr=args.lr)
        
        num_train_steps = int(len(train_data)/args.batch_size*args.max_epochs)   
        scheduler_linear_warmup = get_linear_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=0,
                                num_training_steps=num_train_steps
                                )

        criterion_con = nn.MSELoss()
        print("------------MSE LOSS------------")

        if resume:            
            #resumed_data = torch.load(os.path.join(path_ck, 'latest.pth'))
            resumed_data = torch.load(args.resume_ck)
            print(f"Resuming from epoch {resumed_data['epoch']}")
            net.load_state_dict(resumed_data['model_state_dict'])
            #optimizer = UTI.optimizer(net.parameters(), lr=resumed_data['lr'])
            optimizer.load_state_dict(resumed_data['optim_state_dict'])
            optimizer.param_groups[0]['lr']=resumed_data['lr']
            start_epoch = resumed_data['epoch']
        training_loop(net, train_data, val_data, optimizer, criterion_con, start_epoch, scheduler_linear_warmup=scheduler_linear_warmup)




if __name__ == "__main__":
    main()
