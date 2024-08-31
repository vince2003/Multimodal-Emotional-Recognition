#import argparse
import json
import os
import shutil
import sys
import numpy as np
import random
#import configs.config_template as CONFIG  # Allows use of autocomplete, this is overwritten by cmd line argument
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score
import torchvision
from PIL import Image
from cqa_dataloader_test import build_dataloaders_test
import pdb
import wandb

#===============evoked expression==================
from scipy import stats
import time
from model import PReFIL


GPU_ID = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

#os.environ['WANDB_MODE'] = 'dryrun'

#--------------------------------
#from trial_mode import trial_arg
#---------wandb-----------------------------
#pdb.set_trace()

#---------wandb-----------------------------
import my_utility as UTI
from config import get_config
args = get_config()


wandb.init(config=args, project="testing", name=args.name_graph, save_code=True)
api = wandb.Api()
id_wandb=wandb.run.id
#pdb.set_trace()
#---------Fix model-----------
def seed_torch(seed=0):
    random.seed(seed)#
    os.environ['PYTHONHASHSEED'] = str(seed)#
    np.random.seed(seed)#
    torch.manual_seed(seed)#
    torch.cuda.manual_seed(seed)#
    torch.cuda.manual_seed_all(seed)# # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False#
    torch.backends.cudnn.deterministic = True#
#seed_torch()



# #------------config-------------------------
# import config as CONFIG
# from config import get_config
# args = get_config()

result_path="./result_csv/"
if not os.path.exists(result_path):
    os.makedirs(result_path)  

#-------------------------------------------
interval_st=2000000
columns = ['amusement',
           'anger',
           'awe',
           'concentration',
           'confusion',
           'contempt',
           'contentment',
           'disappointment',
           'doubt',
           'elation',
           'interest',
           'pain',
           'sadness',
           'surprise',
           'triumph']



def path_to_ID_timestamp(path):
    list_id_imag=path.split('/')
    id_ = list_id_imag[-1]
    sep_id=id_.split('_')  
    ID_youtube='_'.join([str(elem) for elem in sep_id[:-1]]) # chan id co chua "_"
    stamp_youtube=sep_id[-1]
    return [ID_youtube, int(stamp_youtube)]

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
    
    
def pear_metrics(label, result):
    #pear=[]
    label=label.reshape(-1)
    result=result.reshape(-1)    
    pearsonr=stats.pearsonr(label, result)[0]
    return pearsonr   

def decreasing_timestamp(value):
    max_batch=int(value)
    list_time=[max_batch]
    for i in range(12):        
        max_batch=int(max_batch-166666.66666666666)        
        if (max_batch%10)==9:
            max_batch=max_batch+1
        list_time.append(int(max_batch))
    return list_time


def df_id(id_cur):        
    id_, stamp = path_to_ID_timestamp(id_cur)
    list_stamp=decreasing_timestamp(stamp)    
    list_stamp=list_stamp[::-1] # dao lai thu tu trong timestamp
    
    list_id=[id_]*13
    df_ids_stamp=pd.DataFrame({'Video ID':list_id, 'Timestamp (milliseconds)':list_stamp})   
    #print(df_ids_stamp)
    #pdb.set_trace()
    if stamp%interval_st:  # du thi lay phan du phia sau, bo phan dau
        at_point=int((stamp%interval_st)/166666)
        df_ids_stamp=df_ids_stamp.iloc[-at_point:]
    else:
        if stamp==2000000:
            df_ids_stamp=df_ids_stamp  # du 13 khi no bat dau bang ko
        else:
            df_ids_stamp=df_ids_stamp.iloc[1:] # chi lay 12, bo cai dau 
    
    df_ids_stamp['id_stamp']=df_ids_stamp['Video ID']+"_"+df_ids_stamp['Timestamp (milliseconds)'].astype(str)# tao index phai la unique
    df_ids_stamp=df_ids_stamp.set_index('id_stamp')
    
    index_in_nan=[np.nan]*(13-df_ids_stamp.shape[0])+list(df_ids_stamp.index) # chen nan vao may cai da bo, cho du size cho viec concatenate
    df_ids_stamp=df_ids_stamp.reindex(index=index_in_nan)
    df_ids_stamp.reset_index(inplace=True)
    #pdb.set_trace()            
    return df_ids_stamp    
    

def df_emotions(batch_pred): 
    #pdb.set_trace()
    batch_pred=batch_pred.reshape(13,15)
    df_emo=pd.DataFrame(batch_pred, columns=columns)     
    #print(df_emo)
    return df_emo
    
def df_combine(list_couple):
    each_batch_df=pd.concat([list_couple[0],list_couple[1]], axis=1)
    #pdb.set_trace()
    each_batch_df.dropna(inplace=True)
    return each_batch_df
    
def to_df(batch_pred, id_cur):
    list_df_emo=list(map(df_emotions,batch_pred))
    list_df_label=list(map(df_id,id_cur))
    list_couple=list(zip(list_df_label,list_df_emo))
    #pdb.set_trace()
    list_df=list(map(df_combine,list_couple))
    
    list_df=list_df[::-1]
    
    df_batch=pd.concat(list_df, axis=0)
    df_batch.reset_index(inplace=True)
    df_batch.pop('index')
    #pdb.set_trace()    
    return df_batch

def predict_val(net, dataloaders, epoch):
    start = time.time()    
    net.eval()
    for data in dataloaders:
        results_con2=[]
        label=[]
        total = 0         
        df_store=[] 

        df_thu_label=[]        
        with torch.no_grad():
            for token_audio, ans_con, img_tensor, id_cur in data:
                token_audio=torch.squeeze(token_audio,1)
                #pdb.set_trace()
                #======for tranformer==========
                token_audio = token_audio.cuda()
 
                ans_con = ans_con.cuda() 
                img_tensor = img_tensor.cuda()
                #pdb.set_trace()
                #show_dataloader(i, 'image_', normalize=True)                
                
                p_con2 = net(token_audio, img_tensor)
                
                #================De tinh metric=============================
                #------continous2--------
                batch_pred_con2=p_con2.tolist()
                results_con2.append(batch_pred_con2)
                
                #------label---------
                batch_label=ans_con.tolist()
                label.append(batch_label) 
                
                #============================================================       
                
                #============tao dataframe cho moi batch prediction====================
                batch_pred_con2_val=p_con2.cpu().numpy()
                df_batch=to_df(batch_pred_con2_val, list(id_cur))
                #pdb.set_trace()
                df_store.append(df_batch)
                #============================================================
                
                #============tao lai dataframe label de check====================
                batch_label_num=ans_con.cpu().numpy()
                df_batch_label=to_df(batch_label_num, list(id_cur))
                #pdb.set_trace()
                df_thu_label.append(df_batch_label)
                #============================================================
                
                total += token_audio.shape[0]
                inline_print(f'Running {data.dataset.split}, Processed {total} of {len(data) * data.batch_size} ')

        #==============prediction=============================
        val_df=pd.concat(df_store,axis=0)
        val_df.reset_index(inplace=True)
        #pdb.set_trace()
        val_df.drop(['index','id_stamp'], inplace=True, axis=1)
        #pdb.set_trace()
        val_df.to_csv('./result_csv/results_val_epoch'+str(epoch)+'.csv', index=False, header=True)    

        #=================tao lai df label de check code dung hay ko=============================================
        lbl_df=pd.concat(df_thu_label,axis=0)
        lbl_df.reset_index(inplace=True)
        #pdb.set_trace()
        lbl_df.drop(['index','id_stamp'], inplace=True, axis=1)
        #pdb.set_trace()
        lbl_df['id_stamp']=lbl_df['Video ID']+"_"+(lbl_df['Timestamp (milliseconds)'].astype(int)).astype(str)# tao index phai la unique
        lbl_df.set_index('id_stamp', inplace=True)  
        #---------data val goc------------------------------
        ori_val_df= pd.read_csv('val.csv')   
        #pdb.set_trace()
        ori_val_df['id_stamp']=ori_val_df['YouTube ID']+"_"+(ori_val_df['Timestamp (milliseconds)'].astype(int)).astype(str)# tao index phai la unique
        ori_val_df.set_index('id_stamp', inplace=True)

        #--------------theo index data goc va dien 0 neu no la Nan------------
        lbl_df=lbl_df.reindex(list(ori_val_df.index))        
        #test_df=test_df.fillna(method='pad')
        lbl_df=lbl_df.fillna(0)        
        #pdb.set_trace()
        #------------dien gia tri vao file submit-----------------------------
        #pdb.set_trace()
        ori_val_df[columns]=lbl_df.iloc[:,2:]
        ori_val_df.reset_index(inplace=True)
        ori_val_df.pop('id_stamp')
        
        ori_val_df.to_csv('./result_csv/results_label_epoch'+str(epoch)+'.csv', index=False, header=True)
        
        #======================================================================================================  


        #=============tinh metrics-==============

        res_con2=np.vstack(results_con2)
        
        lbl=np.vstack(label)        

        pear_metrics_con2=pear_metrics(lbl,res_con2)
        #=========================================

        # ============ measure time ==============
        end = time.time()
        interval=end - start
        print(f'---time to predict epoch {epoch} is {interval}')          
        wandb.log({"time to predict epoch":interval}, step=epoch)   
        #==========================================
        
        #pdb.set_trace()
        print("---val pear_metrics_con2: %3f"%(pear_metrics_con2))          
        
        wandb.log({"PEAR_val_con2":pear_metrics_con2}, step=epoch)

        #pdb.set_trace()
        print("\nDone validation")        
        
def predict_test(net, dataloaders, epoch):

    start = time.time()
    
    net.eval()
    for data in dataloaders:
        label=[]
        total = 0        
        df_store=[]           
        with torch.no_grad():
            for token_audio, ans_con, img_tensor, id_cur in data:
                try:
                    token_audio=torch.squeeze(token_audio,1)
    
                    #======for tranformer==========
                    token_audio = token_audio.cuda()
    
                    #======for efficientNet==========                
                    img_tensor = img_tensor.cuda()
                    #pdb.set_trace()
                    #show_dataloader(i, 'image_', normalize=True)                   
                    p_con2 = net(token_audio, img_tensor)

                    #==========tao dataframe cho tung batch===================
                    batch_pred_con2_val=p_con2.cpu().numpy()
                    df_batch=to_df(batch_pred_con2_val, list(id_cur))
                    #pdb.set_trace()
                    df_store.append(df_batch)
                    #=========================================================                    
    
                    total += token_audio.shape[0]
                    inline_print(f'Running {data.dataset.split}, Processed {total} of {len(data) * data.batch_size} ')  
                    
                except Exception:
                    print("\n----error----")
                    pdb.set_trace()
                    
                    
        #=================tao file submit======================================
        test_df=pd.concat(df_store,axis=0)
        test_df.reset_index(inplace=True)
        #pdb.set_trace()
        test_df.drop(['index','id_stamp'], inplace=True, axis=1)
        test_df['id_stamp']=test_df['Video ID']+"_"+(test_df['Timestamp (milliseconds)'].astype(int)).astype(str)# tao index phai la unique
        test_df.set_index('id_stamp', inplace=True)       
        
        #pdb.set_trace()
        #---------data goc------------------------------
        ori_test_df= pd.read_csv('test.csv')        
        ori_test_df['id_stamp']=ori_test_df['Video ID']+"_"+(ori_test_df['Timestamp (milliseconds)'].astype(int)).astype(str)# tao index phai la unique
        ori_test_df.set_index('id_stamp', inplace=True)
        #-----------------------------------------------
        
        #--------------theo index data goc va dien 0 neu no la Nan------------
        test_df=test_df.reindex(list(ori_test_df.index))        
        #test_df=test_df.fillna(method='pad')
        test_df=test_df.fillna(0)
        
        #------------dien gia tri vao file submit-----------------------------
        #pdb.set_trace()
        ori_test_df[columns]=test_df.iloc[:,2:]
        ori_test_df.reset_index(inplace=True)
        ori_test_df.pop('id_stamp')
        
        ori_test_df.to_csv('./result_csv/results_test_epoch'+str(epoch)+'.csv', index=False, header=True)
        
        #======================================================================
        
        # ============ measure time ==============
        end = time.time()
        interval=end - start
        print(f'---time to predict epoch {epoch} is {interval}')          
        wandb.log({"time to predict epoch":interval}, step=epoch)   
        #==========================================        
        #pdb.set_trace()
        print("\nDone testing")              

def evaluate_val_saved(net, dataloader):
    #pdb.set_trace()
    ck_testing=args.ck_testing
    saved = torch.load(ck_testing)
    net.eval()
    net.load_state_dict(saved['model_state_dict'])
    predict_val(net, dataloader, saved['epoch'])

def evaluate_test_saved(net, dataloader):
    ck_testing=args.ck_testing
    saved = torch.load(ck_testing)
    net.eval()
    net.load_state_dict(saved['model_state_dict'])
    predict_test(net, dataloader, saved['epoch'])
    
    
def main():
    val_data, test_data = build_dataloaders_test(UTI)

    print('Building model to testing: ')    
    net = PReFIL(195)
    net.cuda()
    
    print('Validating....')   
    evaluate_val_saved(net, val_data)
    
    print('Testing....')
    evaluate_test_saved(net, test_data)

if __name__ == "__main__":
    main()
