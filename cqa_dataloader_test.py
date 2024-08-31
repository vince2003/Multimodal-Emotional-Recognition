import json
import os
#from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
#from PIL import Image
#from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import pdb
from trial_mode import trial_arg

#============for bert================
#import os
#import torch
#import pandas as pd
#from scipy import stats
#import numpy as np

#from tqdm import tqdm
#import torch.nn as nn
#import joblib
#import pdb

import transformers
#from transformers import AdamW, get_linear_schedule_with_warmup
#import sys
import traceback

#import pandas as pd
#import transformers
#from sklearn import model_selection

#-------albumentation------
import cv2


#---------evoke_expression------
#from transformers import Wav2Vec2Tokenizer
import librosa

from config import get_config
args = get_config()
#----------------------addition------------------
name_project, trial_ck, max_epochs, data_subset_train, data_subset_val, shuffle, batch_size, resume, num_workers=trial_arg()
def _init_fn(worker_id):
    np.random.seed(0)
    
    
class CQADataset(Dataset):
    def __init__(self, cqa_data, split, tokenizer, UTI):
        self.cqa_data = cqa_data
        self.split = split
        #self.config = config
        
        #=======for audio=========
        self.tokenizer = tokenizer
        
        #=========chon albmentation=======       
        self.prep = UTI.test_transform          
        

    def __len__(self):
        return len(self.cqa_data)

    def __getitem__(self, index):
        try:
        
            if self.split=='val':
                ans_con = torch.tensor(self.cqa_data[index]['label'])
            else:
                ans_con = torch.rand(195) # label gia
            
            #-----------read Image------------------------------------    
            img_id = self.cqa_data[index]['imgs_path']
            img_path = os.path.join(args.root, args.dataset, 'extract_frame', img_id+'.jpg')
            #img = Image.open(img_path).convert('RGB')        
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
            img_tensor = self.prep(image=img)["image"]
    
            #--------------read Audio---------------------------------
            audio_id = self.cqa_data[index]['wavs_path']
            audio_path = os.path.join(args.root, args.dataset, 'extract_wav_timestamp', audio_id+'.wav')
            
            audio_input, _ = librosa.load(audio_path, sr=16000)
            
            token_audio = self.tokenizer(audio_input, return_tensors="pt", padding="longest").input_values 
            
            #pdb.set_trace()
            
        except Exception:
            print("\n----error----")
            traceback.print_exc()
            pdb.set_trace()
        
        #pdb.set_trace()
        
        return token_audio, ans_con, img_tensor, img_id





def collate_batch(data_batch):
    #pdb.set_trace()
    data_batch.sort(key=lambda x: x[-1], reverse=True)
    return torch.utils.data.dataloader.default_collate(data_batch)




# %%
def build_dataloaders_test(UTI):
    #pdb.set_trace()
    tokenizer = transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
 
#--------------------------------------------FOR VAL-----------------------------------------------------------------
    val_datasets = []
    for split in UTI.val_filenames:        
        cqa_val_data = json.load(open(os.path.join(args.root, args.dataset, 'qa', UTI.val_filenames[split])))
        n = int(data_subset_val * len(cqa_val_data))
        cqa_val_data = cqa_val_data[:n]          
        
        val_datasets.append(CQADataset(cqa_data=cqa_val_data,
                                       split=split,
                                       tokenizer=tokenizer,                                       
                                       UTI=UTI
                                       ))

    val_dataloaders = []
    for vds in val_datasets:
        val_dataloaders.append(DataLoader(vds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
                                          num_workers=20, worker_init_fn=_init_fn))   
    
    
    
#--------------------------------------------FOR TEST-----------------------------------------------------------------
    test_datasets = []
    for split in UTI.test_filenames:
        cqa_test_data = json.load(open(os.path.join(args.root, args.dataset, 'qa', UTI.test_filenames[split])))
        n = int(data_subset_val * len(cqa_test_data))
        cqa_test_data = cqa_test_data[:n] 
        
        test_datasets.append(CQADataset(cqa_data=cqa_test_data,
                                       split=split,
                                       tokenizer=tokenizer,                                       
                                       UTI=UTI
                                       ))
    test_dataloaders = []
    for tds in test_datasets:
        test_dataloaders.append(DataLoader(tds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
                                           num_workers=20, worker_init_fn=_init_fn))

    return val_dataloaders, test_dataloaders


def main():
    pass


if __name__ == '__main___':
    main()
