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
import pandas as pd
#from scipy import stats
#import numpy as np

#from tqdm import tqdm
#import torch.nn as nn
#import joblib
import pdb

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
        
        #========================
        
        if self.split == 'train':
            self.prep = UTI.train_transform
        else:
            self.prep = UTI.test_transform         
            

    def __len__(self):
        return len(self.cqa_data)

    def __getitem__(self, index):      

        ans_con = torch.tensor(self.cqa_data[index]['label'])

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
        # except Exception:
        #     print("\n----error----")
        #     traceback.print_exc()
            
        #     token_audio=torch.rand(1, 32000)
        #     ans_con=torch.rand(195)
        #     img_tensor=torch.rand(3, 224, 224)
        #     img_id="ERROR_path"
        #     pdb.set_trace()
            
            #pdb.set_trace()
            # with open(f'error_thing_and_nothing.txt', 'a') as f:
            #       f.write(f'{img_path}\t{e}\n'
            
     
        
        #id_cur=img_id.split('/')[-1]   
        
        # print(audio_input.shape)
        # print(id_cur)
        

        #pdb.set_trace()
        return token_audio, ans_con, img_tensor, img_id
               #torch.tensor(token_type_ids, dtype=torch.long), 
               



def collate_batch(data_batch):
    #pdb.set_trace()
    data_batch.sort(key=lambda x: x[-1], reverse=True)
    return torch.utils.data.dataloader.default_collate(data_batch)



# %%
def build_dataloaders(UTI):
    #pdb.set_trace()
    
    
    cqa_train_data = json.load(open(os.path.join(args.root, args.dataset, 'qa', UTI.train_filename)))


    m = int(data_subset_train * len(cqa_train_data))
    #np.random.seed(666)
    #np.random.shuffle(cqa_train_data)
    cqa_train_data = cqa_train_data[:m]
    
    
    
    tokenizer = transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    
    train_dataset = CQADataset(
        cqa_data=cqa_train_data,
        split='train',
        tokenizer=tokenizer,        
        UTI=UTI
        )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, worker_init_fn=_init_fn)
    
    

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
        #pdb.set_trace()

    val_dataloaders = []
    for vds in val_datasets:
        val_dataloaders.append(DataLoader(vds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
                                          num_workers=num_workers, worker_init_fn=_init_fn))



    return train_dataloader, val_dataloaders


def main():
    pass


if __name__ == '__main___':
    main()
