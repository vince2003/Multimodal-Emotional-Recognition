import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

#from blockwise_densenet import DenseNet

#============for bert================
#import os
#import torch
#import pandas as pd
#from scipy import stats
#import numpy as np

#from tqdm import tqdm
#import torch.nn as nn
#import joblib
import pdb

import transformers
#from transformers import AdamW, get_linear_schedule_with_warmup
#import sys

#import pandas as pd
#import transformers
#from sklearn import model_selection

#import torch
#import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

#--------timm-----------
import timm
#from pprint import pprint
#import os
#import torch

#------------config-------------------------
#import config as CONFIG
from config import get_config
args = get_config()
#pdb.set_trace()

#--------evoked_expression------------
#from transformers import Wav2Vec2ForCTC

class LSTM(nn.Module):

    # define all the layers used in model
    def __init__(self, embedding_dim, lstm_units, lstm_layers,
                 bidirectional=False):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_units,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        num_directions = 2 if bidirectional else 1
        #self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, num_classes)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = num_directions
        self.lstm_units = lstm_units


    def init_hidden(self, batch_size):
        h, c = Variable(torch.zeros(self.lstm_layers * self.num_directions,
                                    batch_size, self.lstm_units)).cuda(),\
               Variable(torch.zeros(self.lstm_layers * self.num_directions,
                                    batch_size, self.lstm_units)).cuda()
        return h, c

    def forward(self, embedded, text_lengths):
        batch_size = embedded.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)
        #h_0, c_0 = torch.tensor(h_0).cuda(), torch.tensor(c_0).cuda()
        #pdb.set_trace()

        #embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths,
                                               batch_first=True)
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
        output_unpacked, output_lengths = pad_packed_sequence(output,
                                                              batch_first=True)
        out = output_unpacked[:, -1, :]
        #rel = self.relu(out)
        #dense1 = self.fc1(rel)
        #drop = self.dropout(dense1)
        #preds = self.fc2(drop)
        return out

import pdb
# from torchtext.data import Field 
# from torchtext.data import Dataset, Example
# from torchtext.data import BucketIterator
# from torchtext.vocab import FastText
# from torchtext.vocab import CharNGram

#embedding = FastText('simple')
# from transformers import BertConfig, BertModel
# model = BertModel.from_pretrained('bert-base-uncased')
class ROBERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(ROBERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.RobertaModel.from_pretrained('roberta-base',
                        cache_dir= "../input/robert_base", return_dict=False)
        
        # for param in self.bert.parameters():
        #     param.requires_grad = False        
        #self.bert_drop = nn.Dropout(torch.tensor(0.3))
        #self.out = nn.Linear(768, 30)
        

    def forward(self, ids, mask):
        #pdb.set_trace()
        out, o2 = self.bert(ids, attention_mask=mask)
        #out = self.bert_drop(out)        
        return out
    
    
class Wav2Vec2Tranformer(nn.Module):
    def __init__(self):
        super(Wav2Vec2Tranformer, self).__init__()

        w2v_model = transformers.Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", return_dict=False)
        self.extractor = w2v_model.wav2vec2
        
        # for param in self.bert.parameters():
        #     param.requires_grad = False        
        #self.bert_drop = nn.Dropout(torch.tensor(0.3))
        #self.out = nn.Linear(768, 30)
        

    def forward(self, token_audio):
        #pdb.set_trace()
        out = self.extractor(token_audio)[0]
        #out = self.bert_drop(out)        
        return out


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        
        #self.embedding = embedding
        #self.embedding = nn.Embedding(40000, config.word_emb_dim)
        #pdb.set_trace()
        
        #===========embedding by bert========
        self.au_model = Wav2Vec2Tranformer()
        if args.freeze_wav2vec:
            for param in self.au_model.parameters():
                param.requires_grad = False
        
        # self.lstm = nn.LSTM(input_size=config.word_emb_dim,
        #                     hidden_size=config.ques_lstm_out,
        #                     num_layers=1)
        #pdb.set_trace()
        self.lstm_mullayer = LSTM(
                            embedding_dim=args.word_emb_dim,
                            lstm_units=args.ques_lstm_out,
                            lstm_layers=args.lstm_layers,
                            bidirectional=False
                            )

    def forward(self, token_audio):
        
        #q_embed = self.embedding(ids)
        #q_embed=input_q  
        #pdb.set_trace()
        audio_embedding = self.au_model(token_audio) 
        
        
        #packed = pack_padded_sequence(bert_embedding, lengths=q_len.cpu(), batch_first=True)
        #o, (h, c) = self.lstm(packed)
        q_len=torch.Tensor([audio_embedding.shape[1]]*audio_embedding.shape[0])
        
        out_mulstm = self.lstm_mullayer(audio_embedding, q_len)
        #pdb.set_trace()
        return out_mulstm #c.squeeze(0)    



# class DenseNetEncoder(nn.Module):
#     def __init__(self, densenet_config):
#         super(DenseNetEncoder, self).__init__()
#         self.densenet = DenseNet(block_config=densenet_config).cuda()

#     def forward(self, img):
#         _, dense, final = self.densenet(img)
#         return dense[0], dense[1], final

class DenseNetEncoder(nn.Module):
    def __init__(self):
        super(DenseNetEncoder, self).__init__()
        #self.densenet = DenseNet(block_config=densenet_config).cuda()
        self.efficientnet = timm.create_model('efficientnet_b0', features_only=True, output_stride=16, out_indices=(2, 4), pretrained=True).cuda()

        #pdb.set_trace()
        if args.freeze_efficient:
            for param in self.efficientnet.parameters():
                param.requires_grad = False 
        #self.efficientnet_drop = nn.Dropout(torch.tensor(0.3))
        print('\n------MODEL EFFICIENTET B0-----------')
        
    def forward(self, img):
        feature = self.efficientnet(img)
        #pdb.set_trace()
        #feature=self.efficientnet_drop(feature)
        return feature[1]


class BimodalEmbedding(nn.Module):
    def __init__(self, num_mmc_units, ques_dim, img_dim, num_mmc_layers=4):
        super(BimodalEmbedding, self).__init__()
        self.bn = nn.BatchNorm2d(ques_dim + img_dim)
        self.transform_convs = []
        self.num_mmc_layers = num_mmc_layers
        self.transform_convs.append(nn.Conv2d(ques_dim + img_dim, num_mmc_units,
                                              kernel_size=1))
        self.transform_convs.append(nn.ReLU())
        for i in range(num_mmc_layers - 1):
            self.transform_convs.append(nn.Conv2d(num_mmc_units, num_mmc_units,
                                                  kernel_size=1))
            self.transform_convs.append(nn.ReLU())
        self.transform_convs = nn.Sequential(*self.transform_convs)

    def forward(self, img_feat, ques_feat):
        # Tile ques_vector, concatenate
        
        _, _, nw, nh = img_feat.shape
        _, qdim = ques_feat.shape
        ques_feat = ques_feat.unsqueeze(2)
        ques_tile = ques_feat.repeat(1, 1, nw * nh)
        ques_tile = ques_tile.view(-1, qdim, nw, nh)
        combine_feat = self.bn(torch.cat([img_feat, ques_tile], dim=1))        
        pdb.set_trace()
        bimodal_emb = self.transform_convs(combine_feat)
        return bimodal_emb


class Classifier(nn.Module):
    def __init__(self, num_classes, feat_in):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(feat_in, args.num_hidden_act)
        self.classifier = nn.Linear(args.num_hidden_act, num_classes)
        self.drop = nn.Dropout(0.3)
        self.use_drop = args.dropout_classifier

    def forward(self, bimodal_emb):
        # Tile ques_vector, concatenate
        projection = self.relu(self.lin1(bimodal_emb))
        # if self.use_drop:
        #     projection = self.drop(projection)
        preds = self.classifier(projection)
        #if not (config.loss=='BCE'):
        preds = self.relu(preds) 
            #pdb.set_trace()
        
        return preds

class Classifier1(nn.Module):
    def __init__(self, num_classes):
        super(Classifier1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.drop = nn.Dropout(0.3)
        self.use_drop = args.dropout_classifier

    def forward(self, bimodal_emb):
        # Tile ques_vector, concatenate
        projection = self.relu(self.lin1(bimodal_emb))
        # if self.use_drop:
        #     projection = self.drop(projection)
        preds = self.classifier(projection)
        #if not (config.loss=='BCE'):
        preds = self.relu(preds) 
            #pdb.set_trace()
        
        return preds

class Classifier2(nn.Module):
    def __init__(self, num_classes):
        super(Classifier2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.drop = nn.Dropout(0.3)
        self.use_drop = args.dropout_classifier

    def forward(self, bimodal_emb):
        # Tile ques_vector, concatenate
        projection = self.relu(self.lin1(bimodal_emb))
        # if self.use_drop:
        #     projection = self.drop(projection)
        preds = self.classifier(projection)
        #if not (config.loss=='BCE'):
        preds = self.relu(preds) 
            #pdb.set_trace()
        
        return preds


class RecurrentFusion(nn.Module):
    def __init__(self, num_bigru_units, feat_in):
        super(RecurrentFusion, self).__init__()
        self.bigru = nn.GRU(input_size=feat_in,
                            hidden_size=num_bigru_units,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, mmc_feat):
        _, fs, nw, nh = mmc_feat.shape
        mmc_feat = mmc_feat.view(-1, fs, nw * nh)
        mmc_feat = torch.transpose(mmc_feat, 1, 2)
        output, h = self.bigru(mmc_feat)
        return torch.flatten(torch.transpose(h, 0, 1), start_dim=1)


class BasePReFIL(nn.Module):
    def __init__(self):
        super(BasePReFIL, self).__init__()
        #self.config = config
        #pdb.set_trace()
        self.rnn = AudioEncoder()  # BERT interate into lstm
        self.cnn = DenseNetEncoder()
        #img_dims = config.densenet_dim
        #self.bimodal_low = BimodalEmbedding(config.num_bimodal_units,
                                            #config.ques_lstm_out, 40)
        
        self.bimodal_high = BimodalEmbedding(args.num_bimodal_units,
                                             args.ques_lstm_out, 320)
        self.maxpool_low = nn.MaxPool2d(kernel_size=3, stride=2,
                                        padding=1, dilation=1)
        #self.config = config
        
        #pdb.set_trace()

        # for param in self.bimodal_low.parameters():
        #     param.requires_grad = False 

        # for param in self.bimodal_high.parameters():
        #     param.requires_grad = False 

    @staticmethod
    def flatten_to_2d(mmc_feat):
        return mmc_feat.reshape(-1,
                    mmc_feat.shape[1] * mmc_feat.shape[2] * mmc_feat.shape[3])

    def forward(self, token_audio, img):
        #pdb.set_trace()
        ques_feat = self.rnn(token_audio)
        feat_high = self.cnn(img)
        #pdb.set_trace()
        #feat_low = self.maxpool_low(feat_low)
        #bimodal_feat_low = self.bimodal_low(feat_low, ques_feat)
        bimodal_feat_high = self.bimodal_high(feat_high, ques_feat)
        return bimodal_feat_high


class PReFIL(BasePReFIL):
    def __init__(self, num_ans_classes):
        super(PReFIL, self).__init__()
        #self.rf_low = RecurrentFusion(config.num_rf_out,
                                      #config.num_bimodal_units)
        self.rf_high = RecurrentFusion(args.num_rf_out,
                                       args.num_bimodal_units)
        #self.classifier = Classifier(num_ans_classes, config.num_rf_out * 4, config)
        
        #pdb.set_trace()
        #self.classifier1 = Classifier1(num_ans_classes, config)
        self.classifier2 = Classifier2(num_ans_classes)

        # for param in self.rf_low.parameters():
        #     param.requires_grad = False 
        # for param in self.rf_high.parameters():
        #     param.requires_grad = False 
        # for param in self.classifier.parameters():
        #     param.requires_grad = False 
        # for param in self.classifier1.parameters():
        #     param.requires_grad = False 
        # for param in self.classifier2.parameters():
        #     param.requires_grad = False 

    def forward(self, token_audio, img):
        
        bimodal_feat_high = super(PReFIL, self).forward(token_audio, img)
        #rf_feat_low = self.rf_low(bimodal_feat_low)
        rf_feat_high = self.rf_high(bimodal_feat_high)
        #final_feat = torch.cat([rf_feat_low, rf_feat_high], dim=1)
        #p_con = self.classifier(final_feat, config)
        #p_con1 = self.classifier1(rf_feat_low, config)
        p_con2 = self.classifier2(rf_feat_high)
        #pdb.set_trace()
        return p_con2


def main():
    pass


if __name__ == '__main___':
    main()
