import argparse

#from torchvision import transforms
#import model

#import model_bert

#----------Config truc tiep tu terminal------------
def str2bool(v):
  return v.lower() in ('true', '1')

def get_config():
  config = parser.parse_args()
  return config  # Training settings

#----init argument
parser = argparse.ArgumentParser()

#------------------------------------------------------------------------------
# Operation
operate_arg = parser.add_argument_group('Operation')
operate_arg.add_argument('--evaluate', default=False, type=str2bool)
operate_arg.add_argument('--resume', default=False, type=str2bool)
operate_arg.add_argument('--expt_name', default='FigureQA', type=str)
operate_arg.add_argument('--trial_mode', default=True, type=str2bool)
#operate_arg.add_argument('--data_root', default='data', type=str)
operate_arg.add_argument('--ck_testing', default='./checkpoint/3johowbu/4.pth', type=str)     

operate_arg.add_argument('--bert_path', default="../input/robert_base", type=str)

operate_arg.add_argument('--num_workers', default=20, type=int)

operate_arg.add_argument('--current_direction', default="", type=str)


#------------------------------------------------------------------------------
#Name_project
project_arg = parser.add_argument_group('Name_project')
project_arg.add_argument('--project', default='sample_code_evoke', type=str)
project_arg.add_argument('--name_graph', default='', type=str)
project_arg.add_argument('--folder_ck', default='./ck_spyder/', type=str)
project_arg.add_argument('--resume_ck', default='./checkpoint/3johowbu/4.pth', type=str)                


# Dataset Store Definitions
dataset_arg = parser.add_argument_group('Dataset')
dataset_arg.add_argument('--train_file', default='train_meta_nhe_v4.json', type=str)#'train_token_correct.json', type=str)#
dataset_arg.add_argument('--val_files', default='val_meta_nhe_v4.json', type=str)#'val_token_correct.json', type=str)#
dataset_arg.add_argument('--test_files', default='test_meta_nhe_v4.json', type=str)#'test_processing_correct.json', type=str)#
dataset_arg.add_argument('--dataset', default='FigureQA', type=str)
dataset_arg.add_argument('--high_img', default=224, type=int)
dataset_arg.add_argument('--wide_img', default=224, type=int)
dataset_arg.add_argument('--size_img', default=224, type=int)


# Data and Preprocessing
preprocess_arg = parser.add_argument_group('Preprocessing')
preprocess_arg.add_argument('--root', default='data', type=str)
preprocess_arg.add_argument('--data_subset_train', default=1, type=int)
preprocess_arg.add_argument('--data_subset_val', default=1, type=int)
preprocess_arg.add_argument('--batch_size', default=64, type=int)
preprocess_arg.add_argument('--lut_location', default='', type=str)


#------------------------------------------------------------------------------
# Network
net_arg = parser.add_argument_group('Network')
net_arg.add_argument('--word_emb_dim', default=768, type=int)
net_arg.add_argument('--ques_lstm_out', default=256, type=int)
net_arg.add_argument('--num_hidden_act', default=1024, type=int)
net_arg.add_argument('--num_rf_out', default=256, type=int)
net_arg.add_argument('--num_bimodal_units', default=256, type=int)
net_arg.add_argument('--loss', default='MSE', type=str)
net_arg.add_argument('--image_encoder', default='dense', type=str)
net_arg.add_argument('--dropout_classifier', default=True, type=str2bool)
net_arg.add_argument('--lstm_layers', default=1, type=int)

net_arg.add_argument('--freeze_efficient', default=False, type=str2bool)
net_arg.add_argument('--freeze_wav2vec', default=False, type=str2bool)


# Training/Optimization
training_arg = parser.add_argument_group('Training')
training_arg.add_argument('--test_interval', default=1, type=int)
training_arg.add_argument('--test_every_epoch_after', default=20, type=int)
training_arg.add_argument('--max_epochs', default=8, type=int)
training_arg.add_argument('--overwrite_expt_dir', default=False, type=str2bool)
training_arg.add_argument('--grad_clip', default=50, type=int)

# Parameters for learning rate schedule
lr_arg = parser.add_argument_group('lr')
lr_arg.add_argument('--epoch_interval', default=1, type=int)
lr_arg.add_argument('--lr', default=8e-6, type=float)
lr_arg.add_argument('--lr_decay_step', default=2, type=int)
lr_arg.add_argument('--lr_decay_rate', default=.9, type=float)
lr_arg.add_argument('--warm_up_to_epoch', default=15, type=int)
lr_arg.add_argument('--warm_down_from_epoch', default=20, type=int)

lr_arg.add_argument('--warmup_step', default=True, type=str2bool)



# Loss function
loss_arg = parser.add_argument_group('loss')
loss_arg.add_argument('--bce_w', type=int, default=0)
loss_arg.add_argument('--mse_w', type=int, default=0)

loss_arg.add_argument('--bce_w1', type=int, default=0)
loss_arg.add_argument('--mse_w1', type=int, default=0)

loss_arg.add_argument('--bce_w2', type=int, default=0)
loss_arg.add_argument('--mse_w2', type=int, default=1)

#------------------------------------------------------------------------------
#============Config gian tiep==================================================
# Dataset Store Definitions











