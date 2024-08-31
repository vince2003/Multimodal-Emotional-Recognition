import pdb
#import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import get_config
args = get_config()

dataset = args.dataset 

train_file = dict()
#train_file['FigureQA'] = 'train_MSE_BCE_processing_embedding50d_2labels_correct.json'
train_file[dataset] = args.train_file

val_files = dict()
#val_files['FigureQA'] = {'val1': '....', val2': '....', val3': '....'}  # Sample structure of validation
#val_files['FigureQA'] = {}
#val_files['FigureQA'] = {'val1': 'val_BCE_processing_embedding50d_1label_correct.json'}

if args.val_files:
    val_files[dataset] = {'val': args.val_files}
else:
    val_files[dataset] = {}
    

test_files = dict()
#test_files['FigureQA'] = {'test1': '....', test2': '....', test3': '....'}  # Sample structure of test

if args.test_files:
    test_files[dataset] = {'test': args.test_files}
else:    
    test_files[dataset] = {}
#test_files[dataset] = {'test1': 'FigureQA_test1_qa.json'}

transform_combo_train = dict()
transform_combo_test = dict()


transform_combo_train[dataset] = A.Compose(
    [
        #A.SmallestMaxSize(max_size=160),
        #A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=6, p=0.5),
        A.Resize(args.size_img, args.size_img),
        #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        #A.RandomBrightnessContrast(p=0.5),
        #A.ChannelShuffle(p=0.1),      
        #A.GridDistortion(p=0.5),
        #A.HorizontalFlip(p=0.5),     
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


transform_combo_test[dataset] = A.Compose(
    [
        A.Resize(args.size_img, args.size_img),        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


train_filename = train_file[dataset]
val_filenames = val_files[dataset]
test_filenames = test_files[dataset]

train_transform = transform_combo_train[dataset]
test_transform = transform_combo_test[dataset]


lr_decay_epochs = range(args.warm_up_to_epoch, args.warm_down_from_epoch, args.lr_decay_step)
lr_warmup_steps = [0.5 * args.lr, 1.0 * args.lr, 1.0 * args.lr, 1.5 * args.lr, 2.0 * args.lr]
#dropout_classifier = args.dropout_classifier