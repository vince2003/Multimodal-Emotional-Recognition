export CUDA_VISIBLE_DEVICES=1 # chay cho RTX3090
source activate DVQA_torch38

name_project=evoked_expression_final
max_epoch=30
num_workers=16
chay_thu=False
ck="./checkpoint/"




#-----------------------train-----------------------------

python run_cqa.py --project $name_project --name_graph 'bs64_effb0_224' --max_epochs $max_epoch --trial_mode $chay_thu --folder_ck $ck --batch_size 2 --size_img 224 --num_workers $num_workers --data_subset_train 1 --data_subset_val 1 --resume True --freeze_efficient=False --freeze_wav2vec=False --lr=0.001 --lstm_layers=2 --resume_ck './checkpoint/latest.pth'


