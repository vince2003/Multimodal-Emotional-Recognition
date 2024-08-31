export CUDA_VISIBLE_DEVICES=0

set -e  # exit when error happen
source activate DVQA_torch38


wandb agent dang/evoked_expression/v2g9w1ww



#python -m compete_slack.py

