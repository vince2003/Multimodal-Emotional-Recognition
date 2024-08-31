
# Multimodal Emotional Recognition

This repository contains an end-to-end model for recognizing emotions from video input. We use the [Weights & Biases (Wandb) platform](https://wandb.ai/site) to manage the project and automate hyperparameter sweeps to find the optimal model configuration.

## Training

To train the model, follow these steps:

1. Run the training script:
   ```bash
   bash train.sh
   ```

2. Execute the training command:
   ```bash
   python run_cqa.py --project $name_project --name_graph 'bs64_effb0_224' --max_epochs $max_epoch --trial_mode $chay_thu --folder_ck $ck --batch_size 2 --size_img 224 --num_workers $num_workers --data_subset_train 1 --data_subset_val 1 --resume True --freeze_efficient=False --freeze_wav2vec=False --lr=0.001 --lstm_layers=2 --resume_ck './checkpoint/latest.pth'
   ```

### Parameter Descriptions

- **`--project`**: The name of the project on the Wandb platform.
- **`--name_graph`**: Specifies the model used to extract visual information.
- **`--trial_mode`**: If set to `true`, the model runs on a smaller dataset, useful for debugging.
- **`--freeze_wav2vec`**: Determines whether to freeze or unfreeze the model responsible for extracting audio information.

## Hyperparameter Sweeping

We use [W&B Sweeps](https://docs.wandb.ai/guides/sweeps) to automate the hyperparameter search process and provide rich, interactive experiment tracking. Popular search methods such as Bayesian optimization, grid search, and random search are used to explore the hyperparameter space. In this project, we use grid search as configured in the "sweep_grid.yaml" file.

To start the sweep agent, run:
```bash
bash sweep_grid.yaml
```

After the sweep completes, the best hyperparameters will be released as defined in the `sweep_grid.yaml` file.

## Testing

To test the model, use the following command:

```bash
bash test.sh
```

The results will be saved in the "result_csv" folder.
